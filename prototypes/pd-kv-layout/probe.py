"""CPU byte-layout prototype, not an attention kernel or Mooncake connector.

Wire object: one layer, K/V plane, global head, sealed token interval [T,D].
Page/rank IDs are local placement only. BF16 payload is opaque uint16.
"""
import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np

PAGE = 128
DIM = 256
LAYERS = 10  # target FA only; draft cache deliberately excluded
SENTINEL = 65535


def validate_map(table, capacity, tokens):
    table = np.asarray(table)
    needed = (tokens + PAGE - 1) // PAGE
    if table.ndim != 1 or len(table) < needed:
        raise ValueError("incomplete page map")
    used = table[:needed]
    if not np.issubdtype(table.dtype, np.integer):
        raise ValueError("page map must be integer")
    if np.any(used < 0) or np.any(used >= capacity) or len(set(used.tolist())) != len(used):
        raise ValueError("invalid/aliased page map")


def spans(start, stop, table):
    while start < stop:
        logical, offset = divmod(start, PAGE)
        count = min(stop - start, PAGE - offset)
        yield int(table[logical]), offset, count
        start += count


def export_head(source, table, start, stop):
    """One TP2 shard: [physical page, token, dimension], no tensor transpose."""
    if not 0 <= start < stop:
        raise ValueError("empty/reversed interval")
    validate_map(table, len(source), stop)
    result = np.empty((stop - start, DIM), dtype=np.uint16)
    cursor = 0
    for page, offset, count in spans(start, stop, table):
        np.copyto(result[cursor:cursor + count], source[page, offset:offset + count])
        cursor += count
    return result


def restore_head(payload, target, table, start, head, layout):
    """Scatter one immutable head object; target may be native or head-major."""
    if payload.ndim != 2 or payload.shape[1] != DIM or payload.dtype != np.uint16:
        raise ValueError("invalid wire payload")
    if start < 0 or head not in (0, 1):
        raise ValueError("invalid interval/head")
    if layout not in ("tp2", "tp1_native", "tp1_head_major"):
        raise ValueError("unknown layout")
    capacity = target.shape[1] if layout == "tp2" else target.shape[0]
    validate_map(table, capacity, start + len(payload))
    cursor = 0
    for page, offset, count in spans(start, start + len(payload), table):
        if layout == "tp2":
            dst = target[head, page, offset:offset + count, :]
        elif layout == "tp1_native":
            dst = target[page, offset:offset + count, head, :]
        else:
            dst = target[page, head, offset:offset + count, :]
        np.copyto(dst, payload[cursor:cursor + count])
        cursor += count


def allocate(layout, pages):
    shapes = {
        "tp2": (2, pages, PAGE, DIM),
        "tp1_native": (pages, PAGE, 2, DIM),
        "tp1_head_major": (pages, 2, PAGE, DIM),
    }
    return np.full(shapes[layout], SENTINEL, dtype=np.uint16)


def logical(target, table, tokens, layout):
    if layout == "tp2":
        data = target[:, table].transpose(1, 2, 0, 3)
    elif layout == "tp1_native":
        data = target[table]
    else:
        data = target[table].transpose(0, 2, 1, 3)
    return data.reshape(-1, 2, DIM)[:tokens]


def benchmark(tokens, repeats):
    """All 10 target layers, K and V. Preallocated CPU destinations.
    Repeated cache-warm copies; no DMA/network/disk, no device overlap claim.
    """
    rng = np.random.default_rng(17)
    pages = (tokens + PAGE - 1) // PAGE
    # Transfer-ready head chunks: [layer, K/V, head, token, dimension].
    wire = rng.integers(0, SENTINEL, (LAYERS, 2, 2, tokens, DIM), dtype=np.uint16)
    layouts = ("tp2", "tp1_native", "tp1_head_major")
    targets = {name: [allocate(name, pages) for _ in range(LAYERS * 2)] for name in layouts}
    table = rng.permutation(pages)
    samples = {name: [] for name in layouts}
    # Rotate execution order to avoid always rewarding the same arm.
    for repeat in range(repeats + 2):
        order = layouts[repeat % 3:] + layouts[:repeat % 3]
        for name in order:
            begin = time.perf_counter_ns()
            for layer in range(LAYERS):
                for plane in range(2):
                    target = targets[name][2 * layer + plane]
                    for head in range(2):
                        restore_head(wire[layer, plane, head], target, table, 0, head, name)
            elapsed = (time.perf_counter_ns() - begin) / 1e6
            if repeat >= 2:
                samples[name].append(elapsed)
    for name in layouts:
        for layer in range(LAYERS):
            for plane in range(2):
                expected = wire[layer, plane].transpose(1, 0, 2)
                np.testing.assert_array_equal(
                    logical(targets[name][2 * layer + plane], table, tokens, name), expected)
    # A separate P paged->wire gather receipt, including output allocation.
    sources = np.full((LAYERS, 2, 2, pages, PAGE, DIM), SENTINEL, dtype=np.uint16)
    for layer in range(LAYERS):
        for plane in range(2):
            for head in range(2):
                cursor = 0
                for page, offset, count in spans(0, tokens, table):
                    sources[layer, plane, head, page, offset:offset + count] = wire[layer, plane, head, cursor:cursor + count]
                    cursor += count
    gather = []
    for repeat in range(repeats + 2):
        begin = time.perf_counter_ns()
        gathered = [export_head(sources[l, p, h], table, 0, tokens)
                    for l in range(LAYERS) for p in range(2) for h in range(2)]
        elapsed = (time.perf_counter_ns() - begin) / 1e6
        if repeat >= 2:
            gather.append(elapsed)
    for got, expected in zip(gathered, wire.reshape(-1, tokens, DIM)):
        np.testing.assert_array_equal(got, expected)
    result = {
        "tokens": tokens,
        "payload_bytes": wire.nbytes,
        "p_gather_alloc_median_ms": statistics.median(gather),
        "arms": {},
    }
    for name in layouts:
        median = statistics.median(samples[name])
        result["arms"][name] = {
            "median_ms": median, "min_ms": min(samples[name]),
            "max_ms": max(samples[name]), "samples_ms": samples[name],
            "payload_GB_s": wire.nbytes / (median * 1e6),
        }
    return result



def benchmark_bulk(tokens, repeats):
    """Contiguous-page lower-complexity arm: one NumPy copy per full tensor.
    No page-map validation/fragment loop. This isolates layout cost, not a
    fragmented allocator implementation. Payload includes both TP2 ranks.
    """
    if tokens % PAGE:
        return None
    pages = tokens // PAGE
    rng = np.random.default_rng(37)
    source = rng.integers(0, SENTINEL, (LAYERS, 2, 2, pages, PAGE, DIM), dtype=np.uint16)
    views = {
        "tp2": source,
        "tp1_native": source.transpose(0, 1, 3, 4, 2, 5),
        "tp1_head_major": source.transpose(0, 1, 3, 2, 4, 5),
    }
    targets = {name: np.empty(view.shape, dtype=view.dtype) for name, view in views.items()}
    samples = {name: [] for name in views}
    names = tuple(views)
    for repeat in range(repeats + 2):
        for name in names[repeat % 3:] + names[:repeat % 3]:
            begin = time.perf_counter_ns()
            np.copyto(targets[name], views[name])
            elapsed = (time.perf_counter_ns() - begin) / 1e6
            if repeat >= 2:
                samples[name].append(elapsed)
    for name in views:
        np.testing.assert_array_equal(targets[name], views[name])
    return {"tokens": tokens, "payload_bytes": source.nbytes,
            "arms": {name: {"median_ms": statistics.median(values), "samples_ms": values}
                     for name, values in samples.items()}}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 127, 128, 129, 1024, 4096, 16384])
    args = parser.parse_args()
    if args.repeats < 1 or any(t < 1 for t in args.tokens):
        parser.error("positive tokens/repeats required")
    result = {
        "scope": "CPU NumPy uint16 byte copies; target FA only; not DMA or kernel performance",
        "numpy": np.__version__, "machine": platform.machine(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "page_tokens": PAGE,
        "kv_heads": 2, "head_dim": DIM, "target_layers": LAYERS,
        "results": [benchmark(t, args.repeats) for t in args.tokens],
        "bulk_results": [benchmark_bulk(t, args.repeats) for t in args.tokens if t % PAGE == 0],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([{"tokens": r["tokens"], "gather_ms": r["p_gather_alloc_median_ms"],
                      **{k: v["median_ms"] for k, v in r["arms"].items()}}
                     for r in result["results"]], indent=2))
