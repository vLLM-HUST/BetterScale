"""Local-only dummy differential, changing-input FULL replay, crossed timings."""

import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch
import torch_npu
from kernel import fused, pack


def errors(a, b):
    d = (a.float() - b.float()).abs()
    return dict(
        max_abs=d.max().item(),
        rms=d.square().mean().sqrt().item(),
        ref_rms=b.float().square().mean().sqrt().item(),
    )


def timing(graph):
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(10):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 10


def capture(fn):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = fn()
    graph.replay()
    torch.npu.synchronize()
    return graph, output, torch.npu.max_memory_allocated() - before


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch.manual_seed(1709)
    reports = []
    for m, h, i in [(512, 5120, 13824), (4096, 5120, 13824)]:
        x = torch.randn(m, h, device="npu", dtype=torch.bfloat16)
        w = (torch.randn(2 * i, h, device="npu") / h**0.5).bfloat16()
        down = (torch.randn(h, i, device="npu") / i**0.5).bfloat16()
        wp = pack(w)
        # Guard both sides of output; all views have contiguous row storage.
        guarded = torch.full((m * i + 256,), 19, device="npu", dtype=torch.bfloat16)
        y = guarded[128:-128].view(m, i)
        originals = [t.cpu() for t in (x, w, wp, down)]

        def native():
            return torch_npu.npu_swiglu(torch.mm(x, w.t()))

        ref = native()
        for tile in [(64, 64, 256), (64, 64, 512)]:
            began = time.monotonic()
            fused(x, wp, y, tile)
            torch.npu.synchronize()
            err = errors(y, ref)
            assert err["max_abs"] <= 0.0625 and err["rms"] <= 0.005, err
            assert torch.all(guarded[:128] == 19) and torch.all(guarded[-128:] == 19)
            for t, old in zip((x, w, wp, down), originals):
                assert torch.equal(t.cpu(), old), "input mutation"
            print(
                json.dumps(
                    dict(
                        shape=[m, h, i],
                        tile=tile,
                        compile_s=time.monotonic() - began,
                        errors=err,
                    )
                ),
                flush=True,
            )
            stages = {}
            for label, ref_fn, cand_fn in [
                ("gate_up_swiglu", native, lambda: fused(x, wp, y, tile)),
                (
                    "whole_ffn",
                    lambda: torch.mm(native(), down.t()),
                    lambda: torch.mm(fused(x, wp, y, tile), down.t()),
                ),
            ]:
                ga, oa, ma = capture(ref_fn)
                gb, ob, mb = capture(cand_fn)
                # Both graphs see changed input on same addresses; old content restored.
                x.mul_(0.875)
                ga.replay()
                gb.replay()
                torch.npu.synchronize()
                e = errors(ob, oa)
                assert e["max_abs"] <= 0.0625 and e["rms"] <= 0.005, e
                x.copy_(originals[0])
                ga.replay()
                gb.replay()
                torch.npu.synchronize()
                aa, bb = [], []
                for repeat in range(6):
                    if repeat % 2:
                        bb.append(timing(gb))
                        aa.append(timing(ga))
                    else:
                        aa.append(timing(ga))
                        bb.append(timing(gb))
                stages[label] = dict(
                    native_ms=aa,
                    candidate_ms=bb,
                    native_median_ms=statistics.median(aa),
                    candidate_median_ms=statistics.median(bb),
                    native_capture_extra_bytes=ma,
                    candidate_capture_extra_bytes=mb,
                    changed_input_errors=e,
                )
                del ga, gb, oa, ob
                gc.collect()
                torch.npu.empty_cache()
            reports.append(dict(shape=[m, h, i], tile=tile, errors=err, stages=stages))
            (args.output / "results.json").write_text(json.dumps(reports, indent=2))
            print(json.dumps(reports[-1]), flush=True)
        del x, w, wp, down, guarded, y, ref, originals
        gc.collect()
        torch.npu.empty_cache()
    (args.output / "complete.json").write_text(json.dumps(dict(status="PASS")))


if __name__ == "__main__":
    main()
