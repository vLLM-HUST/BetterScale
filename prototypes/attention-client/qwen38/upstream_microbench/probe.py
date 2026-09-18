"""Matched Qwen38 leaves; no model weights, no installed runtime changes."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

p = argparse.ArgumentParser()
p.add_argument("--device", required=True)
p.add_argument("--upstream", type=Path, required=True)
p.add_argument(
    "--kind",
    choices=("indexer", "attention", "prefill", "hc", "expansion"),
    required=True,
)
p.add_argument("--smoke", action="store_true")
p.add_argument("--family", choices=("sglang", "ascend"), default="sglang")
a = p.parse_args()
assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == a.device
from livemodule.arch.ascend._native.package import activate_native_package

activate_native_package(load_extension=False)
import torch
import torch_npu

torch.set_num_threads(2)
torch.npu.set_device(0)
activate_native_package(load_extension=True)
torch.manual_seed(3818)


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, a.upstream / "sglang" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_ascend():
    # Pure import adaptation only: preserve pinned kernel/wrapper source.
    source = (a.upstream / "ascend" / "qsa.py").read_text()
    old = "from vllm.triton_utils import HAS_TRITON, tl, triton"
    assert source.count(old) == 1
    source = source.replace(
        old, "import triton\nimport triton.language as tl\nHAS_TRITON = True"
    )
    target = Path("ascend_qsa_standalone.py").absolute()
    target.write_text(source)
    spec = importlib.util.spec_from_file_location("ascend_qsa_standalone", target)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def emit(**record):
    print(json.dumps(record), flush=True)


def capture(fn):
    for _ in range(2):
        fn()
    torch.npu.synchronize()
    before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream), torch.npu.graph(graph):
        result = fn()
    stream.synchronize()
    peak = torch.npu.max_memory_allocated() - before
    return graph, result, peak


def compare_times(arms):
    samples = {name: [] for name in arms}
    for repeat in range(7):
        names = list(arms)
        if repeat % 2:
            names.reverse()
        for name in names:
            graph = arms[name][0]
            begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
                enable_timing=True
            )
            begin.record()
            for _ in range(3):
                graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) / 3)
    return {
        name: dict(
            median_ms=statistics.median(v),
            samples_ms=v,
            capture_peak_allocated_bytes=arms[name][2],
        )
        for name, v in samples.items()
    }


def attention():
    from livemodule.arch.ascend.llm.qwen38.qsa_attention import AscendQwen38QSAAttention

    sparse = load("sparse_attention") if a.family == "sglang" else load_ascend()
    current = AscendQwen38QSAAttention()
    cases = (
        [(1, 1, 8192, 2051)]
        if a.smoke
        else [
            (rows, heads, context, selected)
            for heads in (1, 2)
            for rows, context, selected in (
                (1, 8192, 2051),
                (4, 65536, 2051),
                (32, 8192, 2051),
                (128, 65536, 2051),
                (512, 8192, 2051),
                (128, 8192, 128),
            )
        ]
    )
    for rows, heads, context, selected in cases:
        pages = context // 64
        cache = torch.randn(pages, 64, heads, 256, device="npu", dtype=torch.bfloat16)
        value = torch.randn_like(cache)
        query = torch.randn(
            rows, 1, heads * 12, 256, device="npu", dtype=torch.bfloat16
        )
        table = torch.stack(
            [torch.randperm(pages, device="npu") for _ in range(2)]
        ).int()
        requests = torch.arange(rows, device="npu", dtype=torch.int64) % 2
        # Distinct sparse logical rows; one masked entry plus a short final row.
        indices = torch.randperm(context, device="npu")[:selected].int().repeat(rows, 1)
        indices[:, 3] = -1
        counts = torch.full((rows, 1), selected, device="npu", dtype=torch.int32)
        if rows > 1:
            counts[-1] = 0
        packed = torch.cat((indices, counts), -1)
        columns = torch.arange(selected, device="npu")

        def slots():
            logical = packed[:, :-1].long()
            safe = logical.clamp_min(0)
            physical = table[requests[:, None], safe // 64].long() * 64 + safe % 64
            return torch.where(
                (logical >= 0) & (columns[None] < packed[:, -1:]), physical, -1
            ).int()

        def baseline():
            return current.decode_paged(
                query,
                cache,
                value,
                packed,
                table,
                leading_pages=0,
                request_rows=requests,
                scale=256**-0.5,
            )

        physical = slots()

        def candidate():
            return sparse.sparse_attention(
                query[:, 0], cache.flatten(0, 1), value.flatten(0, 1), slots()
            )[:, None]

        def leaf():
            return sparse.sparse_attention(
                query[:, 0], cache.flatten(0, 1), value.flatten(0, 1), physical
            )[:, None]

        if a.family == "sglang":
            functions = (
                ("current_gather_fia", baseline),
                ("sgl_including_slots", candidate),
                ("sgl_ready_slots", leaf),
            )
        else:
            req32 = requests.int()
            ready_indices = torch.where(
                columns[None] < packed[:, -1:], packed[:, :-1], -1
            ).contiguous()

            def ascend_ready():
                return sparse.qsa_sparse_paged_attention(
                    query[:, 0], cache, value, ready_indices, table, req32
                )[:, None]

            def ascend_adapted():
                valid_indices = torch.where(
                    columns[None] < packed[:, -1:], packed[:, :-1], -1
                ).contiguous()
                return sparse.qsa_sparse_paged_attention(
                    query[:, 0], cache, value, valid_indices, table, requests.int()
                )[:, None]

            functions = (
                ("current_gather_fia", baseline),
                ("ascend_including_metadata", ascend_adapted),
                ("ascend_ready_metadata", ascend_ready),
            )
        arms = {name: capture(fn) for name, fn in functions}
        errors = []
        for generation in range(2):
            query.copy_(torch.randn_like(query))
            value.copy_(torch.randn_like(value))
            for graph, _, _ in arms.values():
                graph.replay()
            torch.npu.synchronize()
            # Independent FP32 CPU oracle on <=3 rows, including all-masked row.
            chosen = sorted(set([0, min(1, rows - 1), rows - 1]))
            cpu_slots = physical[chosen].cpu().long()
            q = query[chosen, 0].cpu().float().reshape(len(chosen), heads, 12, 256)
            k = (
                cache.flatten(0, 1)[physical[chosen].long().clamp_min(0)]
                .cpu()
                .float()
                .permute(0, 2, 1, 3)
            )
            v = (
                value.flatten(0, 1)[physical[chosen].long().clamp_min(0)]
                .cpu()
                .float()
                .permute(0, 2, 1, 3)
            )
            mask = cpu_slots >= 0
            scores = torch.matmul(q, k.transpose(-1, -2)) * 256**-0.5
            probs = torch.softmax(
                scores.masked_fill(~mask[:, None, None], -float("inf")), dim=-1
            ).nan_to_num()
            reference = torch.matmul(probs, v).reshape(len(chosen), 1, heads * 12, 256)
            for name, (_, out, _) in arms.items():
                got = out[chosen].cpu().float()
                rel = float(
                    torch.linalg.vector_norm(got - reference)
                    / torch.linalg.vector_norm(reference).clamp_min(1e-12)
                )
                maximum = float((got - reference).abs().max())
                assert torch.isfinite(got).all() and rel < 0.015, (name, rel, maximum)
                if rows > 1:
                    assert torch.count_nonzero(out[-1]).item() == 0
                errors.append(
                    dict(
                        generation=generation,
                        arm=name,
                        relative_l2=rel,
                        max_abs=maximum,
                    )
                )
        emit(
            kind="attention",
            family=a.family,
            rows=rows,
            kv_heads=heads,
            context=context,
            selected=selected,
            status="PASS",
            errors=errors,
            timings=compare_times(arms),
        )
        for graph, _, _ in arms.values():
            graph.reset()
        del arms
        torch.npu.empty_cache()


def indexer():
    from livemodule.arch.ascend.llm.qwen38.qsa_indexer import AscendQwen38QSAIndexer

    current = AscendQwen38QSAIndexer()
    mqa = load("mqa")
    cases = (
        [(4, 8192)]
        if a.smoke
        else [
            (r, c)
            for r, c in (
                (1, 8192),
                (4, 65536),
                (32, 8192),
                (128, 65536),
                (512, 8192),
                (2048, 65536),
            )
        ]
    )
    for rows, context in cases:
        # Identical page192 layout meets both implementations' native ABI.
        width = context // 4
        pages = (width + 191) // 192
        cache = torch.randn(pages, 192, 1, 128, device="npu", dtype=torch.bfloat16)
        q = torch.randn(rows, 1, 4, 128, device="npu", dtype=torch.bfloat16)
        table = torch.randperm(pages, device="npu").int()[None].repeat(rows, 1)
        lengths = torch.full((rows,), width, device="npu", dtype=torch.int32)
        ones = torch.ones(rows, device="npu", dtype=torch.int32)
        cumulative = torch.arange(1, rows + 1, device="npu", dtype=torch.int32)
        weights = torch.full((rows, 4), 128**-0.5, device="npu", dtype=torch.float32)

        def old():
            return current.select_decode_groups(
                q, cache, compressed_lengths=lengths, block_table=table, block_topk=512
            )

        def new():
            return torch.ops.npu.npu_lightning_indexer.default(
                q[:, 0],
                cache,
                weights,
                actual_seq_lengths_query=cumulative,
                actual_seq_lengths_key=lengths,
                block_table=table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=512,
                sparse_mode=0,
                pre_tokens=9223372036854775807,
                next_tokens=9223372036854775807,
                return_value=False,
            )[0].reshape(rows, 512)

        def sgl():
            scores = mqa.mqa_decode(q[:, 0], cache, table, lengths, width)
            return scores.topk(512, dim=-1).indices.int()

        funcs = {"current_bsnd": old, "native_tnd": new}
        if rows <= 128:
            funcs["sgl_mqa_topk"] = sgl
        arms = {}
        for name, fn in funcs.items():
            try:
                arms[name] = capture(fn)
            except Exception as error:
                emit(
                    kind="indexer",
                    rows=rows,
                    context=context,
                    arm=name,
                    status="UNSUPPORTED",
                    error=str(error)[:1600],
                )
        agreements = []
        for generation in range(2):
            q.copy_(torch.randn_like(q))
            for graph, _, _ in arms.values():
                graph.replay()
            torch.npu.synchronize()
            base = next(iter(arms.values()))[1].sort(-1).values
            for name, (_, out, _) in arms.items():
                # Ranking is unused after expansion; selected set is the contract.
                got = out.sort(-1).values
                disagreements = int((got != base).any(-1).sum())
                agreements.append(
                    dict(
                        generation=generation,
                        arm=name,
                        rows_with_different_set=disagreements,
                    )
                )
        emit(
            kind="indexer",
            rows=rows,
            context=context,
            status="MEASURED_NOT_PROMOTED",
            agreements=agreements,
            timings=compare_times(arms),
        )
        for graph, _, _ in arms.values():
            graph.reset()
        del arms
        torch.npu.empty_cache()


def prefill():
    from livemodule.arch.ascend.llm.qwen38.qsa_indexer import AscendQwen38QSAIndexer

    current = AscendQwen38QSAIndexer()
    cases = (
        [(128, 8192)]
        if a.smoke
        else [(128, 8192), (512, 8192), (2048, 8192), (4096, 65536)]
    )
    for rows, context in cases:
        width = context // 4
        pages = (width + 191) // 192
        cache = torch.randn(pages, 192, 1, 128, device="npu", dtype=torch.bfloat16)
        query = torch.randn(1, rows, 4, 128, device="npu", dtype=torch.bfloat16)
        table = torch.randperm(pages, device="npu").int()[None]
        starts = torch.tensor([context - rows], device="npu", dtype=torch.int64)
        lengths = torch.tensor([rows], device="npu", dtype=torch.int32)
        positions = torch.arange(
            context - rows, context, device="npu", dtype=torch.int64
        )[None]

        # Include each implementation's layout/metadata construction in capture.
        def old():
            slots, ql, kl, bt, inverse = current.prepare_prefill_lanes(
                starts, lengths, positions, table, ratio=4, leading_pages=0
            )
            q = query.flatten(0, 1).index_select(0, slots).reshape(4, rows // 4, 4, 128)
            result = current.select_prefill_groups(
                q,
                cache,
                query_lengths=ql,
                compressed_lengths=kl,
                block_table=bt,
                block_topk=512,
            )
            return result.reshape(rows, 512).index_select(0, inverse)

        def new(unit_weights=False):
            visible = ((positions.flatten() + 1) // 4).int()
            bt = table.expand(rows, -1).contiguous()
            cumulative = torch.arange(1, rows + 1, device="npu", dtype=torch.int32)
            weights = (
                torch.ones((rows, 4), device="npu", dtype=torch.bfloat16)
                if unit_weights
                else torch.full((rows, 4), 128**-0.5, device="npu", dtype=torch.float32)
            )
            return torch.ops.npu.npu_lightning_indexer.default(
                query[0],
                cache,
                weights,
                actual_seq_lengths_query=cumulative,
                actual_seq_lengths_key=visible,
                block_table=bt,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=512,
                sparse_mode=0,
                pre_tokens=9223372036854775807,
                next_tokens=9223372036854775807,
                return_value=False,
            )[0].reshape(rows, 512)

        arms = {
            name: capture(fn)
            for name, fn in (
                ("current_causal_lanes", old),
                ("upstream_row_tnd", new),
                ("tnd_unit_weights", lambda: new(True)),
            )
        }
        differences = []
        for generation in range(2):
            query.copy_(torch.randn_like(query))
            for graph, _, _ in arms.values():
                graph.replay()
            torch.npu.synchronize()
            base = arms["current_causal_lanes"][1].sort(-1).values
            differences.append(
                {
                    name: int((base != value[1].sort(-1).values).any(-1).sum())
                    for name, value in arms.items()
                }
            )
        emit(
            kind="prefill_indexer",
            rows=rows,
            context=context,
            status="MEASURED_NOT_PROMOTED",
            rows_with_different_set=differences,
            timings=compare_times(arms),
        )
        for graph, _, _ in arms.values():
            graph.reset()
        del arms
        torch.npu.empty_cache()


def hc():
    from livemodule.arch.ascend.llm.qwen38.residual import AscendQwen38ResidualBackend
    import torch.nn.functional as F

    from hc_adapter import BorrowedHCLeaves

    current = AscendQwen38ResidualBackend()
    upstream = load("hc")
    borrowed = BorrowedHCLeaves(upstream)
    count, hidden, lowrank = 4, 2560, 320
    for rows in ((4,) if a.smoke else (1, 4, 32)):
        x = torch.randn(rows, count * hidden, device="npu", dtype=torch.bfloat16)
        block = torch.randn(rows, hidden, device="npu", dtype=torch.bfloat16)
        down = (
            torch.randn(lowrank, count * hidden, device="npu", dtype=torch.bfloat16)
            * 0.01
        )
        up = (
            torch.randn(count * hidden, lowrank, device="npu", dtype=torch.bfloat16)
            * 0.01
        )
        inject = (
            torch.randn(count, count * hidden, device="npu", dtype=torch.bfloat16)
            * 0.01
        )
        weight = torch.randn(count * hidden, device="npu", dtype=torch.bfloat16) * 0.01

        def old(backend=current):
            normalized = backend.normalize(x, weight, hidden, 1e-6)
            logits = F.linear(
                backend.silu_scaled(F.linear(normalized, down), count), up
            )
            mixed = backend.mix(logits, normalized, count)
            gates = backend.injection_weights(F.linear(normalized, inject), count)
            return mixed, backend.inject(block, x, gates)

        def new():
            normalized = upstream.grouped_norm(x, weight, hidden, 1e-6)
            return upstream.mix(normalized, down, up, count, hidden), upstream.combine(
                block, x, normalized, inject, count, hidden
            )

        arms = {
            name: capture(fn)
            for name, fn in (
                ("current_hc", old),
                ("sgl_hc", new),
                ("borrowed_norm_mix", lambda: old(borrowed)),
            )
        }
        differences = []
        for generation in range(12):
            x.copy_(torch.randn_like(x))
            for graph, _, _ in arms.values():
                graph.replay()
            torch.npu.synchronize()
            for name in ("sgl_hc", "borrowed_norm_mix"):
                for leaf, (base, got) in enumerate(
                    zip(arms["current_hc"][1], arms[name][1])
                ):
                    rel = float(
                        torch.linalg.vector_norm(base.float() - got.float())
                        / torch.linalg.vector_norm(base.float())
                    )
                    differences.append(
                        dict(
                            generation=generation,
                            arm=name,
                            leaf=leaf,
                            relative_l2=rel,
                            bitwise_equal=torch.equal(base, got),
                        )
                    )
                    assert rel < 0.005, (name, leaf, rel)
        emit(
            kind="hc",
            rows=rows,
            status="PASS",
            differences=differences,
            timings=compare_times(arms),
        )
        for graph, _, _ in arms.values():
            graph.reset()
        del arms
        torch.npu.empty_cache()


def expansion():
    from livemodule.arch.ascend.llm.qwen38.qsa_expand import expand_groups

    upstream = load("expansion")
    ascend = load_ascend()
    for rows in (1, 4, 32, 128):
        groups = torch.stack(
            [torch.randperm(2047, device="npu")[:512] for _ in range(rows)]
        ).int()
        positions = torch.full((rows,), 8190, device="npu", dtype=torch.int64)
        lengths = positions + 1
        request = torch.arange(rows, device="npu", dtype=torch.int32)

        def old():
            return expand_groups(
                groups, positions, compress_ratio=4, token_budget=2048
            )[:, :-1]

        def sgl():
            return upstream.expand_blocks(groups, positions, lengths, 4, 2048)

        def asc():
            return ascend.expand_qsa_block_indices_npu(
                groups, positions, lengths, request, 4, 2048
            )

        arms = {
            name: capture(fn)
            for name, fn in (
                ("current_expand", old),
                ("sgl_expand", sgl),
                ("ascend_expand", asc),
            )
        }
        for generation in range(2):
            groups.copy_(
                torch.stack(
                    [torch.randperm(2047, device="npu")[:512] for _ in range(rows)]
                ).int()
            )
            for graph, _, _ in arms.values():
                graph.replay()
            torch.npu.synchronize()
            for name, (_, out, _) in arms.items():
                assert torch.equal(out, arms["current_expand"][1]), name
        emit(kind="expansion", rows=rows, status="PASS", timings=compare_times(arms))
        for graph, _, _ in arms.values():
            graph.reset()
        del arms
        torch.npu.empty_cache()


emit(
    stage="environment",
    torch=torch.__version__,
    torch_npu=torch_npu.__version__,
    kind=a.kind,
)
{
    "attention": attention,
    "indexer": indexer,
    "prefill": prefill,
    "hc": hc,
    "expansion": expansion,
}[a.kind]()
