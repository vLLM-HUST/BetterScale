"""Kernel-only dynamic packed GDN: one graph, varying lengths/counts/state slots.

Uses existing Ascend Triton H/O kernels instead of the host-list AscendC H/O
entrypoints. No service/runner changes, no model load, TP2-local head geometry.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op

assert enable_custom_op()
torch.npu.set_device(0)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

init_device_properties_triton()
from vllm_ascend.ops.triton.fla import chunk
from vllm_ascend.ops.gdn_attn_builder import _build_non_spec_chunked_prefill_metadata

chunk.get_forward_context = lambda: NS(attn_metadata=NS(num_decodes=0))
chunk.get_pcp_group = lambda: NS(world_size=1)
builder = NS(
    vllm_config=NS(
        model_config=NS(hf_text_config=NS(linear_num_value_heads=48)),
        parallel_config=NS(tensor_parallel_size=2),
    )
)
root = Path(os.environ["CAPSULE"])
receipt = dict(
    status="RUNNING",
    capacity_tokens=512,
    capacity_requests=4,
    head_geometry=dict(qk=8, v=24, kdim=128, vdim=128),
    cases=[],
)


def save():
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2))


def compare(a, b):
    d = (a.float() - b.float()).abs()
    return dict(
        close=bool(torch.allclose(a, b, atol=0.01, rtol=0.01)),
        max_abs=float(d.max()),
        rms=float(d.square().mean().sqrt()),
        finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
    )


T, N = 512, 5  # fifth request is a permanent empty sentinel for unused chunk tasks
cu = torch.zeros(N + 1, dtype=torch.int64, device="npu")
offsets = torch.zeros(N + 1, dtype=torch.int64, device="npu")
slots = torch.arange(N, dtype=torch.int64, device="npu")
active = torch.zeros(N, dtype=torch.bool, device="npu")
continuing = torch.ones(N, dtype=torch.bool, device="npu")
indices = {
    64: torch.empty((12, 2), dtype=torch.int64, device="npu"),
    256: torch.empty((6, 2), dtype=torch.int64, device="npu"),
    1216: torch.empty((5, 2), dtype=torch.int64, device="npu"),
}
torch.manual_seed(173)
values = dict(
    q=torch.randn(1, T, 8, 128, device="npu", dtype=torch.bfloat16),
    k=torch.randn(1, T, 8, 128, device="npu", dtype=torch.bfloat16),
    v=torch.randn(1, T, 24, 128, device="npu", dtype=torch.bfloat16),
    g=-torch.rand(1, T, 24, device="npu") * 0.1,
    beta=torch.rand(1, T, 24, device="npu", dtype=torch.bfloat16),
)
values["q"] = torch.nn.functional.normalize(values["q"].float(), dim=-1).to(
    torch.bfloat16
)
values["k"] = torch.nn.functional.normalize(values["k"].float(), dim=-1).to(
    torch.bfloat16
)
bank = torch.randn(8, 24, 128, 128, device="npu", dtype=torch.float32) * 0.01
seed = bank.clone()


def prepare(lengths, slot_ids, cold):
    padded = [*lengths, *([0] * (N - len(lengths)))]
    ends = [0]
    chunks = [0]
    for length in padded:
        ends.append(ends[-1] + length)
        chunks.append(chunks[-1] + (length + 63) // 64)
    cu.copy_(torch.tensor(ends, dtype=torch.int64, device="npu"))
    offsets.copy_(torch.tensor(chunks, dtype=torch.int64, device="npu"))
    slots.copy_(torch.tensor(slot_ids, dtype=torch.int64, device="npu"))
    active.copy_(torch.tensor([i < len(lengths) for i in range(N)], device="npu"))
    continuing.fill_(not cold)
    for size, dest in indices.items():
        rows = [
            (i, j) for i, n in enumerate(lengths) for j in range((n + size - 1) // size)
        ]
        assert len(rows) <= len(dest)
        rows += [(N - 1, 0)] * (len(dest) - len(rows))
        dest.copy_(torch.tensor(rows, dtype=torch.int64, device="npu"))
    return ends


def dynamic():
    q, k, v, g, beta = [values[n] for n in ("q", "k", "v", "g", "beta")]
    g = chunk.chunk_local_cumsum(
        g, chunk_size=64, cu_seqlens=cu, block_indices=indices[256]
    )
    a = chunk.chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g_cumsum=g,
        cu_seqlens=cu,
        chunk_indices=indices[64],
        output_dtype=torch.float32,
    )
    a = chunk.solve_tril(
        a,
        cu_seqlens=cu,
        chunk_indices_large_block=indices[1216],
        chunk_indices_bt=indices[64],
        output_dtype=k.dtype,
    )
    w, u = chunk.recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=a, g_cumsum=g, cu_seqlens=cu, chunk_indices=indices[64]
    )
    before = bank[slots]
    initial = torch.where(continuing[:, None, None, None], before, 0)
    h, vnew, final = chunk.chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial,
        output_final_state=True,
        cu_seqlens=cu,
        chunk_indices=indices[64],
        chunk_offsets=offsets,
    )
    out = chunk.chunk_fwd_o(
        q=q,
        k=k,
        v=vnew,
        h=h,
        g=g,
        scale=128**-0.5,
        cu_seqlens=cu,
        chunk_offsets=offsets,
    )
    bank.index_copy_(0, slots, torch.where(active[:, None, None, None], final, before))
    return out


def measure(graph):
    a = torch.npu.Event(enable_timing=True)
    b = torch.npu.Event(enable_timing=True)
    a.record()
    for _ in range(20):
        graph.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / 20


with torch.inference_mode():
    prepare((512,), (0, 1, 2, 3, 4), False)
    save()
    for _ in range(2):
        dynamic()
    torch.npu.synchronize()
    bank.copy_(seed)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = dynamic()
    receipt["captured"] = True
    save()
    for lengths, slot_ids, cold in [
        ((512,), (0, 1, 2, 3, 4), False),
        ((1, 511), (5, 2, 0, 1, 4), True),
        ((1, 1, 256, 254), (3, 6, 2, 0, 4), False),
        ((129, 63, 1), (6, 0, 5, 1, 4), False),
        ((64, 64, 64, 64), (2, 3, 6, 7, 4), True),
        ((1, 1, 1, 1), (7, 6, 5, 0, 4), False),
        ((512,), (2, 1, 0, 3, 4), False),
    ]:
        ends = prepare(lengths, slot_ids, cold)
        n = len(lengths)
        total = sum(lengths)
        bank.copy_(seed)
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        actual = output[:, :total].clone()
        actual_bank = bank.clone()
        cpu = torch.tensor(ends[: n + 1], dtype=torch.int32)
        meta = _build_non_spec_chunked_prefill_metadata(
            builder, cpu, torch.device("npu")
        )
        initial = seed[list(slot_ids[:n])].clone()
        if cold:
            initial.zero_()
        inputs = {k: v[:, :total].contiguous() for k, v in values.items()}

        def oracle():
            return chunk.chunk_gated_delta_rule(
                **inputs,
                initial_state=initial,
                output_final_state=True,
                cu_seqlens=cpu.to("npu"),
                prebuilt_meta=meta,
                head_first=False,
                use_qk_l2norm_in_kernel=False,
            )

        expected, final = oracle()
        torch.npu.synchronize()
        expected_bank = seed.clone()
        expected_bank[list(slot_ids[:n])] = final
        checks = dict(
            output=compare(actual, expected), state=compare(actual_bank, expected_bank)
        )
        # Same captured graph consumes the just-updated state for a second pass.
        bank.copy_(actual_bank)
        graph.replay()
        torch.npu.synchronize()
        second = output[:, :total].clone()
        second_bank = bank.clone()
        if not cold:
            initial.copy_(final)
        expected2, final2 = oracle()
        torch.npu.synchronize()
        expected_bank[list(slot_ids[:n])] = final2
        checks.update(
            continuation_output=compare(second, expected2),
            continuation_state=compare(second_bank, expected_bank),
        )
        row = dict(lengths=lengths, slots=slot_ids, cold=cold, checks=checks)
        row["dynamic_graph_ms"] = measure(graph)
        # Fixed-shape native graph is a kernel-cost control, not HTTP latency.
        oracle()
        torch.npu.synchronize()
        control = torch.npu.NPUGraph()
        with torch.npu.graph(control):
            ref = oracle()
        row["native_fixed_graph_ms"] = measure(control)
        receipt["cases"].append(row)
        save()
    receipt["status"] = (
        "PASS"
        if all(
            c["close"] and c["finite"]
            for r in receipt["cases"]
            for c in r["checks"].values()
        )
        else "NUMERICAL_MISMATCH"
    )
    save()
print(json.dumps(receipt), flush=True)
if receipt["status"] != "PASS":
    raise SystemExit(1)
