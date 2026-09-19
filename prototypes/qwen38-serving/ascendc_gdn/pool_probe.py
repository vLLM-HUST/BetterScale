"""Direct state-pool full GDN pipeline: dynamic replay against native chunk oracle."""

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
# One pinned metadata publication per wave, shared by every GDN layer.
metadata = torch.empty(67, dtype=torch.int64, device="npu")
host_metadata = torch.empty(67, dtype=torch.int64, pin_memory=True)
cu = metadata[:6]
state_meta = metadata[6:16].view(5, 2)
slots = metadata[16:21]
indices = {
    64: metadata[21:45].view(12, 2),
    256: metadata[45:57].view(6, 2),
    1216: metadata[57:67].view(5, 2),
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
from runtime import Kernels

engine = Kernels(os.environ["ASCENDC_GDN_LIB"], T, N, 12, state_pool=True)
bank = torch.randn(8, 24, 128, 128, device="npu", dtype=torch.float32) * 0.01
seed = bank.clone()
clean_seed = seed.clone()


def prepare(lengths, slot_ids, cold):
    assert 0 < len(lengths) <= 4 and all(n > 0 for n in lengths) and sum(lengths) <= T
    assert len(set(slot_ids[: len(lengths)])) == len(lengths)
    assert all(0 <= i < len(bank) for i in slot_ids[: len(lengths)])
    padded = [*lengths, *([0] * (N - len(lengths)))]
    ends = [0]
    for n in padded:
        ends.append(ends[-1] + n)
    flags = (
        [not cold] * N
        if isinstance(cold, bool)
        else [*cold, *([False] * (N - len(cold)))]
    )
    data = [*ends, *[v for pair in zip(slot_ids, flags) for v in pair], *slot_ids]
    for size, dest in indices.items():
        rows = [
            (i, j) for i, n in enumerate(lengths) for j in range((n + size - 1) // size)
        ]
        assert len(rows) <= len(dest)
        rows += [(N - 1, 0)] * (len(dest) - len(rows))
        data.extend(v for pair in rows for v in pair)
    host_metadata.copy_(torch.tensor(data, dtype=torch.int64))
    metadata.copy_(host_metadata, non_blocking=True)
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
    qh, kh, wh, uh, gh = [x.transpose(1, 2).contiguous() for x in (q, k, w, u, g)]
    engine.pool_forward(qh, kh, wh, uh, gh, bank, cu, state_meta, indices[64])
    out = engine.o.transpose(1, 2).contiguous()
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
        ((1, 127, 63, 321), (7, 2, 6, 0, 4), (True, False, True, False)),
    ]:
        seed.copy_(clean_seed)
        case_flags = [not cold] * len(lengths) if isinstance(cold, bool) else list(cold)
        cold_slots = [slot_ids[i] for i, f in enumerate(case_flags) if not f]
        if cold_slots:
            seed[cold_slots] = float("nan")
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
        flags = [not cold] * n if isinstance(cold, bool) else list(cold)
        flag_tensor = torch.tensor(flags, device="npu")[:, None, None, None]
        initial.copy_(torch.where(flag_tensor, initial, 0))
        inputs = {k: v[:, :total].contiguous() for k, v in values.items()}

        device_cu = cpu.to("npu")

        def oracle(initial_arg=None):
            return chunk.chunk_gated_delta_rule(
                **inputs,
                initial_state=initial if initial_arg is None else initial_arg,
                output_final_state=True,
                cu_seqlens=device_cu,
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
        initial.copy_(torch.where(flag_tensor, final, 0))
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
        native_bank = seed.transpose(-1, -2).contiguous()
        native_slots = torch.tensor(slot_ids[:n], dtype=torch.int64, device="npu")

        from vllm_ascend.ops.triton.fla.utils import clear_ssm_states

        def native_stateful():
            gathered = native_bank[native_slots].transpose(-1, -2).contiguous()
            clear_ssm_states(gathered, flag_tensor.flatten())
            output, final_state = oracle(gathered)
            native_bank[native_slots] = final_state.transpose(-1, -2).contiguous()
            return output

        native_stateful()
        torch.npu.synchronize()
        stateful_graph = torch.npu.NPUGraph()
        with torch.npu.graph(stateful_graph):
            native_stateful()
        # Actual donor role split: recurrent decode prefix, chunk prefill tail.
        d = 0
        while d < n and lengths[d] == 1 and flags[d]:
            d += 1
        decode_slots = native_slots[:d].to(torch.int32)
        actual_lengths = torch.tensor([0] + [1] * d, dtype=torch.int32, device="npu")
        tail_inputs = {k: v[:, d:] for k, v in inputs.items()}
        tail_cpu = cpu[d:] - d
        tail_cu = tail_cpu.to("npu")
        tail_meta = (
            _build_non_spec_chunked_prefill_metadata(
                builder, tail_cpu, torch.device("npu")
            )
            if d < n
            else None
        )

        def native_roles():
            dec = None
            if d:
                dec = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
                    query=inputs["q"][:, :d].squeeze(0),
                    key=inputs["k"][:, :d].squeeze(0),
                    value=inputs["v"][:, :d].squeeze(0),
                    g=inputs["g"][:, :d].squeeze(0),
                    beta=inputs["beta"][:, :d].squeeze(0),
                    state=native_bank,
                    scale=128**-0.5,
                    actual_seq_lengths=actual_lengths,
                    ssm_state_indices=decode_slots,
                ).unsqueeze(0)
            if d < n:
                gathered = native_bank[native_slots[d:]].transpose(-1, -2).contiguous()
                clear_ssm_states(gathered, flag_tensor.flatten()[d:])
                pre, final_state = chunk.chunk_gated_delta_rule(
                    **tail_inputs,
                    initial_state=gathered,
                    output_final_state=True,
                    cu_seqlens=tail_cu,
                    prebuilt_meta=tail_meta,
                    head_first=False,
                    use_qk_l2norm_in_kernel=False,
                )
                native_bank[native_slots[d:]] = final_state.transpose(
                    -1, -2
                ).contiguous()
                return torch.cat([dec, pre], dim=1) if d else pre
            return dec

        native_roles()
        torch.npu.synchronize()
        roles_graph = torch.npu.NPUGraph()
        with torch.npu.graph(roles_graph):
            role_out = native_roles()
        bank.copy_(seed)
        native_bank.copy_(seed.transpose(-1, -2))
        graph.replay()
        roles_graph.replay()
        torch.npu.synchronize()
        row["native_role_checks"] = {
            "output": compare(output[:, :total], role_out),
            "state": compare(bank, native_bank.transpose(-1, -2)),
        }
        row["decode_prefix"] = d
        candidate_graph = graph
        if d == n:
            from decode_kv import fused_recurrent_gated_delta_rule_fwd as kv_decode

            def pure_decode():
                return kv_decode(
                    **values,
                    scale=128**-0.5,
                    initial_state=bank,
                    inplace_final_state=True,
                    cu_seqlens=cu[:5],
                    ssm_state_indices=slots[:4],
                )[0]

            pure_decode()
            torch.npu.synchronize()
            candidate_graph = torch.npu.NPUGraph()
            with torch.npu.graph(candidate_graph):
                decode_out = pure_decode()
            bank.copy_(seed)
            native_bank.copy_(seed.transpose(-1, -2))
            candidate_graph.replay()
            roles_graph.replay()
            torch.npu.synchronize()
            row["pure_decode_checks"] = {
                "output": compare(decode_out[:, :total], role_out),
                "state": compare(bank, native_bank.transpose(-1, -2)),
            }
        row["role_policy_graph_ms"] = {"candidate": [], "native": []}
        for name in ["native", "candidate", "candidate", "native"] * 2:
            row["role_policy_graph_ms"][name].append(
                measure(candidate_graph if name == "candidate" else roles_graph)
            )
        times = {"pool": [], "native_stateful": [], "native_compute_only": []}
        for name in ["native_stateful", "pool", "pool", "native_stateful"] * 2:
            times[name].append(measure(graph if name == "pool" else stateful_graph))
        times["native_compute_only"].append(measure(control))
        row["matched_graph_ms"] = times
        receipt["cases"].append(row)
        save()
    receipt["status"] = (
        "PASS"
        if all(
            c["close"] and c["finite"]
            for r in receipt["cases"]
            for checkset in [
                r["checks"],
                r["native_role_checks"],
                r.get("pure_decode_checks", {}),
            ]
            for c in checkset.values()
        )
        else "NUMERICAL_MISMATCH"
    )
    save()
print(json.dumps(receipt), flush=True)
if receipt["status"] != "PASS":
    raise SystemExit(1)
