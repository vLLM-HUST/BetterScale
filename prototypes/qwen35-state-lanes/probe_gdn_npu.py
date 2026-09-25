"""Bounded NPU proof: accepted-prefix conv/GDN continuation in two FULL graphs.

Derived from the September24 candidate_state_probe; no model weights or serving hooks.
Real Qwen0.8B TP1 State geometry, realized by LiveInference; separate conv seats
and recurrent candidates. Native NPUGraphs are explicitly reset before close. Independent CPU recurrence checks every live
output and candidate state, not merely graph/eager agreement. Metadata changes
between replays; state stays in its physical K-V pool throughout execution.
"""

import json
import os
from pathlib import Path

import torch
import torch_npu
from livemodule import LiveRuntime, TorchStateBackend, live_runtime
from state import Capacity, Geometry, QwenStateRoot
from vllm_ascend.utils import enable_custom_op

from gdn_candidates import fused_recurrent_gated_delta_rule_fwd


def main():
    assert enable_custom_op()
    torch.npu.set_device(0)
    torch.manual_seed(71)
    torch.set_num_threads(4)
    device = "npu"
    width = 3
    requests, capacity, channels = 5, 12, 6144
    geometry = Geometry.from_config(
        json.loads(Path(__file__).with_name("qwen35-0.8b-text-config.json").read_text())
    )
    with live_runtime(
        LiveRuntime(
            device=device,
            state_backend=TorchStateBackend(device, memory_budget_bytes=512 << 20),
        )
    ):
        root = QwenStateRoot(geometry, Capacity(4, 5, token_pages=32))
    assert not any(s.is_bound for _, s in root.named_states())
    root.activate()
    # Nontrivial resident order; seat4 remains hot/inactive throughout all waves.
    resident_ids = torch.tensor([2, 0, 3, 1, -1])
    physical = resident_ids[:, None] * width + torch.arange(width)[None, :]
    physical[-1].fill_(-1)
    pool, conv = root.target["0"].recurrent.tensor, root.target["0"].conv.tensor
    pool_seed = torch.randn(pool.shape) * 0.01
    conv_seed = (torch.randn(conv.shape) * 0.1).bfloat16()
    pool.copy_(pool_seed)
    conv.copy_(conv_seed)
    weight_cpu = (torch.randn(4, channels) * 0.2).bfloat16()
    weight = weight_cpu.to(device)
    x = torch.empty(capacity, channels, dtype=torch.bfloat16, device=device)
    g = torch.empty(1, capacity, 16, device=device)
    beta = torch.empty_like(g)
    banks = []
    for _ in range(2):
        # Non-unit column stride guards the candidate address calculation.
        slots_storage = torch.full(
            (requests, 2 * width), -1, dtype=torch.int64, device=device
        )
        banks.append(
            dict(
                ids=torch.arange(5, dtype=torch.int64, device=device),
                cu=torch.tensor(
                    [0, width, 2 * width, 3 * width, 4 * width, 4 * width],
                    dtype=torch.int32,
                    device=device,
                ),
                slots=slots_storage[:, ::2],
                conv_slots=resident_ids[:, None].int().contiguous().to(device),
                accepted=torch.ones(requests, dtype=torch.int32, device=device),
            )
        )
        banks[-1]["slots"].copy_(physical)

    def forward(bank):
        y = torch.empty_like(x)
        torch.ops._C_ascend.npu_causal_conv1d_custom(
            y,
            x,
            weight,
            conv_state=conv,
            bias_opt=None,
            query_start_loc_opt=bank["cu"],
            cache_indices_opt=bank["conv_slots"],
            initial_state_mode_opt=None,
            num_accepted_tokens_opt=bank["accepted"],
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=1,
        )
        q, k, v = (
            part.reshape(1, capacity, heads, 128).contiguous()
            for part, heads in zip(y.split([2048, 2048, 2048], -1), [16, 16, 16])
        )
        out, _ = fused_recurrent_gated_delta_rule_fwd(
            q,
            k,
            v,
            g,
            beta,
            128**-0.5,
            pool,
            cu_seqlens=bank["cu"],
            ssm_state_indices=bank["slots"],
            num_accepted_tokens=bank["accepted"],
            use_qk_l2norm_in_kernel=True,
        )
        return y, out

    x.zero_()
    g.fill_(-0.1)
    beta.fill_(0.5)
    graphs, outputs = [], []
    try:
        with torch.inference_mode():
            for bank in banks:
                forward(bank)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    result = forward(bank)
                graphs.append(graph)
                outputs.append(result)
            pool.copy_(pool_seed)
            conv.copy_(conv_seed)
            expected_pool, expected_conv = pool_seed.clone(), conv_seed.clone()
            previous_lengths = [width] * 4
            rows = []
            for wave in range(24):
                lengths = (
                    [width] * 4,
                    [1, 2, width, 0],
                    [width, 1, 2, 1],
                    [2, width, 1, width],
                )[wave % 4]
                accepted = [1 + (wave + i + 2) % previous_lengths[i] for i in range(4)]
                # Explicitly exercise current T=1 selecting old candidate column 2.
                if wave % 4 == 1:
                    accepted[0] = width
                order = list(range(4)) if wave % 2 == 0 else [2, 0, 3, 1]
                ordered_lengths = [lengths[i] for i in order] + [0]
                ordered_accepted = [accepted[i] for i in order] + [1]
                mapping = physical[order + [4]]
                ends = torch.tensor([0] + ordered_lengths).cumsum(0).int()
                cpu_x = (torch.randn(capacity, channels) * 0.1).bfloat16()
                cpu_g = -torch.rand(1, capacity, 16) * 0.2
                cpu_beta = torch.rand(1, capacity, 16)
                bank = banks[wave % 2]
                bank["cu"].copy_(ends)
                bank["slots"].copy_(mapping)
                bank["conv_slots"].copy_(resident_ids[order + [4], None].int())
                bank["accepted"].copy_(
                    torch.tensor(ordered_accepted, dtype=torch.int32)
                )
                x.copy_(cpu_x)
                g.copy_(cpu_g)
                beta.copy_(cpu_beta)
                graphs[wave % 2].replay()
                torch.npu.synchronize()
                actual_y, actual_o = [t.cpu() for t in outputs[wave % 2]]
                reference_y, reference_o = [], []
                for row, request in enumerate(order):
                    length = lengths[request]
                    if length == 0:
                        continue
                    start = int(ends[row])
                    count = accepted[request]
                    slots = mapping[row]
                    conv_slot = int(resident_ids[request])
                    history = expected_conv[conv_slot, count - 1 : count + 2].clone()
                    initial = history.clone()
                    h = expected_pool[int(slots[count - 1])].clone()
                    for t in range(length):
                        index = start + t
                        window = torch.cat([history, cpu_x[index : index + 1]])
                        y = torch.nn.functional.silu(
                            (window.float() * weight_cpu.float()).sum(0)
                        ).bfloat16()
                        reference_y.append(y)
                        history = window[1:]
                        # Check convolution separately below. Feed its observed BF16
                        # rounding into the independent CPU recurrence so a rare
                        # one-ULP SiLU rounding difference cannot masquerade as a
                        # persistent candidate-state protocol error.
                        q, k, v = [
                            p.reshape(heads, 128).float()
                            for p, heads in zip(
                                actual_y[index].split([2048, 2048, 2048]), [16, 16, 16]
                            )
                        ]
                        q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
                        k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
                        q = q * 128**-0.5
                        h *= cpu_g[0, index].exp()[:, None, None]
                        delta = (v - (h * k[:, :, None]).sum(1)) * cpu_beta[
                            0, index, :, None
                        ]
                        h += k[:, :, None] * delta[:, None, :]
                        reference_o.append((h * q[:, :, None]).sum(1).bfloat16())
                        expected_pool[int(slots[t])] = h
                    expected_conv[conv_slot, :2] = initial[1:]
                    expected_conv[conv_slot, 2 : 2 + length] = cpu_x[
                        start : start + length
                    ]
                    previous_lengths[request] = length
                used = sum(lengths)
                expected_y, expected_o = (
                    torch.stack(reference_y),
                    torch.stack(reference_o),
                )
                # BF16 convolution permits one rounding unit; propagation is checked
                # against a separate CPU history across all 24 waves, never reset.
                torch.testing.assert_close(
                    actual_y[:used], expected_y, rtol=0.02, atol=2e-4
                )
                torch.testing.assert_close(
                    actual_o[0, :used], expected_o, rtol=0.03, atol=3e-4
                )
                actual_pool = pool.cpu()
                state_errors = (
                    (actual_pool - expected_pool).abs().flatten(1).amax(1).tolist()
                )
                print(
                    json.dumps(
                        dict(
                            wave=wave,
                            lengths=lengths,
                            accepted=accepted,
                            state_errors=state_errors,
                        )
                    ),
                    flush=True,
                )
                if max(state_errors) > 5e-4:
                    torch.save(
                        dict(
                            actual=actual_pool,
                            expected=expected_pool,
                            wave=wave,
                            lengths=lengths,
                            accepted=accepted,
                        ),
                        Path(os.environ["CAPSULE"], "state-diagnostic.pt"),
                    )
                torch.testing.assert_close(
                    actual_pool, expected_pool, rtol=1e-4, atol=1e-7
                )
                torch.testing.assert_close(conv.cpu(), expected_conv, rtol=0, atol=0)
                rows.append(
                    dict(
                        wave=wave,
                        lengths=lengths,
                        accepted=accepted,
                        conv_max=float(
                            (actual_y[:used].float() - expected_y.float()).abs().max()
                        ),
                        output_max=float(
                            (actual_o[0, :used].float() - expected_o.float())
                            .abs()
                            .max()
                        ),
                        state_max=float((actual_pool - expected_pool).abs().max()),
                    )
                )
            receipt = dict(
                passed=True,
                mtp_tokens=width - 1,
                graphs=2,
                waves=rows,
                state_layout="K-V",
                allocator="LiveInference StateTensor",
                resident_seats=5,
                token_pages=32,
                numerical_lanes=50,
                model="Qwen3.5-0.8B",
                scope="one real-geometry GDN leaf, full State allocation; not model/serving integration",
            )
            Path(os.environ["CAPSULE"], "receipt.json").write_text(
                json.dumps(receipt, indent=2)
            )
            print(
                json.dumps(
                    dict(
                        passed=True,
                        waves=len(rows),
                        state_max=max(r["state_max"] for r in rows),
                    )
                ),
                flush=True,
            )
    finally:
        torch.npu.synchronize()
        for graph in graphs:
            graph.reset()
        root.close()


if __name__ == "__main__":
    main()
