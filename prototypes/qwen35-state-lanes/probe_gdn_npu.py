"""Owned graph lifecycle plus independent CPU candidate-state numerical oracle.

Reuses the previous 24-wave candidate oracle; capture/replay/retirement now belong
to the BetterScale root, not external NPUGraph holders. No weights or serving.
"""

import json
import os
from pathlib import Path

import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op
from betterscale.live import (
    LiveModuleError,
    LiveRuntime,
    TorchStateBackend,
    live_runtime,
)
from betterscale.live.arch.ascend.graph import ACLGraphBackend
from betterscale.live.llm.qwen35 import Capacity, Geometry
from betterscale.live.llm.qwen35.gdn_graph import GDNGraphRoot


def main():
    assert enable_custom_op()
    torch.npu.set_device(0)
    torch.manual_seed(71)
    torch.set_num_threads(4)
    width, capacity, channels = 3, 12, 6144
    geometry = Geometry.from_config(
        json.loads(Path(__file__).with_name("qwen35-0.8b-text-config.json").read_text())
    )
    weight_cpu = (torch.randn(4, channels) * 0.2).bfloat16()
    with live_runtime(
        LiveRuntime(
            device="npu",
            state_backend=TorchStateBackend("npu", memory_budget_bytes=512 << 20),
            graph_backend=ACLGraphBackend(device="npu"),
        )
    ):
        root = GDNGraphRoot(geometry, Capacity(4, 5, token_pages=32), weight_cpu)
    assert not any(s.is_bound for _, s in root.named_states())
    assert not any(g.prepared for _, g in root.named_graphs())
    root.activate()
    assert all(not g.metadata.requires_forward_replay for _, g in root.named_graphs())
    stream = torch.npu.current_stream()
    old_execution = dict(root.named_graphs())["bank0"]._execution
    resident_ids = torch.tensor([2, 0, 3, 1, -1])
    physical = resident_ids[:, None] * width + torch.arange(width)[None, :]
    physical[-1].fill_(-1)
    pool, conv = root.target["0"].recurrent.tensor, root.target["0"].conv.tensor
    # Warmup/capture must restore the seeded State before READY.
    assert torch.all(pool == 0.125).item()
    assert torch.all(conv == 0.25).item()
    pool_seed = torch.randn(pool.shape) * 0.01
    conv_seed = (torch.randn(conv.shape) * 0.1).bfloat16()
    pool.copy_(pool_seed)
    conv.copy_(conv_seed)
    try:
        with torch.inference_mode():
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
                invocation = root.replay(
                    f"bank{wave % 2}",
                    cpu_x,
                    cpu_g,
                    cpu_beta,
                    ends,
                    mapping,
                    resident_ids[order + [4], None].int(),
                    torch.tensor(ordered_accepted, dtype=torch.int32),
                    wave % 2,
                    stream=stream,
                )
                if wave == 0:
                    try:
                        root.close()
                    except LiveModuleError:
                        pass
                    else:
                        raise AssertionError("root closed over an in-flight invocation")
                torch.npu.synchronize()
                invocation.retire()
                actual_y, actual_o = [t.cpu() for t in root.outputs(wave % 2)]
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
                allocator="betterscale.live StateTensor",
                lifecycle="root-owned graphs and invocations",
                resident_seats=5,
                token_pages=32,
                numerical_lanes=50,
                model="Qwen3.5-0.8B",
                scope="one real-geometry GDN leaf, full State allocation; not model/serving integration",
            )
    finally:
        torch.npu.synchronize()
        root.close()
    assert not any(g.prepared for _, g in root.named_graphs())
    assert not any(s.is_bound for _, s in root.named_states())
    # Reactivation publishes a different generation, not a reusable old capture.
    root.activate()
    try:
        try:
            old_execution.replay(stream=stream)
        except LiveModuleError:
            pass
        else:
            raise AssertionError("retired graph executed against a new generation")
        assert torch.all(root.target["0"].recurrent.tensor == 0.125).item()
    finally:
        root.close()
    receipt["lifecycle_passed"] = True
    receipt["forward_shadow"] = False
    Path(os.environ["CAPSULE"], "receipt.json").write_text(
        json.dumps(receipt, indent=2)
    )
    print(
        "owned lifecycle: close-busy rejection, reactivation, stale graph rejection passed",
        flush=True,
    )


if __name__ == "__main__":
    main()
