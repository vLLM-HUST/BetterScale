"""Check that decode can consume the same K-V pool without global transposes."""

import json, os
from pathlib import Path
import torch, torch_npu
from vllm_ascend.utils import enable_custom_op

assert enable_custom_op()
torch.npu.set_device(0)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

init_device_properties_triton()
from decode_kv import fused_recurrent_gated_delta_rule_fwd as decode

root = Path(os.environ["CAPSULE"])
torch.manual_seed(107)


def measure(gr):
    a, b = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    a.record()
    for _ in range(30):
        gr.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / 30


with torch.inference_mode():
    seed = torch.randn(8, 24, 128, 128, device="npu") * 0.01
    bank = seed.clone()
    native_bank = seed.transpose(-1, -2).contiguous()
    q = torch.nn.functional.normalize(
        torch.randn(1, 4, 8, 128, device="npu"), dim=-1
    ).bfloat16()
    k = torch.nn.functional.normalize(torch.randn_like(q.float()), dim=-1).bfloat16()
    v = torch.randn(1, 4, 24, 128, device="npu").bfloat16()
    g = -torch.rand(1, 4, 24, device="npu") * 0.1
    beta = torch.rand(1, 4, 24, device="npu").bfloat16()
    cu = torch.arange(5, device="npu", dtype=torch.int32)
    slots = torch.tensor([5, 2, 7, 1], device="npu", dtype=torch.int32)
    actual = torch.tensor([0, 1, 1, 1, 1], device="npu", dtype=torch.int32)

    def candidate():
        return decode(q, k, v, g, beta, 128**-0.5, bank, True, cu, slots)[0]

    def native():
        return torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
            query=q.squeeze(0),
            key=k.squeeze(0),
            value=v.squeeze(0),
            g=g.squeeze(0),
            beta=beta.squeeze(0),
            state=native_bank,
            scale=128**-0.5,
            actual_seq_lengths=actual,
            ssm_state_indices=slots,
        ).unsqueeze(0)

    graphs = {}
    outputs = {}
    for name, fn in [("candidate", candidate), ("native", native)]:
        fn()
        torch.npu.synchronize()
        gr = torch.npu.NPUGraph()
        with torch.npu.graph(gr):
            outputs[name] = fn()
        graphs[name] = gr
    bank.copy_(seed)
    native_bank.copy_(seed.transpose(-1, -2))
    results = []
    for ids in ([5, 2, 7, 1], [3, 6, 2, 4], [1, 2, 5, 7], [7, 5, 3, 6]):
        slots.copy_(torch.tensor(ids, device="npu", dtype=torch.int32))
        graphs["candidate"].replay()
        graphs["native"].replay()
        torch.npu.synchronize()
        row = {}
        for name, a, b in [
            ("output", outputs["candidate"], outputs["native"]),
            ("state", bank, native_bank.transpose(-1, -2)),
        ]:
            row[name] = {
                "max_abs": float((a.float() - b.float()).abs().max()),
                "close": bool(torch.allclose(a, b, atol=0.01, rtol=0.01)),
                "finite": bool(torch.isfinite(a).all()),
            }
        results.append(row)
    times = {"candidate": [], "native": []}
    for name in ["native", "candidate", "candidate", "native"] * 2:
        times[name].append(measure(graphs[name]))
    result = {
        "status": (
            "PASS"
            if all(v["close"] and v["finite"] for r in results for v in r.values())
            else "FAIL"
        ),
        "checks": results,
        "ms": times,
    }
(root / "receipt.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
assert result["status"] == "PASS"
