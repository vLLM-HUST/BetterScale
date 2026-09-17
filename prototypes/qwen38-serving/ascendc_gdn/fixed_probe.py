"""Fixed shape acceptance of owned build before capacity-stride modification."""

import os, json
from pathlib import Path
import torch, torch_npu
from vllm_ascend.utils import enable_custom_op
from runtime import Kernels

assert enable_custom_op()
torch.npu.set_device(0)
torch.manual_seed(73)
root = Path(os.environ["CAPSULE"])
engine = Kernels(os.environ["ASCENDC_GDN_LIB"], 512, 1, 8)
bf = dict(dtype=torch.bfloat16, device="npu")
q = torch.randn(1, 8, 512, 128, **bf) * 0.05
k = torch.randn_like(q) * 0.05
w = torch.randn(1, 24, 512, 128, **bf) * 0.01
u = torch.randn_like(w) * 0.1
g = (
    -torch.rand(1, 24, 512, device="npu")
    .reshape(1, 24, 8, 64)
    .cumsum(-1)
    .reshape(1, 24, 512)
    * 0.01
)
initial = torch.randn(1, 24, 128, 128, device="npu") * 0.01
cu = torch.tensor([0, 512], device="npu", dtype=torch.int64)
indices = torch.tensor([(0, i) for i in range(8)], device="npu", dtype=torch.int64)
args = (q, k, w, u, g, initial, cu, indices)
with torch.inference_mode():
    ho, vn, fs = torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g,
        gk=None,
        initial_state=initial,
        output_final_state=True,
        chunk_size=64,
        save_new_value=True,
        cu_seqlens=(0, 512),
        chunk_indices=tuple(indices.cpu().flatten().tolist()),
        use_exp2=False,
        transpose_state_layout=False,
    )
    ref = torch.ops._C_ascend.chunk_fwd_o(
        q,
        k,
        vn,
        ho,
        g=g,
        scale=128**-0.5,
        g_gamma=None,
        cu_seqlens=(0, 512),
        chunk_indices=tuple(indices.cpu().flatten().tolist()),
        chunk_size=64,
        transpose_state_layout=False,
    )
    torch.npu.synchronize()
    engine(*args)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out, state = engine(*args)
    graph.replay()
    torch.npu.synchronize()
    result = {}
    for name, a, b in [
        ("output", out, ref),
        ("state", state, fs),
        ("h", engine.h, ho),
        ("vnew", engine.v, vn),
    ]:
        result[name] = {
            "max_abs": float((a.float() - b.float()).abs().max()),
            "close": bool(torch.allclose(a, b, atol=0.01, rtol=0.01)),
        }
    result["status"] = "PASS" if all(v["close"] for v in result.values()) else "FAIL"
    (root / "receipt.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    assert result["status"] == "PASS"
