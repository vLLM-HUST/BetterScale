"""One-card real expert gate: native W8A8 kernels, independent integer oracle.

This is not a remote-server or whole-model qualification. The oracle consumes
actual dynamically quantized inputs to isolate integer GEMM/dequant boundaries.
"""

import argparse
import json
from pathlib import Path

import torch
import torch_npu
from weights import Checkpoint

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
torch.npu.set_device(0)
torch.npu.config.allow_internal_format = True
torch.manual_seed(17)
checkpoint = Checkpoint()
records = []
for layer, expert in ((0, 0), (3, 511), (47, 257), (48, 0)):
    kind, host = checkpoint.expert(layer, expert)
    if kind == "w8a8_dynamic":
        weights = {
            k: (torch_npu.npu_format_cast(w.T.contiguous().npu(), 29), scale.npu())
            for k, (w, scale) in host.items()
        }

        def linear(x, key):
            q, sx = torch_npu.npu_dynamic_quant(x)
            w, sw = weights[key]
            y = torch_npu.npu_quant_matmul(
                q, w, sw, pertoken_scale=sx, output_dtype=torch.bfloat16
            )
            return y, q, sx

        def forward(x):
            gate, _, _ = linear(x, "gate_proj")
            up, _, _ = linear(x, "up_proj")
            z = torch.nn.functional.silu(gate) * up
            return linear(z, "down_proj")[0]

    else:
        weights = {k: v.npu() for k, v in host.items()}

        def forward(x):
            gate, up = torch.nn.functional.linear(x, weights["gate_up_proj"]).chunk(
                2, -1
            )
            return torch.nn.functional.linear(
                torch.nn.functional.silu(gate) * up, weights["down_proj"]
            )

    for rows in (1, 32, 128):
        x = torch.randn(rows, 2560, dtype=torch.bfloat16).npu()
        stage_errors = {}
        if kind == "w8a8_dynamic":
            for key, width in (
                ("gate_proj", 2560),
                ("up_proj", 2560),
                ("down_proj", 640),
            ):
                operand = (
                    x
                    if width == 2560
                    else torch.randn(rows, width, dtype=torch.bfloat16).npu()
                )
                y, q, sx = linear(operand, key)
                # Exact integer accumulation fits FP64 for these admitted shapes.
                w, sw = host[key]
                ref = (
                    (q.cpu().double() @ w.T.double())
                    * sx.cpu().double()[:, None]
                    * sw.double()[None, :]
                ).to(torch.bfloat16)
                err = float(
                    (y.cpu().float() - ref.float()).norm()
                    / ref.float().norm().clamp_min(1e-9)
                )
                assert err < 0.005, (layer, expert, rows, key, err)
                stage_errors[key] = err
        eager = forward(x)
        graph = torch.npu.NPUGraph()
        torch.npu.synchronize()
        with torch.npu.graph(graph):
            out = forward(x)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(out, eager, rtol=0, atol=0)
        # Changed values at stable input addresses, not just first replay.
        x.copy_(torch.randn_like(x))
        reference = forward(x)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(out, reference, rtol=0, atol=0)
        assert torch.isfinite(out).all()
        records.append(
            dict(
                layer=layer,
                expert=expert,
                rows=rows,
                kind=kind,
                stage_relative_l2=stage_errors,
                changed_replay_exact=True,
            )
        )
        del graph, out
    del weights
result = dict(
    status="PASS",
    scope="selected real expert math and graph; no remote transport",
    cases=records,
)
a.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
