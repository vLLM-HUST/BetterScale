"""Real layer0 E4 IPC math/graph gate, not a language-model serving claim."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu
from client import Session
from weights import Checkpoint

p = argparse.ArgumentParser()
p.add_argument("--directory", type=Path, required=True)
p.add_argument("--build", type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(2)
torch.npu.set_device(0)
torch_npu.npu.config.allow_internal_format = True
torch.manual_seed(777)
checkpoint = Checkpoint()
experts = [0, 1, 127, 128, 129, 255, 256, 383, 384, 511]
weights = [checkpoint.expert(0, e)[1] for e in experts]
session = Session(a.directory, a.build)
records = []
for n in (1, 4, 32):
    x = torch.randn(n, 2560, dtype=torch.bfloat16, device="npu")
    ids = (
        torch.tensor(experts, dtype=torch.int32, device="npu")
        .expand(n, -1)
        .contiguous()
    )
    probs = torch.full((n, 10), 0.1, dtype=torch.bfloat16, device="npu")

    def run():
        return session.forward_routed(
            0, x, ids, probs, lambda h: h * 0.125, priority=int(n > 1)
        )

    eager = run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        with torch.npu.graph(graph):
            output = run()
    stream.synchronize()
    for generation in range(2):
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.npu.synchronize()
        got = output.cpu().float()
        q, scale = torch_npu.npu_dynamic_quant(x)
        q = q.cpu().double()
        scale = scale.cpu()
        routed = []
        for w in weights:
            wu = torch.cat([w["gate_proj"][0], w["up_proj"][0]], 0).T
            su = torch.cat([w["gate_proj"][1], w["up_proj"][1]])
            u = (q @ wu.double()).float() * scale[:, None] * su[None, :]
            gate, value = u.chunk(2, -1)
            z = torch.nn.functional.silu(gate) * value
            sz = z.abs().amax(-1) / 127
            sz = torch.where(sz > 0, sz, torch.ones_like(sz))
            zq = (
                (z / sz[:, None]).half().float().round().clamp(-127, 127).to(torch.int8)
            )
            y = (
                (zq.double() @ w["down_proj"][0].T.double()).float()
                * sz[:, None]
                * w["down_proj"][1][None, :]
            ).bfloat16()
            routed.append(y)
        # Match native unpermute's BF16 output boundary, then shared addition.
        ref = (
            (torch.stack(routed, 1).float() * probs.cpu().float()[:, :, None])
            .sum(1)
            .bfloat16()
        )
        ref = (ref + x.cpu() * 0.125).float()
        error = float((got - ref).norm() / ref.norm().clamp_min(1e-9))
        assert error < 0.006, (n, generation, error)
        records.append(dict(rows=n, generation=generation, relative_l2=error))
    graph.reset()
count = session.close()
(a.directory / "client.json").write_text(
    json.dumps(dict(status="PASS", calls=count, cases=records), indent=2)
)
print("PASS E4 wire", records, flush=True)
