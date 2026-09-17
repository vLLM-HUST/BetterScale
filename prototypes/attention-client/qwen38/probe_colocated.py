"""Eight-rank real-layer native EP gate; not an end-to-end serving benchmark."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import torch
import torch_npu
from catalog import load
from colocated_ep import ColocatedEP
from weights import Checkpoint

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
rank = int(os.environ["RANK"])
torch.set_num_threads(2)
torch.npu.set_device(0)
torch.npu.config.allow_internal_format = True
torch.distributed.init_process_group(
    "hccl", rank=rank, world_size=8, timeout=timedelta(seconds=300)
)
probe = torch.ones(1, device="npu")
torch.distributed.all_reduce(probe)
torch.npu.synchronize()
assert probe.item() == 8
catalog = load(rank, layers=1, owners=8)
ep = ColocatedEP(torch.distributed.group.WORLD, catalog)
# Uneven source row counts represented by masks, including a wholly idle rank.
rows = 4
active_rows = rank % 5
active = torch.arange(rows, device="npu") < active_rows
torch.manual_seed(100 + rank)
x = torch.randn(rows, 2560, dtype=torch.bfloat16, device="npu")
expert_ids = [0, 1, 63, 64, 127, 128, 255, 256, 384, 511]
ids = (
    torch.tensor(expert_ids, dtype=torch.int32, device="npu")
    .expand(rows, -1)
    .contiguous()
)
probs = torch.full((rows, 10), 0.1, dtype=torch.bfloat16, device="npu")


def run():
    return ep.routed(0, x, ids, probs, active=active)


eager = run()
torch.npu.synchronize()
print("EAGER_OK", rank, flush=True)
graph = torch.npu.NPUGraph()
stream = torch.npu.Stream()
stream.wait_stream(torch.npu.current_stream())
with torch.npu.stream(stream):
    with torch.npu.graph(graph):
        out = run()
stream.synchronize()
x.mul_(0.75)
graph.replay()
torch.npu.synchronize()
reference = run()
torch.npu.synchronize()
assert torch.equal(out[:active_rows], reference[:active_rows])
q, scales = torch_npu.npu_dynamic_quant(x)
q, scales = q.cpu().double(), scales.cpu()
checkpoint = Checkpoint()
routed = []
for expert in expert_ids:
    w = checkpoint.expert(0, expert)[1]
    wu = torch.cat([w["gate_proj"][0], w["up_proj"][0]], 0).T
    su = torch.cat([w["gate_proj"][1], w["up_proj"][1]])
    u = (q @ wu.double()).float() * scales[:, None] * su[None, :]
    gate, up = u.chunk(2, -1)
    z = torch.nn.functional.silu(gate) * up
    sz = z.abs().amax(-1).clamp_min(1e-30) / 127
    zq = (z / sz[:, None]).half().float().round().clamp(-127, 127)
    y = (
        (zq.double() @ w["down_proj"][0].T.double()).float()
        * sz[:, None]
        * w["down_proj"][1][None, :]
    ).bfloat16()
    routed.append(y)
expected = (
    (torch.stack(routed, 1).float() * probs.cpu().float()[:, :, None])
    .sum(1)
    .bfloat16()
    .float()[:active_rows]
)
got = out.cpu().float()[:active_rows]
error = float((got - expected).norm() / expected.norm().clamp_min(1e-9))
assert error < 0.006, (rank, error)
a.output.mkdir(exist_ok=True, parents=True)
(a.output / f"rank{rank}.json").write_text(
    json.dumps(
        dict(
            status="PASS",
            rank=rank,
            active_rows=active_rows,
            relative_l2=error,
            full_eager_replay_equal=True,
        )
    )
)
graph.reset()
torch.distributed.barrier()
torch.distributed.destroy_process_group()
print("PASS", rank, error, flush=True)
