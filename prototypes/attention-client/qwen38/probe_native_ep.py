"""Eight-card real-layer MC2 seam: broad/narrow routes, prefix masks, FULL replay."""

import os
import json
from datetime import timedelta
import torch
import torch_npu
from catalog import load
from colocated_ep import ColocatedEP

rank = int(os.environ["RANK"])
torch.set_num_threads(2)
torch.npu.set_device(rank)
torch_npu.npu.config.allow_internal_format = True
torch.distributed.init_process_group("hccl", timeout=timedelta(seconds=120))
with torch.inference_mode():
    catalog = load(rank, owners=8, layers=1)
    ep = ColocatedEP(torch.distributed.group.WORLD, catalog)
    torch.distributed.barrier()
    for n, narrow in ((32, False), (256, False), (256, True), (1020, False)):
        hidden = torch.randn(n, 2560, dtype=torch.bfloat16, device="npu")
        ids = (torch.arange(n * 10, device="npu").reshape(n, 10) + rank * 17) % 512
        if narrow:
            ids = torch.arange(10, device="npu").expand(n, -1)
        ids = ids.to(torch.int32).contiguous()
        probs = torch.full((n, 10), 0.1, dtype=torch.bfloat16, device="npu")
        active = torch.arange(n, device="npu") % 7 == 2
        print(
            json.dumps(dict(rank=rank, rows=n, narrow=narrow, stage="begin")),
            flush=True,
        )
        expected = ep.routed(0, hidden, ids, probs, active=active)
        torch.npu.synchronize()
        assert torch.isfinite(expected).all().cpu()
        assert not expected[~active].count_nonzero().cpu()
        graph = torch.npu.NPUGraph()
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream), torch.npu.graph(graph):
            actual = ep.routed(0, hidden, ids, probs, active=active)
        stream.synchronize()
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(actual, expected)
        graph.reset()
        print(
            json.dumps(dict(rank=rank, rows=n, narrow=narrow, stage="PASS")), flush=True
        )
    torch.distributed.barrier()
torch.distributed.destroy_process_group()
