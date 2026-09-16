"""Check dynamic group metadata and both native GEMMs before raw IPC integration."""

import json
import os
from pathlib import Path
import torch
import torch_npu

torch.npu.set_device(0)
torch_npu.npu.config.allow_internal_format = True
torch.manual_seed(742)
x = (torch.randn(512, 2048) * 0.1).to(torch.bfloat16).npu()
w = (torch.randn(128, 2048, 1536) * 0.01).to(torch.bfloat16).npu()
v = (torch.randn(128, 768, 2048) * 0.01).to(torch.bfloat16).npu()
records = []
for mode in ("ND", "NZ", "NZ_T"):
    weight = w if mode == "ND" else torch_npu.npu_format_cast(w, 29)
    weight2 = v if mode == "ND" else torch_npu.npu_format_cast(v, 29)
    if mode == "NZ_T":
        weight = torch_npu.npu_format_cast(
            w.transpose(1, 2).contiguous(), 29
        ).transpose(1, 2)
        weight2 = torch_npu.npu_format_cast(
            v.transpose(1, 2).contiguous(), 29
        ).transpose(1, 2)
    counts = torch.zeros(128, dtype=torch.int64, device="npu")
    counts[-1] = 512
    groups = counts.cumsum(0)

    def body():
        up = torch_npu.npu_grouped_matmul(
            [x],
            [weight],
            split_item=2,
            group_list=groups,
            group_type=0,
            group_list_type=0,
        )[0]
        act = torch_npu.npu_swiglu(up)
        out = torch_npu.npu_grouped_matmul(
            [act],
            [weight2],
            split_item=2,
            group_list=groups,
            group_type=0,
            group_list_type=0,
        )[0]
        return up, act, out

    def check(out, ids):
        for e, begin, end in ids:
            ref = torch_npu.npu_swiglu(x[begin:end] @ w[e]) @ v[e]
            torch.testing.assert_close(out[begin:end], ref, rtol=0.02, atol=2e-5)

    up, act, out = body()
    check(out, [(127, 0, 512)])
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        up, act, out = body()
    for pattern in ("broad", "hot8", "padding"):
        counts.zero_()
        if pattern == "broad":
            counts[:64] = 2
            counts[-1] = 384
            ids = [(e, e * 2, e * 2 + 2) for e in range(64)]
        elif pattern == "hot8":
            counts[:4] = 32
            counts[-1] = 384
            ids = [(e, e * 32, (e + 1) * 32) for e in range(4)]
        else:
            counts[-1] = 512
            ids = [(127, 0, 512)]
        groups.copy_(counts.cumsum(0))
        graph.replay()
        torch.npu.synchronize()
        check(out, ids)
        a = torch.npu.Event(enable_timing=True)
        b = torch.npu.Event(enable_timing=True)
        a.record()
        for _ in range(32):
            graph.replay()
        b.record()
        b.synchronize()
        records.append(
            dict(
                mode=mode,
                pattern=pattern,
                formats=[
                    int(torch_npu.get_npu_format(t)) for t in (weight, up, act, out)
                ],
                us=a.elapsed_time(b) * 1000 / 32,
            )
        )
        print(records[-1], flush=True)
        Path(os.environ["LOCAL_EXPERT_RESULT"]).write_text(
            json.dumps(records, indent=2)
        )
    graph.reset()
