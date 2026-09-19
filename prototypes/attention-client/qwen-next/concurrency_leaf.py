"""Ready-at-admission control: both sources published before server execution.

One local expert owner, not remote transport/whole-model performance. Same routes
and 32 rows per source; only layer identity changes. This separates batching
capability from the arrival skew of two real independent clients.
"""

import json
import os
from pathlib import Path
import statistics

import torch
import torch_npu
from next_weights import weights
from persistent_service import PersistentEngine

torch.npu.set_device(0)
torch_npu.npu.config.allow_internal_format = True
catalog = weights(0, layers=[0, 1])
table = torch.tensor(
    [[u.data_ptr(), d.data_ptr()] for u, d in catalog], dtype=torch.int64, device="npu"
)
sources = [torch.zeros(65536, dtype=torch.int32, device="npu") for _ in range(2)]
outputs = [torch.zeros(524288, dtype=torch.int32, device="npu") for _ in range(2)]
ids = (
    (
        torch.arange(32, device="npu")[:, None] * 13
        + torch.arange(10, device="npu")[None, :] * 53
    )
    % 512
).int()
for c, src in enumerate(sources):
    torch.manual_seed(824 + c)
    x = (torch.randn((32, 2048), device="npu") * 0.1).bfloat16()
    src[64:384].copy_(ids.flatten())
    src[1024 : 1024 + 32768].copy_(x.view(torch.int32).flatten())
results = []
# Alternate to reduce simple order/cache drift; two warmups per case excluded.
for repeat in range(10):
    for case in ("same", "different") if repeat % 2 == 0 else ("different", "same"):
        for c, src in enumerate(sources):
            layer = c if case == "different" else 0
            src[8:16] = torch.tensor([1, layer, 32, 0, 0, 0, 0, 0], device="npu")
            src[0] = 1
            outputs[c].zero_()
        engine = PersistentEngine(
            [s.data_ptr() for s in sources],
            [o.data_ptr() for o in outputs],
            *catalog[0],
            0,
            tasks=1,
            weight_table=table,
        )
        torch.npu.synchronize()
        engine.replay()
        receipt = engine.finish()
        assert receipt["waves"] == (1 if case == "same" else 2)
        assert all(o[0].item() == 1 for o in outputs)
        ev = receipt["events"]
        span = (max(e[4] for e in ev) - min(e[3] for e in ev)) / 50
        results.append(
            dict(case=case, repeat=repeat, waves=receipt["waves"], span_us=span)
        )
        engine.close()
summary = {
    case: statistics.median(
        x["span_us"] for x in results if x["case"] == case and x["repeat"] >= 2
    )
    for case in ("same", "different")
}
Path(os.environ["LOCAL_EXPERT_RESULT"]).write_text(
    json.dumps(
        dict(
            scope="Single local server, two prepublished sources; excludes remote transport and client latency",
            medians_us=summary,
            samples=results,
        ),
        indent=2,
    )
)
print(summary, flush=True)
