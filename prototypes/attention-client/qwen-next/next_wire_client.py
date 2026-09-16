"""Isolate remote transport and layer metadata from native hybrid execution."""

import json
import os
from pathlib import Path
import torch
import torch_npu
from next_remote import Session

torch.npu.set_device(0)
torch_npu.npu.config.allow_internal_format = True
import vllm_ascend.vllm_ascend_C

if os.environ.get("NEXT_WIRE_CONCURRENCY") == "1":
    from concurrency_probe import run

    run()
    raise SystemExit(0)

session = Session()
print("session ready", flush=True)
source = int(os.environ["EXPERT_SOURCE_ID"])
stress = os.environ.get("NEXT_WIRE_STRESS") == "1"
work = (
    (((i % 4), (1, 3, 32, 17)[i % 4]) for i in range(290 if source == 0 else 1058))
    if stress
    else ((0, 1), (1, 3), (3, 32))
)
for step, (layer, rows) in enumerate(work):
    x = torch.full((rows, 2048), 0.01, dtype=torch.bfloat16, device="npu")
    logits = (
        torch.arange(512, device="npu")
        .float()
        .roll(source * 128 + (step * 37 if stress else 0))
        .expand(rows, 512)
        .contiguous()
    )
    y = session.forward(layer, x, logits, lambda z: torch.zeros_like(z))
    torch.npu.synchronize()
    assert torch.isfinite(y).all()
    if not stress or step % 100 == 0:
        print(f"step {step} layer {layer} rows {rows} passed", flush=True)
count = session.close()
Path(os.environ["EXPERT_ROLE_DIRECTORY"], f"client{source}.json").write_text(
    json.dumps(dict(status="pass", completed=count))
)
