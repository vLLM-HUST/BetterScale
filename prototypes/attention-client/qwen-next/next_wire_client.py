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

session = Session()
print("session ready", flush=True)
source = int(os.environ["EXPERT_SOURCE_ID"])
for layer, rows in ((0, 1), (1, 3), (3, 32)):
    x = torch.full((rows, 2048), 0.01, dtype=torch.bfloat16, device="npu")
    logits = (
        torch.arange(512, device="npu")
        .float()
        .roll(source * 128)
        .expand(rows, 512)
        .contiguous()
    )
    y = session.forward(layer, x, logits, lambda z: torch.zeros_like(z))
    torch.npu.synchronize()
    assert torch.isfinite(y).all()
    print(f"layer {layer} rows {rows} passed", flush=True)
count = session.close()
Path(os.environ["EXPERT_ROLE_DIRECTORY"], f"client{source}.json").write_text(
    json.dumps(dict(status="pass", completed=count))
)
