"""Single-card open-service EOF and malformed-generation gate, no neural work."""

import json
import os
from pathlib import Path
import torch
import torch_npu
from persistent_service import PersistentEngine

torch.npu.set_device(0)
torch_npu.npu.config.allow_internal_format = True
up = torch_npu.npu_format_cast(
    torch.zeros((128, 2048, 1536), dtype=torch.bfloat16, device="npu"), 29
)
down = torch_npu.npu_format_cast(
    torch.zeros((128, 768, 2048), dtype=torch.bfloat16, device="npu"), 29
)
results = []
for invalid in (False, True):
    source = [torch.zeros(65536, dtype=torch.int32, device="npu") for _ in range(2)]
    output = [torch.zeros(262224, dtype=torch.int32, device="npu") for _ in range(2)]
    source[0][0] = -2 if invalid else -1
    source[1][0] = -1
    engine = PersistentEngine(
        [x.data_ptr() for x in source],
        [x.data_ptr() for x in output],
        up,
        down,
        0,
        tasks=32,
        open_service=True,
    )
    engine.replay()
    if invalid:
        try:
            engine.finish()
        except AssertionError:
            assert engine.control[0, 0].item() in (-31, -32)
        else:
            raise AssertionError("invalid EOF accepted")
    else:
        result = engine.finish()
        assert result["completed_counts"] == [0, 0] and result["waves"] == 0
    engine.close()
    results.append(dict(invalid_eof=invalid, passed=True))
Path(os.environ["LOCAL_EXPERT_RESULT"]).write_text(json.dumps(results, indent=2))
