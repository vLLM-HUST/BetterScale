"""Single-device engine-coexistence and actual BF16 neural oracle."""

import json
import os
from pathlib import Path
import torch
import torch_npu
from persistent_service import PersistentEngine

torch.npu.set_device(0)
torch_npu.npu.config.allow_internal_format = True
torch.manual_seed(742)
# Independent random expert weights, not identical repeated expert matrices.
up_nd = (torch.randn(128, 2048, 1536) * 0.01).to(torch.bfloat16).npu()
down_nd = (torch.randn(128, 768, 2048) * 0.01).to(torch.bfloat16).npu()
up = torch_npu.npu_format_cast(up_nd, 29)
down = torch_npu.npu_format_cast(down_nd, 29)
results = []
for pattern in ("hot", "broad", "zero", "skew", "one"):
    sources = [torch.zeros(65536, dtype=torch.int32, device="npu") for _ in range(2)]
    outputs = [
        torch.full((262224,), -71, dtype=torch.int32, device="npu") for _ in range(2)
    ]
    expected = []
    ids_all = []
    for source in range(2):
        n = (
            32
            if pattern == "skew"
            else (1 if pattern == "one" else (16 if source else 32))
        )
        x = (torch.randn(n, 2048) * 0.1).to(torch.bfloat16).npu()
        ids = torch.arange(n * 8).reshape(n, 8) % 64
        if pattern == "hot":
            ids %= 4
        elif pattern == "zero":
            ids += 64
        elif pattern == "skew":
            ids.zero_()
        ids = ids.to(torch.int32).npu()
        layer = 0 if pattern == "skew" else source
        src = sources[source]
        src[8:16] = torch.tensor([1, layer, n, 0, 0, 0, 0, 0], device="npu")
        src[64 : 64 + n * 8].copy_(ids.flatten())
        src[1024 : 1024 + n * 1024].copy_(x.view(torch.int32).flatten())
        src[0] = 1
        ref = torch.empty((n * 8, 2048), dtype=torch.bfloat16, device="npu")
        for e in ids.unique().cpu().tolist():
            rows = (ids.flatten() == e).nonzero().flatten()
            if e < 64:
                tokens = rows // 8
                ref[rows] = (
                    torch_npu.npu_swiglu(x[tokens] @ up_nd[layer * 64 + e])
                    @ down_nd[layer * 64 + e]
                )
        expected.append(ref)
        ids_all.append(ids)
    saved = [x.clone() for x in sources]
    engine = PersistentEngine(
        [x.data_ptr() for x in sources],
        [x.data_ptr() for x in outputs],
        up,
        down,
        0,
        tasks=1,
    )
    engine.replay()
    receipt = engine.finish()
    for c in range(2):
        assert torch.equal(sources[c], saved[c])
        assert outputs[c][0].item() == 1
        actual = (
            outputs[c][64 : 64 + expected[c].numel() // 2]
            .view(torch.bfloat16)
            .reshape_as(expected[c])
        )
        mask = ids_all[c].flatten() < 64
        torch.testing.assert_close(
            actual[mask], expected[c][mask], rtol=0.02, atol=2e-5
        )
        assert torch.all(outputs[c][-16:] == -71)
        assert torch.all(actual[~mask].view(torch.int32) == -71)
    results.append(dict(pattern=pattern, **receipt))
    engine.close()
# Invalid descriptor: clean device termination, not a host/process timeout.
sources[0][0] = 1
sources[0][9] = 7  # invalid layer, outside [0,2)
engine = PersistentEngine(
    [x.data_ptr() for x in sources],
    [x.data_ptr() for x in outputs],
    up,
    down,
    0,
    tasks=1,
)
engine.replay()
try:
    engine.finish()
    raise RuntimeError("invalid descriptor was accepted")
except AssertionError:
    assert engine.control[0, 0].item() in (-31, -32)
finally:
    engine.close()
results.append(dict(pattern="invalid_descriptor", rejected=True))
# No source publication: bounded device watchdog must end both engine teams.
for src in sources:
    src.zero_()
engine = PersistentEngine(
    [x.data_ptr() for x in sources],
    [x.data_ptr() for x in outputs],
    up,
    down,
    0,
    tasks=1,
)
engine.config[9] = 10000
torch.npu.synchronize()
engine.replay()
try:
    engine.finish()
    raise RuntimeError("missing source did not time out")
except AssertionError:
    assert engine.control[0, 0].item() in (-11, -21, -33)
finally:
    engine.close()
results.append(dict(pattern="missing_source", bounded_timeout=True))
Path(os.environ["LOCAL_EXPERT_RESULT"]).write_text(json.dumps(results, indent=2))
