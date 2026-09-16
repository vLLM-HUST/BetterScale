"""Single-device engine-coexistence and actual BF16 neural oracle."""

import json
import os
from pathlib import Path
import torch
import torch_npu
from persistent_service import PersistentEngine

torch.npu.set_device(0)
torch_npu.npu.config.allow_internal_format = True
from next_weights import weights

catalog = weights(0, layers=[0, 1])
up, down = catalog[0]
weight_table = torch.tensor(
    [[u.data_ptr(), d.data_ptr()] for u, d in catalog], dtype=torch.int64, device="npu"
)
reference = [
    (torch_npu.npu_format_cast(u, 2), torch_npu.npu_format_cast(d, 2))
    for u, d in catalog
]
results = []
for pattern in ("hot", "broad", "zero", "skew", "one"):
    sources = [torch.zeros(65536, dtype=torch.int32, device="npu") for _ in range(2)]
    outputs = [
        torch.full((327760,), -71, dtype=torch.int32, device="npu") for _ in range(2)
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
        ids = torch.arange(n * 10).reshape(n, 10) % 128
        if pattern == "hot":
            ids %= 4
        elif pattern == "zero":
            ids += 128
        elif pattern == "skew":
            ids.zero_()
        ids = ids.to(torch.int32).npu()
        layer = 0 if pattern == "skew" else source
        src = sources[source]
        src[8:16] = torch.tensor([1, layer, n, 0, 0, 0, 0, 0], device="npu")
        src[64 : 64 + n * 10].copy_(ids.flatten())
        src[1024 : 1024 + n * 1024].copy_(x.view(torch.int32).flatten())
        src[0] = 1
        ref = torch.empty((n * 10, 2048), dtype=torch.bfloat16, device="npu")
        for e in ids.unique().cpu().tolist():
            rows = (ids.flatten() == e).nonzero().flatten()
            if e < 128:
                tokens = rows // 10
                ref[rows] = (
                    torch_npu.npu_swiglu(x[tokens] @ reference[layer][0][e])
                    @ reference[layer][1][e]
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
        weight_table=weight_table,
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
        mask = ids_all[c].flatten() < 128
        torch.testing.assert_close(
            actual[mask], expected[c][mask], rtol=0.02, atol=2e-5
        )
        assert torch.all(outputs[c][-16:] == -71)
        assert torch.all(actual[~mask].view(torch.int32) == -71)
    results.append(dict(pattern=pattern, **receipt))
    engine.close()
Path(os.environ["LOCAL_EXPERT_RESULT"]).write_text(json.dumps(results, indent=2))
print("next leaf passed", flush=True)
