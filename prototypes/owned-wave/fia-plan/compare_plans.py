"""Diagnostic only: native short/actual-length launch plans, no timing baseline."""

import ctypes
import json
import os
from pathlib import Path
import torch
import torch_npu

lib = ctypes.CDLL(os.environ["LD_PRELOAD"])
lib.select_query.argtypes = [ctypes.c_uint64]
torch.npu.set_device(0)
q = torch.randn(4, 16, 128, device="npu", dtype=torch.bfloat16)
k = torch.randn(1024, 128, 256, device="npu", dtype=torch.bfloat16)
v = torch.randn_like(k)
mask = torch.ones(2048, 2048, device="npu", dtype=torch.bool).triu_(1)
table = torch.arange(1024, device="npu", dtype=torch.int32).reshape(4, 256)
root = Path.cwd()
for name, lengths in [
    ("bootstrap", [1] * 4),
    ("representative", [4558, 4627, 4397, 4386]),
    ("long", [16384] * 4),
]:
    (root / name).mkdir()
    os.chdir(root / name)
    torch.npu.synchronize()
    lib.select_query(q.data_ptr())
    out = torch_npu.npu_fused_infer_attention_score(
        q,
        k,
        v,
        block_table=table,
        atten_mask=mask,
        actual_seq_lengths=[1, 2, 3, 4],
        actual_seq_lengths_kv=lengths,
        num_heads=16,
        num_key_value_heads=2,
        scale=128**-0.5,
        block_size=128,
        input_layout="TND",
        sparse_mode=3,
    )[0]
    torch.npu.synchronize()
    assert Path("fia-launch.bin").exists()
    Path("lengths.json").write_text(json.dumps(lengths))
    print(name, Path("fia-launch.meta").read_text(), flush=True)
os.chdir(root)
Path("complete.json").write_text(
    json.dumps(
        dict(
            status="PASS",
            scope="native launch metadata only, not a performance baseline",
        )
    )
)
