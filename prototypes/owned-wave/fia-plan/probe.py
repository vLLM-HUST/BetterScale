"""Native non-FD FIA: immutable launch plan, owned GM lengths, two ACL banks."""

import ctypes
import json
import os
from pathlib import Path
import torch
import torch_npu

lib = ctypes.CDLL(os.environ["LD_PRELOAD"])
lib.plan_begin.argtypes = [ctypes.c_uint64]
lib.plan_finish.restype = ctypes.c_int
lib.plan_workspace.argtypes = [ctypes.c_int]
lib.plan_workspace.restype = ctypes.c_uint64
lib.plan_bind.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
lib.plan_launch.argtypes = [ctypes.c_int, ctypes.c_void_p]
torch.npu.set_device(0)
torch.manual_seed(173)
width = int(os.environ.get("FIA_QUERY_WIDTH", "1"))
rows = 4 if width == 1 else 1
qheads, kvheads = 16, int(os.environ.get("FIA_KV_HEADS", "2"))
qoffsets = [width * (i + 1) for i in range(rows)]
initial = [max(width, n) for n in [17, 129, 257, 513][:rows]]
length_sets = [
    [max(width, n) for n in values[:rows]]
    for values in (
        [17, 129, 257, 513],
        [127, 128, 129, 1024],
        [511, 512, 513, 900],
        [1023, 1024, 1025, 2048],
        [2047, 2048, 2049, 4096],
        [4095, 4096, 4097, 8192],
    )
]
k = torch.randn(rows * 64, 128, kvheads * 128, device="npu", dtype=torch.bfloat16)
v = torch.randn_like(k)
mask = torch.ones(2048, 2048, device="npu", dtype=torch.bool).triu_(1)
kwargs = dict(
    num_heads=16,
    num_key_value_heads=kvheads,
    scale=128**-0.5,
    block_size=128,
    input_layout="TND",
    sparse_mode=3,
    next_tokens=0,
)
banks = []
for bank in range(2):
    q = torch.randn(rows * width, 16, 128, device="npu", dtype=torch.bfloat16)
    table = torch.arange(rows * 64, device="npu", dtype=torch.int32).reshape(rows, 64)
    ql = torch.tensor(qoffsets, device="npu", dtype=torch.int64)
    kl = torch.tensor(initial, device="npu", dtype=torch.int64)
    torch.npu.synchronize()
    assert lib.plan_begin(q.data_ptr()) == 0
    original = torch_npu.npu_fused_infer_attention_score(
        q,
        k,
        v,
        block_table=table,
        atten_mask=mask,
        actual_seq_lengths=qoffsets,
        actual_seq_lengths_kv=initial,
        **kwargs,
    )[0]
    torch.npu.synchronize()
    plan = lib.plan_finish()
    assert plan >= 0, plan
    size = lib.plan_workspace(plan)
    assert 0 < size < 256 * 1024**2, size
    workspace = torch.zeros(size, device="npu", dtype=torch.uint8)
    out = torch.empty_like(original)
    bindings = (ctypes.c_uint64 * 9)(
        *[t.data_ptr() for t in (q, mask, ql, kl, table, out, workspace, k, v)]
    )
    assert lib.plan_bind(plan, bindings) == 0
    torch.npu.synchronize()
    assert lib.plan_launch(plan, torch.npu.current_stream().npu_stream) == 0
    torch.npu.synchronize()
    torch.testing.assert_close(out, original, rtol=0.01, atol=0.002)
    print("eager native-plan PASS", bank, size, flush=True)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        assert lib.plan_launch(plan, torch.npu.current_stream().npu_stream) == 0
    banks.append((q, table, ql, kl, out, workspace, graph))
checks = []
for step, lengths in enumerate(length_sets * 2):
    b = step % 2
    q, table, ql, kl, out, workspace, graph = banks[b]
    kl.copy_(torch.tensor(lengths, device="npu", dtype=torch.int64))
    table.copy_(
        torch.roll(
            torch.arange(rows * 64, device="npu", dtype=torch.int32), step
        ).reshape(rows, 64)
    )
    graph.replay()
    torch.npu.synchronize()
    ref = torch_npu.npu_fused_infer_attention_score(
        q,
        k,
        v,
        block_table=table,
        atten_mask=mask,
        actual_seq_lengths=qoffsets,
        actual_seq_lengths_kv=lengths,
        **kwargs,
    )[0]
    torch.testing.assert_close(out, ref, rtol=0.01, atol=0.002)
    assert kl.cpu().tolist() == list(lengths)
    checks.append(
        dict(bank=b, lengths=lengths, max_error=(out - ref).abs().max().item())
    )
    print("ACL native-plan PASS", checks[-1], flush=True)
Path("complete.json").write_text(
    json.dumps(dict(status="PASS", checks=checks), indent=2)
)
