"""Installed host FD planner -> banked GM tiling -> original captured kernel."""

import ctypes
import json
import os
from pathlib import Path
import torch
import torch_npu

lib = ctypes.CDLL(os.environ["FIA_PLAN_LIBRARY"])
u64 = ctypes.c_uint64
i64 = ctypes.c_int64
lib.plan_native_queries.argtypes = [
    ctypes.POINTER(u64),
    *[ctypes.c_int] * 6,
    ctypes.c_double,
    ctypes.POINTER(i64),
    ctypes.POINTER(i64),
    ctypes.c_void_p,
]
lib.plan_bind.argtypes = [ctypes.c_int, ctypes.POINTER(u64)]
lib.plan_metadata.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
lib.plan_bind_metadata.argtypes = [ctypes.c_int, u64]
lib.plan_launch.argtypes = [ctypes.c_int, ctypes.c_void_p]
lib.plan_workspace.argtypes = [ctypes.c_int]
lib.plan_workspace.restype = u64


def check(rc):
    if rc < 0:
        raise RuntimeError(f"native plan failure {rc}")
    return rc


torch.npu.set_device(0)
torch.manual_seed(173)
width = int(os.environ.get("FIA_QUERY_WIDTH", "1"))
rows = int(os.environ.get("FIA_ROWS", "4" if width == 1 else "1"))
valid_widths = [
    int(x) for x in os.environ.get("FIA_VALID_WIDTHS", str(width)).split(",")
]
assert all(1 <= n <= width for n in valid_widths)
assert rows == 1 or valid_widths == [width]
kvheads = 2
pages = rows * 256
k = torch.randn(pages, 128, kvheads * 128, device="npu", dtype=torch.bfloat16)
v = torch.randn_like(k)
mask = torch.ones(2048, 2048, device="npu", dtype=torch.bool).triu_(1)
workspace = torch.zeros(128 * 1024**2, device="npu", dtype=torch.uint8)
length_sets = [
    [max(width, n) for n in ls[:rows]]
    for ls in [
        [1, 1, 1, 1],
        [1023, 1, 129, 511],
        [1024, 1, 129, 511],
        [4558, 4627, 4397, 4386],
        [16384] * 4,
        [32768, 1, 257, 8193],
        [8191, 8192, 8193, 16383],
        [513, 1025, 4097, 32767],
        [4628, 4398, 4387, 4559],
        [32768] * 4,
    ]
]
banks = []
for bank in range(2):
    q = torch.randn(rows * width, 16, 128, device="npu", dtype=torch.bfloat16)
    table = torch.arange(pages, device="npu", dtype=torch.int32).reshape(rows, 256)
    out = torch.empty_like(q)
    ql = torch.tensor(
        [width * (i + 1) for i in range(rows)], device="npu", dtype=torch.int64
    )
    kl = torch.ones(rows, device="npu", dtype=torch.int64)
    gm = torch.zeros(4096, device="npu", dtype=torch.uint8)
    banks.append((q, table, out, ql, kl, gm, {}))
torch.npu.synchronize()
checks = []
for step, lengths in enumerate(length_sets * 2):
    valid = valid_widths[(step // 2) % len(valid_widths)]
    offsets = [valid * (i + 1) for i in range(rows)]
    bank = step % 2
    q, table, out, ql, kl, gm, graphs = banks[bank]
    pointers = (u64 * 7)(
        *[t.data_ptr() for t in (q, k, v, mask, table, out, workspace)]
    )
    plan = check(
        lib.plan_native_queries(
            pointers,
            rows,
            width,
            16,
            kvheads,
            pages,
            256,
            128**-0.5,
            (i64 * rows)(*lengths),
            (i64 * rows)(*offsets),
            torch.npu.current_stream().npu_stream,
        )
    )
    native_blocks = lib.plan_blocks(plan)
    if os.environ.get("FIA_PAD_BLOCKS") == "1":
        check(lib.plan_pad_blocks(plan))
    signature = (lib.plan_is_fd(plan), lib.plan_blocks(plan))
    host = torch.zeros(4096, dtype=torch.uint8, pin_memory=True)
    size = check(lib.plan_metadata(plan, host.data_ptr(), host.numel()))
    gm.copy_(host, non_blocking=True)
    ql.copy_(torch.tensor(offsets, dtype=torch.int64, device="npu"))
    kl.copy_(torch.tensor(lengths, dtype=torch.int64, device="npu"))
    table.copy_(
        torch.roll(torch.arange(pages, device="npu", dtype=torch.int32), step).reshape(
            rows, 256
        )
    )
    if signature not in graphs:
        bindings = (u64 * 9)(
            *[t.data_ptr() for t in (q, mask, ql, kl, table, out, workspace, k, v)]
        )
        check(lib.plan_bind(plan, bindings))
        check(lib.plan_bind_metadata(plan, gm.data_ptr()))
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            check(lib.plan_launch(plan, torch.npu.current_stream().npu_stream))
        graphs[signature] = (graph, plan)
    else:
        check(lib.plan_release(plan))
    graphs[signature][0].replay()
    torch.npu.synchronize()
    ref = torch_npu.npu_fused_infer_attention_score(
        q[: rows * valid],
        k,
        v,
        atten_mask=mask,
        block_table=table,
        actual_seq_lengths=offsets,
        actual_seq_lengths_kv=lengths,
        num_heads=16,
        num_key_value_heads=kvheads,
        scale=128**-0.5,
        block_size=128,
        input_layout="TND",
        sparse_mode=3,
        next_tokens=0,
    )[0]
    actual = out[: rows * valid]
    torch.testing.assert_close(actual, ref, rtol=0.01, atol=0.002)
    result = dict(
        bank=bank,
        capacity=width,
        actual_query=valid,
        lengths=lengths,
        fd=signature[0],
        blocks=signature[1],
        native_blocks=native_blocks,
        tiling_bytes=size,
        max_error=(actual - ref).abs().max().item(),
    )
    checks.append(result)
    print(result, flush=True)
Path("complete.json").write_text(
    json.dumps(dict(status="PASS", checks=checks), indent=2)
)
