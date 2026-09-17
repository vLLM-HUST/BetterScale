"""Real selected-expert INT8 Cube + row-local Vector composition gate."""

import argparse
import ctypes as C
import json
from pathlib import Path

import torch
import torch_npu
from quant_gmm import QuantGmm
from weights import Checkpoint

p = argparse.ArgumentParser()
p.add_argument("--build", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
torch.npu.set_device(0)
torch.npu.config.allow_internal_format = True
torch.manual_seed(919)
lib = C.CDLL(str(a.build / "launch.so"))
lib.load_server.argtypes = [
    C.c_char_p,
    C.c_char_p,
    C.POINTER(C.c_void_p),
    C.POINTER(C.c_void_p),
]
lib.launch_blocks.argtypes = [C.c_void_p] * 5 + [C.c_uint32]
binary, fn = C.c_void_p(), C.c_void_p()
assert (
    lib.load_server(
        str(a.build / "quant_vector.o").encode(),
        b"quant_vector",
        C.byref(binary),
        C.byref(fn),
    )
    == 0
)


def vector(cfg, x, y):
    assert (
        lib.launch_blocks(
            fn,
            torch.npu.current_stream().npu_stream,
            cfg.data_ptr(),
            x.data_ptr(),
            y.data_ptr(),
            16,
        )
        == 0
    )


host = [Checkpoint().expert(0, e)[1] for e in (0, 1, 257, 511)]
wu = torch.stack(
    [torch.cat([h["gate_proj"][0], h["up_proj"][0]], 0).T for h in host]
).contiguous()
wd = torch.stack([h["down_proj"][0].T for h in host]).contiguous()
su = torch.stack([torch.cat([h["gate_proj"][1], h["up_proj"][1]]) for h in host]).npu()
sd = torch.stack([h["down_proj"][1] for h in host]).npu()
wu_nz, wd_nz = [torch_npu.npu_format_cast(w.npu(), 29) for w in (wu, wd)]
capacity = 160
groups = torch.zeros(4, dtype=torch.int64, device="npu")
x = torch.zeros(capacity, 2560, dtype=torch.bfloat16, device="npu")
xq = torch.empty_like(x, dtype=torch.int8)
sx = torch.empty(capacity, 8, dtype=torch.float32, device="npu")
u = torch.empty(capacity, 1280, dtype=torch.int32, device="npu")
zq = torch.empty(capacity, 640, dtype=torch.int8, device="npu")
sz = torch.empty_like(sx)
d = torch.empty(capacity, 2560, dtype=torch.int32, device="npu")
y = torch.empty_like(x)
qcfg = torch.tensor(
    [0, 2560, capacity, 0, 0, 0, 0, sx.data_ptr()], dtype=torch.int64, device="npu"
)
acfg = torch.tensor(
    [
        1,
        1280,
        capacity,
        sx.data_ptr(),
        su.data_ptr(),
        groups.data_ptr(),
        4,
        sz.data_ptr(),
    ],
    dtype=torch.int64,
    device="npu",
)
dcfg = torch.tensor(
    [2, 2560, capacity, sz.data_ptr(), sd.data_ptr(), groups.data_ptr(), 4, 0],
    dtype=torch.int64,
    device="npu",
)
up, down = QuantGmm(a.build, wu_nz, groups, capacity), QuantGmm(
    a.build, wd_nz, groups, capacity
)


def forward():
    native_q, native_scale = torch_npu.npu_dynamic_quant(x)
    xq.copy_(native_q)
    sx[:, 0].copy_(native_scale)
    up(xq, u)
    vector(acfg, u, zq)
    down(zq, d)
    vector(dcfg, d, y)


graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    forward()
records = []
for counts in ((0, 1, 32, 0), (16, 0, 128, 1), (0, 0, 0, 0), (1, 1, 1, 1)):
    groups.copy_(torch.tensor(counts, dtype=torch.int64).cumsum(0))
    x.copy_(torch.randn(capacity, 2560, dtype=torch.bfloat16))
    x[0].zero_()
    y.fill_(77)
    graph.replay()
    torch.npu.synchronize()
    nativeq, natives = torch_npu.npu_dynamic_quant(x)
    qdiff = int((xq.cpu().short() - nativeq.cpu().short()).abs().max())
    assert qdiff == 0, qdiff
    torch.testing.assert_close(sx[:, 0], natives, rtol=0, atol=0)
    got = y.cpu().float()
    begin = 0
    errors = []
    z_mismatch = 0
    for expert, count in enumerate(counts):
        end = begin + count
        if count:
            raw = (xq[begin:end].cpu().double() @ wu[expert].double()).float()
            deq = raw * sx[begin:end, 0].cpu()[:, None] * su[expert].cpu()[None, :]
            gate, val = deq.chunk(2, -1)
            z = torch.nn.functional.silu(gate) * val
            scale = z.abs().amax(-1) / 127
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            # Owned upstream fused-quant cast is FP32->FP16 RINT->INT8 RINT.
            zref = (
                (z / scale[:, None])
                .half()
                .float()
                .round()
                .clamp(-127, 127)
                .to(torch.int8)
            )
            z_mismatch += int((zref != zq[begin:end].cpu()).sum())
            ref = (
                (
                    (zref.double() @ wd[expert].double()).float()
                    * scale[:, None]
                    * sd[expert].cpu()[None, :]
                )
                .bfloat16()
                .float()
            )
            err = float((got[begin:end] - ref).norm() / ref.norm().clamp_min(1e-9))
            assert err < 0.005, (counts, expert, err)
            errors.append(err)
        begin = end
    assert (got[begin:] == 77).all(), "inactive output changed"
    records.append(
        dict(
            counts=counts,
            input_quant_max_integer_diff=qdiff,
            z_quant_mismatch_elements=z_mismatch,
            relative_l2=errors,
            tail_untouched=True,
        )
    )
graph.reset()
torch.npu.synchronize()
up.close()
down.close()
lib.unload_server.argtypes = [C.c_void_p]
assert lib.unload_server(binary) == 0
a.output.write_text(json.dumps(dict(status="PASS", cases=records), indent=2) + "\n")
print(json.dumps(records), flush=True)
