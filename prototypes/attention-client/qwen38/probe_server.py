"""One-card server gate: two source mailboxes, mixed target/MTP layer catalogs.

No cross-device transport or attention root is claimed. Only four selected real
experts per catalog are populated; routes outside this owner owe no payload.
"""

import argparse
import json
from pathlib import Path

import torch
import torch_npu
from server_engine import Engine
from weights import Checkpoint

p = argparse.ArgumentParser()
p.add_argument("--build", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
torch.npu.set_device(0)
torch.npu.config.allow_internal_format = True
torch.set_num_threads(2)
torch.manual_seed(62)
checkpoint = Checkpoint()
target = [checkpoint.expert(0, e)[1] for e in range(4)]
mtp = [checkpoint.expert(48, e)[1] for e in range(4)]
tu = torch.zeros(128, 2560, 1280, dtype=torch.int8)
td = torch.zeros(128, 640, 2560, dtype=torch.int8)
su = torch.ones(128, 1280, dtype=torch.float32)
sd = torch.ones(128, 2560, dtype=torch.float32)
bu = torch.zeros_like(tu, dtype=torch.bfloat16)
bd = torch.zeros_like(td, dtype=torch.bfloat16)
for e in range(4):
    tu[e].copy_(torch.cat([target[e]["gate_proj"][0], target[e]["up_proj"][0]], 0).T)
    td[e].copy_(target[e]["down_proj"][0].T)
    su[e].copy_(torch.cat([target[e]["gate_proj"][1], target[e]["up_proj"][1]]))
    sd[e].copy_(target[e]["down_proj"][1])
    bu[e].copy_(mtp[e]["gate_up_proj"].T)
    bd[e].copy_(mtp[e]["down_proj"].T)
quant = (
    torch_npu.npu_format_cast(tu.npu(), 29),
    torch_npu.npu_format_cast(td.npu(), 29),
    su.npu(),
    sd.npu(),
)
bfloat = (
    torch_npu.npu_format_cast(bu.npu(), 29),
    torch_npu.npu_format_cast(bd.npu(), 29),
    None,
    None,
)
catalog = [quant] * 48 + [bfloat]
records = []
for layers, priorities in (((0, 0), (1, 1)), ((0, 48), (0, 1)), ((48, 0), (1, 0))):
    sources = [
        torch.zeros(1024 * 256, dtype=torch.int32, device="npu") for _ in range(2)
    ]
    outputs = [
        torch.full((64 + 320 * 1280,), -991, dtype=torch.int32, device="npu")
        for _ in range(2)
    ]
    engine = Engine(
        a.build,
        [s.data_ptr() for s in sources],
        [o.data_ptr() for o in outputs],
        catalog,
        0,
    )
    references = []
    routes = []
    for source, (layer, priority, n) in enumerate(zip(layers, priorities, (4, 16))):
        x = torch.randn(n, 2560, dtype=torch.bfloat16).npu()
        ids = torch.randint(0, 4, (n, 10), dtype=torch.int32)
        ids[:, 8] = 128
        ids[:, 9] = 256
        routes.append(ids)
        refs = []
        if layer < 48:
            q, scale = torch_npu.npu_dynamic_quant(x)
            for e in range(4):
                u = (
                    (q.cpu().double() @ tu[e].double()).float()
                    * scale.cpu()[:, None]
                    * su[e][None, :]
                )
                gate, value = u.chunk(2, -1)
                z = torch.nn.functional.silu(gate) * value
                sz = z.abs().amax(-1) / 127
                sz = torch.where(sz > 0, sz, torch.ones_like(sz))
                zq = (
                    (z / sz[:, None])
                    .half()
                    .float()
                    .round()
                    .clamp(-127, 127)
                    .to(torch.int8)
                )
                refs.append(
                    (
                        (zq.double() @ td[e].double()).float()
                        * sz[:, None]
                        * sd[e][None, :]
                    ).bfloat16()
                )
            sources[source][1024 : 1024 + n * 640].view(torch.int8).reshape(
                n, 2560
            ).copy_(q)
            scales = (
                sources[source][512 : 512 + n * 8].view(torch.float32).reshape(n, 8)
            )
            scales[:, 0].copy_(scale)
        else:
            for e in range(4):
                refs.append((torch_npu.npu_swiglu(x @ bu[e].npu()) @ bd[e].npu()).cpu())
            sources[source][1024 : 1024 + n * 1280].view(torch.bfloat16).reshape(
                n, 2560
            ).copy_(x)
        references.append(refs)
        sources[source][64 : 64 + n * 10].copy_(ids.flatten())
        sources[source][8:12].copy_(
            torch.tensor([1, layer, n, priority], dtype=torch.int32)
        )
        sources[source][0] = 1
    torch.npu.synchronize()
    engine.replay()
    receipt = engine.finish()
    errors = []
    for source, n in enumerate((4, 16)):
        assert int(outputs[source][0].cpu()) == 1
        got = (
            outputs[source][64 : 64 + n * 10 * 1280]
            .view(torch.bfloat16)
            .reshape(n, 10, 2560)
            .cpu()
        )
        for e in range(4):
            where = (routes[source] == e).nonzero()
            actual = got[where[:, 0], where[:, 1]].float()
            expected = references[source][e][where[:, 0]].float()
            error = float((actual - expected).norm() / expected.norm().clamp_min(1e-9))
            assert error < 0.005, (layers, source, e, error)
            errors.append(error)
        assert (
            outputs[source][64:].reshape(320, 1280)[
                (routes[source].flatten() >= 128).nonzero().flatten()
            ]
            == -991
        ).all()
    assert receipt["completed_counts"] == [1, 1]
    if layers == (0, 0):
        assert receipt["waves"] == 1, receipt
    records.append(
        dict(layers=layers, priorities=priorities, relative_l2=errors, receipt=receipt)
    )
    engine.close()
a.output.write_text(json.dumps(dict(status="PASS", cases=records), indent=2) + "\n")
print("PASS", len(records), "persistent mixed-catalog cases", flush=True)
