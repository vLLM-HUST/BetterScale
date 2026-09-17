"""Exact integer gate for actual-count INT8 groups, including empty groups."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu
from quant_gmm import QuantGmm

p = argparse.ArgumentParser()
p.add_argument("--build", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
torch.npu.set_device(0)
torch.npu.config.allow_internal_format = True
torch.manual_seed(919)
records = []
for k, n in ((2560, 1280), (640, 2560)):
    w = torch.randint(-127, 128, (4, k, n), dtype=torch.int8)
    nz = torch_npu.npu_format_cast(w.npu(), 29)
    groups = torch.zeros(4, dtype=torch.int64, device="npu")
    op = QuantGmm(a.build, nz, groups, 160)
    x = torch.randint(-127, 128, (160, k), dtype=torch.int8).npu()
    out = torch.full((160, n), -123456789, dtype=torch.int32, device="npu")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        op(x, out)
    for counts in ((0, 1, 32, 0), (16, 0, 128, 1), (0, 0, 0, 0), (1, 1, 1, 1)):
        x.copy_(torch.randint(-127, 128, (160, k), dtype=torch.int8))
        groups.copy_(torch.tensor(counts, dtype=torch.int64).cumsum(0))
        out.fill_(-123456789)
        graph.replay()
        torch.npu.synchronize()
        got, host_x = out.cpu(), x.cpu().double()
        begin = 0
        for expert, count in enumerate(counts):
            end = begin + count
            expected = (host_x[begin:end] @ w[expert].double()).to(torch.int32)
            torch.testing.assert_close(got[begin:end], expected, rtol=0, atol=0)
            begin = end
        assert (got[begin:] == -123456789).all(), "inactive output overwritten"
        records.append(dict(k=k, n=n, counts=counts, exact=True, tail_untouched=True))
    graph.reset()
    op.close()
a.output.write_text(json.dumps(dict(status="PASS", cases=records), indent=2) + "\n")
print("PASS", len(records), "exact integer cases", flush=True)
