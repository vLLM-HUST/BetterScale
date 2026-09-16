"""Actual-count GEMM: changing groups, empty waves, skew, canaries and FULL replay."""

import json
import os
from pathlib import Path
import statistics
import torch
import torch_npu
from actual_gmm import ActualGmm

torch.npu.set_device(0)
torch_npu.npu.config.allow_internal_format = True
torch.manual_seed(742)
results = []
for k, n in ((2048, 1536), (768, 2048)):
    nd = (torch.randn(128, k, n) * 0.01).to(torch.bfloat16).npu()
    weight = torch_npu.npu_format_cast(nd, 29)
    x = (torch.randn(512, k) * 0.1).to(torch.bfloat16).npu()
    x_saved = x.clone()
    groups = torch.zeros(128, device="npu", dtype=torch.int64)
    storage = torch.full((514, n), -17.0, device="npu", dtype=torch.bfloat16)
    out = storage[1:513]
    fn = ActualGmm(weight, groups)
    fn(x, out)
    torch.npu.synchronize()
    assert torch.all(storage == -17.0)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        fn(x, out)
    for pattern in ("broad1", "broad2", "hot", "layer1", "skew", "empty", "broad1"):
        sizes = [0] * 128
        if pattern.startswith("broad"):
            sizes[:64] = [2 if pattern == "broad1" else 4] * 64
        elif pattern == "hot":
            sizes[:4] = [32] * 4
        elif pattern == "layer1":
            sizes[64:] = [3] * 64
        elif pattern == "skew":
            sizes[127] = 512
        live = sum(sizes)
        groups.copy_(torch.tensor(sizes, device="npu", dtype=torch.int64).cumsum(0))
        storage.fill_(-17.0)
        graph.replay()
        torch.npu.synchronize()
        offset = 0
        exact = True
        for e, m in enumerate(sizes):
            if m:
                ref = x[offset : offset + m] @ nd[e]
                torch.testing.assert_close(
                    out[offset : offset + m], ref, rtol=0.02, atol=2e-4
                )
                exact = exact and torch.equal(out[offset : offset + m], ref)
            offset += m
        assert torch.all(storage[0] == -17.0) and torch.all(storage[513] == -17.0)
        assert torch.all(out[live:] == -17.0)
        assert torch.equal(x, x_saved)
        trials = []
        for _ in range(3):
            a = torch.npu.Event(enable_timing=True)
            b = torch.npu.Event(enable_timing=True)
            a.record()
            for _ in range(32):
                graph.replay()
            b.record()
            b.synchronize()
            trials.append(a.elapsed_time(b) * 1000 / 32)
        record = dict(
            k=k,
            n=n,
            pattern=pattern,
            live=live,
            exact=exact,
            median_us=statistics.median(trials),
        )
        results.append(record)
        print(record, flush=True)
        Path(os.environ["LOCAL_EXPERT_RESULT"]).write_text(
            json.dumps(results, indent=2)
        )
    graph.reset()
    fn.close()
