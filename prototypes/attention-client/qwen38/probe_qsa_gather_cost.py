"""Bounded head-major gather timing; same rows/counts for both overlay versions."""

import argparse
import json
import os
import torch
import torch_npu

p = argparse.ArgumentParser()
p.add_argument("--device", required=True)
a = p.parse_args()
assert os.environ.get("ASCEND_RT_VISIBLE_DEVICES") == a.device
torch.npu.set_device(0)
torch.set_num_threads(2)
from livemodule.arch.ascend.llm.qwen38.qsa_gather import gather_qsa_kv

for heads in (1, 2):
    rows, selected = 128, 2051
    cache = torch.randn(40, 64, heads, 256, device="npu", dtype=torch.bfloat16)
    value = torch.randn_like(cache)
    query = torch.zeros(rows, 1, 12 * heads, 256, device="npu", dtype=torch.bfloat16)
    table = torch.arange(40, device="npu", dtype=torch.int32)[None]
    requests = torch.zeros(rows, device="npu", dtype=torch.int64)
    for count in (128, 1024):
        indices = torch.arange(selected, device="npu", dtype=torch.int32).repeat(
            rows, 1
        )
        packed = torch.cat(
            (indices, torch.full((rows, 1), count, device="npu", dtype=torch.int32)), 1
        )

        def run():
            return gather_qsa_kv(query, cache, value, packed, table, requests)

        run()
        torch.npu.synchronize()
        times = []
        for _ in range(3):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
                enable_timing=True
            )
            start.record()
            got = run()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        expected = cache.reshape(-1, heads, 256)[:selected].clone()
        expected[count:] = 0
        assert torch.equal(got[0], expected[None].expand(rows, -1, -1, -1))
        print(
            json.dumps(
                dict(
                    heads=heads,
                    count=count,
                    rows=rows,
                    selected=selected,
                    milliseconds=times,
                    status="PASS",
                )
            ),
            flush=True,
        )
