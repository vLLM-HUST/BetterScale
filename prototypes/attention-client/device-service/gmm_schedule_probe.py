"""Prepacked GEMM scheduling study: rows/expert and warm-vs-alternating weights.

Same device, same real-count kernel, no routing, IPC, polling or client timing.
Whole-chain timings include capacity512 native SwiGLU. Not serving throughput.
"""

import json
import os
from pathlib import Path
import statistics

import torch
import torch_npu
from actual_gmm import ActualGmm

torch.npu.set_device(0)
torch_npu.npu.config.allow_internal_format = True
torch.set_num_threads(2)
torch.manual_seed(872)
up_nd = (torch.randn(128, 2048, 1536) * 0.01).to(torch.bfloat16).npu()
down_nd = (torch.randn(128, 768, 2048) * 0.01).to(torch.bfloat16).npu()
up_w = torch_npu.npu_format_cast(up_nd, 29)
down_w = torch_npu.npu_format_cast(down_nd, 29)
x = (torch.randn(512, 2048) * 0.1).to(torch.bfloat16).npu()
results = []
for experts, rows in [(16, 2), (64, 2), (64, 4), (64, 8)]:
    live = experts * rows
    banks = []
    for layer in (0, 1):
        counts = torch.zeros(128, dtype=torch.int64, device="npu")
        counts[layer * 64 : layer * 64 + experts] = rows
        groups = counts.cumsum(0)
        up = ActualGmm(up_w, groups)
        down = ActualGmm(down_w, groups)
        u = torch.empty((512, 1536), dtype=torch.bfloat16, device="npu")
        y = torch.empty((512, 2048), dtype=torch.bfloat16, device="npu")
        up(x, u)
        act = torch_npu.npu_swiglu(u)
        down(act, y)
        torch.npu.synchronize()
        for expert in range(experts):
            lo, hi = expert * rows, (expert + 1) * rows
            ref = (
                torch_npu.npu_swiglu(x[lo:hi] @ up_nd[layer * 64 + expert])
                @ down_nd[layer * 64 + expert]
            )
            torch.testing.assert_close(y[lo:hi], ref, rtol=0.02, atol=2e-5)
        graphs = []
        for kind in ("up", "down", "chain"):
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                if kind in ("up", "chain"):
                    up(x, u)
                if kind == "chain":
                    act_chain = torch_npu.npu_swiglu(u)
                    down(act_chain, y)
                elif kind == "down":
                    down(act, y)
            graphs.append(graph)
        banks.append((graphs, up, down, u, y, groups, act))
    for kind in (0, 1, 2):
        for alternating in (False, True):
            for iteration in range(8):
                banks[iteration % 2 if alternating else 0][0][kind].replay()
            torch.npu.synchronize()
            trials = []
            for trial in range(5):
                a, b = torch.npu.Event(enable_timing=True), torch.npu.Event(
                    enable_timing=True
                )
                a.record()
                for iteration in range(32):
                    banks[iteration % 2 if alternating else 0][0][kind].replay()
                b.record()
                b.synchronize()
                trials.append(a.elapsed_time(b) * 1000 / 32)
            results.append(
                dict(
                    experts=experts,
                    rows_per_expert=rows,
                    live=live,
                    kind=("up", "down", "chain")[kind],
                    alternating_layers=alternating,
                    median_us=statistics.median(trials),
                    trials_us=trials,
                )
            )
    for graphs, up, down, *_ in banks:
        for graph in graphs:
            graph.reset()
        up.close()
        down.close()
Path(os.environ["LOCAL_EXPERT_RESULT"]).write_text(json.dumps(results, indent=2))
