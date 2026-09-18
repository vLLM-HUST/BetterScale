"""Native installed W8A8 fused up/activation gate, independent of server scheduling."""

import argparse
import json
from pathlib import Path
import statistics
import torch
import torch_npu

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(2)
torch.npu.set_device(0)
torch.npu.config.allow_internal_format = True
torch.manual_seed(145)
groups_count, k, n = 171, 2560, 1280
# Exact model geometry, dummy integer weights. Include high expert IDs and empty groups.
w_cpu = torch.randint(-4, 5, (groups_count, k, n), dtype=torch.int8)
w = torch_npu.npu_format_cast(w_cpu.npu(), 29)
ws = torch.full((groups_count, n), 0.02, device="npu")
records = []
for capacity in (4096, 51200):
    x = torch.randint(-8, 9, (capacity, k), dtype=torch.int8, device="npu")
    scale = torch.full((capacity,), 0.03, device="npu")
    ends = torch.zeros(groups_count, dtype=torch.int64, device="npu")
    counts = torch.zeros(groups_count, dtype=torch.int64)
    counts[[0, 1, 169, 170]] = 1024
    ends.copy_(counts.cumsum(0))

    def call():
        return torch_npu.npu_grouped_matmul_swiglu_quant(x, w, ends, ws, scale)

    eager = call()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = call()
    checks = []
    for active in (4096, 3072):
        counts.zero_()
        counts[[0, 169, 170]] = 1024
        if active == 4096:
            counts[1] = 1024
        ends.copy_(counts.cumsum(0))
        x.copy_(torch.randint(-8, 9, x.shape, dtype=torch.int8, device="npu"))
        graph.replay()
        torch.npu.synchronize()
        reference = call()
        torch.npu.synchronize()
        assert torch.equal(out[0][:active], reference[0][:active])
        torch.testing.assert_close(
            out[1][:active], reference[1][:active], rtol=0, atol=0
        )
        errors = []
        begin = 0
        for expert, count in enumerate(counts.tolist()):
            if not count:
                continue
            inds = [begin, begin + count - 1]
            u = (x[inds].cpu().double() @ w_cpu[expert].double()).float() * 0.03 * 0.02
            gate, value = u.chunk(2, -1)
            expected = torch.nn.functional.silu(gate) * value
            got = out[0][inds].cpu().float() * out[1][inds].cpu().float().reshape(-1, 1)
            error = float((got - expected).norm() / expected.norm())
            assert error < 0.025, (capacity, expert, error)
            errors.append(error)
            begin += count
        times = []
        for _ in range(10):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
                enable_timing=True
            )
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        checks.append(
            dict(
                active_rows=active,
                graph_exact=True,
                max_dequantized_relative_l2=max(errors),
                median_ms=statistics.median(times),
                samples=times,
            )
        )
    graph.reset()
    records.append(dict(capacity=capacity, cases=checks))
a.output.write_text(
    json.dumps(
        dict(
            status="PASS",
            scope="dummy weights; native standalone fused up/activation only, not persistent integration",
            cases=records,
        ),
        indent=2,
    )
    + "\n"
)
print("PASS", records, flush=True)
