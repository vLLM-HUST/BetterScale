"""Separate expert batch size, padded rows and layer-group count from transport.

Uses the same synthetic broad32 weights/routes as DFC and the remote control.
This measures a prepacked LOCAL GMM/SwiGLU/GMM chain, never remote latency.
"""

import json
import os
from pathlib import Path
import statistics

import torch
import torch_npu

H, M, E, K = 2048, 768, 128, 8
torch.npu.set_device(0)
torch.set_num_threads(2)
torch_npu.npu.config.allow_internal_format = True
torch.manual_seed(742)
base_up = (torch.randn(H, 2 * M) * 0.01).to(torch.bfloat16).npu()
base_down = (torch.randn(M, H) * 0.01).to(torch.bfloat16).npu()
factors = torch.tensor([0.5 if e % 2 == 0 else 1.0 for e in range(64)], device="npu")
up_nd = (base_up.unsqueeze(0) * factors[:, None, None]).to(torch.bfloat16)
down_nd = base_down.unsqueeze(0).repeat(64, 1, 1)
xs, route_ids = [], []
for source in (0, 1):
    torch.manual_seed(180 + source + 32)
    xs.append((torch.randn(32, H) * 0.1).to(torch.bfloat16).npu())
    route_ids.append((torch.arange(32 * K).reshape(32, K) + source * K) % E)
results = []
for layout in os.environ.get("GMM_BATCH_LAYOUTS", "ND,NZ").split(","):
    for sources in (1, 2):
        # Stable expert-major order, source0 rows followed by source1 per expert.
        rows, counts, expected = [], [], []
        for expert in range(64):
            found = []
            for source in range(sources):
                indices = (route_ids[source] == expert).nonzero()[:, 0].npu()
                found.append(xs[source].index_select(0, indices))
            block = torch.cat(found)
            counts.append(block.shape[0])
            rows.append(block)
            expected.append(
                torch_npu.npu_swiglu(block @ up_nd[expert]) @ down_nd[expert]
            )
        live = torch.cat(rows)
        reference = torch.cat(expected)
        for shape in os.environ.get(
            "GMM_BATCH_SHAPES",
            "64_live,64_padded,128_padded",
        ).split(","):
            group_count = 128 if shape.startswith("128_") else 64
            capacity = live.shape[0] if shape == "64_live" else 512
            packed = torch.zeros(capacity, H, device="npu", dtype=torch.bfloat16)
            packed[: live.shape[0]].copy_(live)
            sizes = counts + [0] * (group_count - 64)
            # Capacity-only is an explicit characterization of this binary,
            # outside its documented group-list-sum == input-M contract. Never
            # enable it as a production optimization merely because it passes.
            if not shape.endswith("capacity_only"):
                sizes[-1] += capacity - live.shape[0]
            groups = torch.tensor(sizes, device="npu", dtype=torch.int64).cumsum(0)
            w1 = up_nd if group_count == 64 else up_nd.repeat(2, 1, 1)
            w2 = down_nd if group_count == 64 else down_nd.repeat(2, 1, 1)
            if layout == "NZ":
                w1 = torch_npu.npu_format_cast(w1, 29)
                w2 = torch_npu.npu_format_cast(w2, 29)

            def body():
                up = torch_npu.npu_grouped_matmul(
                    [packed],
                    [w1],
                    split_item=2,
                    group_list=groups,
                    group_type=0,
                    group_list_type=0,
                )[0]
                act = torch_npu.npu_swiglu(up)
                return torch_npu.npu_grouped_matmul(
                    [act],
                    [w2],
                    split_item=2,
                    group_list=groups,
                    group_type=0,
                    group_list_type=0,
                )[0]

            body()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = body()
            for _ in range(3):
                graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(
                output[: live.shape[0]], reference, rtol=0.02, atol=2e-5
            )
            trials = []
            for _ in range(3):
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                for _ in range(32):
                    graph.replay()
                end.record()
                end.synchronize()
                trials.append(start.elapsed_time(end) * 1000 / 32)
            record = dict(
                layout=layout,
                sources=sources,
                rows_per_source=32,
                shape=shape,
                live_rows=live.shape[0],
                capacity=capacity,
                rows_per_expert=counts[0],
                weight_format=int(torch_npu.get_npu_format(w1)),
                median_us=statistics.median(trials),
                trials_us=trials,
                exact=torch.equal(output[: live.shape[0]], reference),
                documented_group_sum_contract=not shape.endswith("capacity_only"),
            )
            results.append(record)
            print(record, flush=True)
            Path(os.environ["LOCAL_EXPERT_RESULT"]).write_text(
                json.dumps(results, indent=2)
            )
            graph.reset()
