"""Real routed layer0 plus dummy BF16 shared GEMMs: fork/join, not model quality."""

import argparse
import json
import os
from pathlib import Path
import statistics
import torch
import torch_npu
from client import Session

p = argparse.ArgumentParser()
p.add_argument("--directory", type=Path, required=True)
p.add_argument("--build", type=Path, required=True)
p.add_argument("--source", type=int, default=0)
a = p.parse_args()
torch.set_num_threads(2)
torch.npu.set_device(0)
torch.manual_seed(731)
os.environ["QWEN38_SHARED_OVERLAP"] = "1"
session = Session(a.directory, a.build, a.source)
up = torch.randn(2560, 1280, device="npu", dtype=torch.bfloat16) * 0.01
down = torch.randn(640, 2560, device="npu", dtype=torch.bfloat16) * 0.01


def shared(x):
    gate, value = (x @ up).chunk(2, -1)
    return (torch.nn.functional.silu(gate) * value) @ down


experts = (
    [0, 1, 170, 171, 172, 341, 342, 343, 510, 511]
    if session.layout.owners == 3
    else [0, 1, 127, 128, 129, 255, 256, 383, 384, 511]
)
records = []
for n in sorted({1, 32, session.layout.rows}):
    x = torch.randn(n, 2560, device="npu", dtype=torch.bfloat16)
    ids = (
        torch.tensor(experts, dtype=torch.int32, device="npu")
        .expand(n, -1)
        .contiguous()
    )
    probs = torch.full((n, 10), 0.1, device="npu", dtype=torch.bfloat16)
    for priority in (0, 1):
        graphs, outputs = {}, {}
        for overlap in (False, True):
            session.shared_overlap = overlap
            for _ in range(2):
                session.forward_routed(0, x, ids, probs, shared, priority=priority)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                with torch.npu.graph(graph):
                    outputs[overlap] = session.forward_routed(
                        0, x, ids, probs, shared, priority=priority
                    )
            stream.synchronize()
            graphs[overlap] = graph
        for generation in range(3):
            x.copy_(torch.randn_like(x))
            # Alternate route order as well as input content in stable buffers.
            ids.copy_(ids.roll(1, dims=1))
            for overlap in (False, True):
                graphs[overlap].replay()
                torch.npu.synchronize()
            assert torch.equal(outputs[False], outputs[True]), (n, priority, generation)
        times = {False: [], True: []}
        for repeat in range(10):
            for overlap in ((False, True) if repeat % 2 == 0 else (True, False)):
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
                    enable_timing=True
                )
                start.record()
                graphs[overlap].replay()
                end.record()
                end.synchronize()
                times[overlap].append(start.elapsed_time(end))
        records.append(
            dict(
                rows=n,
                priority=priority,
                exact=True,
                serial_ms=statistics.median(times[False]),
                overlap_ms=statistics.median(times[True]),
                samples=times,
            )
        )
        if n == session.layout.rows and priority == 1:
            profile_dir = a.directory / f"profile-client{a.source}"
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    str(profile_dir), analyse_flag=False
                ),
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1
                ),
            ):
                for overlap in (False, True):
                    with torch.profiler.record_function(
                        "shared_overlap" if overlap else "shared_serial"
                    ):
                        graphs[overlap].replay()
                        torch.npu.synchronize()
        for graph in graphs.values():
            graph.reset()
        print(records[-1], flush=True)
count = session.close()
(a.directory / f"client{a.source}.json").write_text(
    json.dumps(dict(status="PASS", calls=count, cases=records), indent=2)
)
