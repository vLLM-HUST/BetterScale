"""Fixed-order collect control/candidate: real server and already-ready replay.

Ready timing excludes server work, uses warm outputs, and is NOT a bandwidth
roofline. Each numeric observation is cloned before another graph can reuse it.
"""

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
torch.manual_seed(851 + a.source)
os.environ.update(
    QWEN38_SHARED_OVERLAP="1",
    QWEN38_PARALLEL_PACK="1",
    QWEN38_FUSED_COLLECT="1",
    QWEN38_PIPELINED_COLLECT="0",
    QWEN38_ONLINE_COLLECT="0",
)
session = Session(a.directory, a.build, a.source)
assert session.kernels.pipelined_client_collect
functions = {
    name: session.kernels.load(symbol)
    for name, symbol in (
        ("serial", "neural_collect_fused"),
        ("pipeline", "neural_collect_pipelined"),
    )
}
up = torch.randn(2560, 1280, dtype=torch.bfloat16, device="npu") * 0.01
down = torch.randn(640, 2560, dtype=torch.bfloat16, device="npu") * 0.01


def shared(x):
    gate, value = (x @ up).chunk(2, -1)
    return (torch.nn.functional.silu(gate) * value) @ down


def capture(body):
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    graph = torch.npu.NPUGraph()
    with torch.npu.stream(stream), torch.npu.graph(graph):
        output = body()
    stream.synchronize()
    return graph, output


def measure(graphs, replays):
    samples = {mode: [] for mode in graphs}
    for iteration in range(12):
        for mode in (list(graphs) if iteration % 2 == 0 else list(reversed(graphs))):
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
                enable_timing=True
            )
            start.record()
            for _ in range(replays):
                graphs[mode].replay()
            end.record()
            end.synchronize()
            samples[mode].append(start.elapsed_time(end) / replays)
    return {
        "medians_ms": {m: statistics.median(t) for m, t in samples.items()},
        "samples_ms": samples,
    }


records = []
for n in sorted({1, 7, 32, 127, min(512, session.layout.rows), session.layout.rows}):
    x = torch.randn(n, 2560, dtype=torch.bfloat16, device="npu")
    ids = (
        torch.arange(n * 10, device="npu", dtype=torch.int32).view(n, 10) % 512
    ).contiguous()
    probs = torch.full((n, 10), 0.1, dtype=torch.bfloat16, device="npu")
    graphs, outputs = {}, {}
    for mode, fn in functions.items():
        session.collect_fused = fn
        for _ in range(2):
            session.forward_routed(0, x, ids, probs, shared, priority=1)
        torch.npu.synchronize()
        graphs[mode], outputs[mode] = capture(
            lambda: session.forward_routed(0, x, ids, probs, shared, priority=1)
        )
    numeric = []
    for generation in range(4):
        x.normal_()
        ids.copy_(torch.randint(0, 512, ids.shape, device="npu", dtype=torch.int32))
        if generation == 1:
            ids.zero_()  # Duplicate expert, all other owners empty.
        w = torch.rand(n, 10, device="npu")
        probs.copy_(w / w.sum(-1, keepdim=True))
        snapshots = {}
        for mode in functions:
            graphs[mode].replay()
            snapshots[mode] = outputs[mode].clone()
            torch.npu.synchronize()
        assert torch.equal(snapshots["serial"], snapshots["pipeline"]), (n, generation)
        if os.environ.get("QWEN38_SAVE_COLLECT_OUTPUTS") == "1":
            torch.save(
                snapshots["serial"].cpu(),
                a.directory / f"output-{a.source}-{n}-{generation}.pt",
            )
        numeric.append(dict(generation=generation, exact=True))
    leaf = measure(graphs, 1)
    # No publication/retirement: all outputs and source generation remain stable.
    bank = session.banks[(n, True, True)]
    ready = {}
    for mode, fn in functions.items():
        ready[mode], _ = capture(
            lambda: session.kernels.call(
                fn, bank.config, bank.input, bank.ids_storage, 16
            )
        )
    ready_times = measure(ready, 16)
    records.append(dict(rows=n, numeric=numeric, leaf=leaf, ready=ready_times))
    print(json.dumps(records[-1]), flush=True)
    for graph in [*ready.values(), *graphs.values()]:
        graph.reset()
count = session.close()
(a.directory / f"client{a.source}.json").write_text(
    json.dumps(dict(status="PASS", calls=count, cases=records), indent=2) + "\n"
)
