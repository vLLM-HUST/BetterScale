"""Real routed layer0 plus dummy BF16 shared GEMMs: fork/join, not model quality."""

import argparse
import json
import os
from pathlib import Path
import statistics
import torch
import torch_npu
from client import Session, Bank

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
# Optional four-arm experiment keeps shared overlap and packing independent.
modes = {"serial": (False, False), "overlap": (True, False)}
pack_checks = []
if os.environ.get("QWEN38_TEST_PACK") == "1":
    assert session.kernels.parallel_client_pack
    modes.update(pack=(False, True), pack_overlap=(True, True))
    # Private, unexported backing verifies both target and MTP wire layouts;
    # publishing here cannot wake any real server or alter its generation.
    for n in (1, 7, 32, 127, session.layout.rows):
        for quantized in (False, True):
            bank = Bank(session, n, quantized)
            bank.input.copy_(
                torch.randint(-100, 100, bank.input.shape, device="npu").to(
                    bank.input.dtype
                )
            )
            bank.scales.copy_(torch.rand_like(bank.scales))
            bank.ids_storage.copy_(
                torch.arange(bank.ids_storage.numel(), device="npu", dtype=torch.int32)
            )
            backing = torch.zeros(
                session.layout.source_bytes // 4, device="npu", dtype=torch.int32
            )
            counter = torch.zeros(8, device="npu", dtype=torch.int32)
            bank.config[0] = backing.data_ptr()
            bank.config[7] = counter.data_ptr()
            session.kernels.call(
                session.submit, bank.config, bank.input, bank.ids_storage
            )
            expected = backing.clone()
            backing.zero_()
            session.kernels.call(
                session.pack, bank.config, bank.input, bank.ids_storage, 16
            )
            session.kernels.call(
                session.publish, bank.config, bank.input, bank.ids_storage
            )
            torch.npu.synchronize()
            assert torch.equal(backing, expected), (n, quantized, "wire bytes")
            pack_checks.append(dict(rows=n, quantized=quantized, bytes_equal=True))
if os.environ.get("QWEN38_TEST_FUSED") == "1":
    assert session.kernels.fused_client_collect
    modes.update(fused=(False, True), fused_overlap=(True, True))
    if os.environ.get("QWEN38_SWEEP_COLLECT") == "1":
        modes.update(fused32=(False, True), fused48=(False, True))
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
        for mode, (overlap, pack) in modes.items():
            session.shared_overlap = overlap
            session.parallel_pack = pack
            session.fused_collect = mode.startswith("fused")
            session.collect_blocks = (
                32 if mode == "fused32" else 48 if mode == "fused48" else 16
            )
            for _ in range(2):
                session.forward_routed(0, x, ids, probs, shared, priority=priority)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                with torch.npu.graph(graph):
                    outputs[mode] = session.forward_routed(
                        0, x, ids, probs, shared, priority=priority
                    )
            stream.synchronize()
            graphs[mode] = graph
        for generation in range(3):
            x.copy_(torch.randn_like(x))
            # Alternate route order as well as input content in stable buffers.
            ids.copy_(ids.roll(1, dims=1))
            if os.environ.get("QWEN38_TEST_FUSED") == "1":
                weights = torch.rand(n, 10, device="npu")
                probs.copy_(weights / weights.sum(-1, keepdim=True))
            for mode, (overlap, pack) in modes.items():
                graphs[mode].replay()
                torch.npu.synchronize()
            for mode, out in outputs.items():
                ref = outputs["serial"].float()
                error = float((out.float() - ref).norm() / ref.norm().clamp_min(1e-9))
                assert torch.equal(outputs["serial"], out), (
                    n,
                    priority,
                    generation,
                    mode,
                    error,
                    float((out.float() - ref).abs().max()),
                )
        times = {mode: [] for mode in modes}
        for repeat in range(10):
            for mode in (list(modes) if repeat % 2 == 0 else list(reversed(modes))):
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
                    enable_timing=True
                )
                start.record()
                graphs[mode].replay()
                end.record()
                end.synchronize()
                times[mode].append(start.elapsed_time(end))
        records.append(
            dict(
                rows=n,
                priority=priority,
                exact=True,
                serial_ms=statistics.median(times["serial"]),
                overlap_ms=statistics.median(times["overlap"]),
                medians_ms={mode: statistics.median(ts) for mode, ts in times.items()},
                samples=times,
            )
        )
        if n == session.layout.rows and priority == 1:
            # All servers have finished this generation. Replay only collect:
            # no submit, no mutation or retirement, so this isolates warm pull
            # cost from remote scheduling/compute wait (not a cold-bandwidth test).
            bank = session.banks[(n, True)]
            ready_graph = torch.npu.NPUGraph()
            ready_stream = torch.npu.Stream()
            ready_stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(ready_stream):
                with torch.npu.graph(ready_graph):
                    session.kernels.call(
                        session.collect, bank.config, bank.input, bank.ids_storage, 16
                    )
            ready_stream.synchronize()
            ready_times = []
            for _ in range(10):
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                ready_graph.replay()
                end.record()
                end.synchronize()
                ready_times.append(start.elapsed_time(end))
            records[-1]["ready_collect_ms"] = statistics.median(ready_times)
            records[-1]["ready_collect_samples"] = ready_times
            fused_ready_graph = None
            if "fused" in modes:
                fused_bank = session.banks[(n, True, True)]
                fused_ready_graph = torch.npu.NPUGraph()
                with torch.npu.stream(ready_stream):
                    with torch.npu.graph(fused_ready_graph):
                        session.kernels.call(
                            session.collect_fused,
                            fused_bank.config,
                            fused_bank.input,
                            fused_bank.ids_storage,
                            16,
                        )
                ready_stream.synchronize()
                fused_times = []
                for _ in range(10):
                    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
                        enable_timing=True
                    )
                    start.record()
                    fused_ready_graph.replay()
                    end.record()
                    end.synchronize()
                    fused_times.append(start.elapsed_time(end))
                records[-1]["ready_fused_ms"] = statistics.median(fused_times)
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
                for mode, (overlap, pack) in modes.items():
                    with torch.profiler.record_function("shared_" + mode):
                        graphs[mode].replay()
                        torch.npu.synchronize()
                with torch.profiler.record_function("collect_already_ready"):
                    ready_graph.replay()
                    torch.npu.synchronize()
                if fused_ready_graph is not None:
                    with torch.profiler.record_function("fused_collect_already_ready"):
                        fused_ready_graph.replay()
                        torch.npu.synchronize()
            if fused_ready_graph is not None:
                fused_ready_graph.reset()
            ready_graph.reset()
        for graph in graphs.values():
            graph.reset()
        print(records[-1], flush=True)
count = session.close()
(a.directory / f"client{a.source}.json").write_text(
    json.dumps(
        dict(status="PASS", calls=count, cases=records, pack_checks=pack_checks),
        indent=2,
    )
)
