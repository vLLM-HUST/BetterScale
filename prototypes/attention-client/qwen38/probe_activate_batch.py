"""Same-device ACTIVATE-only QuantRow vs four-row batch; no GEMM/transport."""

import argparse
import ctypes as C
import json
from pathlib import Path
import statistics

import torch
import torch_npu

p = argparse.ArgumentParser()
p.add_argument("--build", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(2)
torch.npu.set_device(0)
torch.manual_seed(138)
lib = C.CDLL(str(a.build / "launch.so"))
lib.load_server.argtypes = [
    C.c_char_p,
    C.c_char_p,
    C.POINTER(C.c_void_p),
    C.POINTER(C.c_void_p),
]
lib.launch_blocks.argtypes = [C.c_void_p] * 5 + [C.c_uint32]
binary, fn = C.c_void_p(), C.c_void_p()
assert (
    lib.load_server(
        str(a.build / "activate_probe.o").encode(),
        b"activate_probe",
        C.byref(binary),
        C.byref(fn),
    )
    == 0
)


def call(cfg, x, out):
    assert (
        lib.launch_blocks(
            fn,
            torch.npu.current_stream().npu_stream,
            cfg.data_ptr(),
            x.data_ptr(),
            out.data_ptr(),
            16,
        )
        == 0
    )


records = []
for capacity in (1, 7, 32, 129, 1024, 4096, 10003):
    groups = 171
    x = torch.empty(capacity, 1280, device="npu", dtype=torch.int32)
    channel = torch.rand(groups, 1280, device="npu") * 0.03
    scale = torch.zeros(capacity, 8, device="npu")
    scale[:, 0] = torch.rand(capacity, device="npu") * 0.04
    ends = torch.zeros(groups, dtype=torch.int64, device="npu")
    ys = [
        torch.full((capacity + 1, 640), 73, device="npu", dtype=torch.int8)
        for _ in range(2)
    ]
    ss = [torch.full((capacity + 1, 8), -7.0, device="npu") for _ in range(2)]
    cfgs = [
        torch.tensor(
            [
                capacity,
                groups,
                ends.data_ptr(),
                channel.data_ptr(),
                scale.data_ptr(),
                ss[mode].data_ptr(),
                mode,
            ],
            dtype=torch.int64,
            device="npu",
        )
        for mode in range(2)
    ]
    for pattern in ("single", "sparse", "wide"):
        counts = torch.zeros(groups, dtype=torch.int64)
        ids = (
            [170]
            if pattern == "single"
            else ([0, 1, 169, 170] if pattern == "sparse" else list(range(groups)))
        )
        for row in range(capacity):
            counts[ids[row % len(ids)]] += 1
        ends.copy_(counts.cumsum(0))
        x.copy_(torch.randint(-50000, 50001, x.shape, dtype=torch.int32, device="npu"))
        x[0].zero_()
        graphs = []
        for mode in range(2):
            call(cfgs[mode], x, ys[mode])
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                call(cfgs[mode], x, ys[mode])
            graphs.append(graph)
        for generation in range(3):
            x.copy_(
                torch.randint(-50000, 50001, x.shape, dtype=torch.int32, device="npu")
            )
            x[0].zero_()
            # Change expert boundaries at fixed addresses inside FULL replay.
            counts = counts.roll(1)
            ends.copy_(counts.cumsum(0))
            channel.copy_(torch.rand_like(channel) * 0.03)
            for graph in graphs:
                graph.replay()
            torch.npu.synchronize()
            assert torch.equal(ys[0], ys[1]), (
                capacity,
                pattern,
                generation,
                "quant bytes",
            )
            assert torch.equal(ss[0], ss[1]), (
                capacity,
                pattern,
                generation,
                "scale bytes",
            )
            assert torch.all(ys[0][-1] == 73) and torch.all(ss[0][-1] == -7)
            assert torch.all(ys[0][0] == 0) and float(ss[0][0, 0]) == 1.0
        times = [[], []]
        for repeat in range(12):
            for mode in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
                    enable_timing=True
                )
                start.record()
                graphs[mode].replay()
                end.record()
                end.synchronize()
                times[mode].append(start.elapsed_time(end))
        # Reduce active M in the captured graph, check inactive tail is untouched.
        if capacity > 1:
            active = capacity // 2
            counts.zero_()
            counts[170] = active
            ends.copy_(counts.cumsum(0))
            for mode in range(2):
                cfgs[mode][0] = active
                ys[mode].fill_(73)
                ss[mode].fill_(-7.0)
                graphs[mode].replay()
            torch.npu.synchronize()
            assert torch.equal(ys[0], ys[1]) and torch.equal(ss[0], ss[1])
            assert torch.all(ys[1][active:] == 73) and torch.all(ss[1][active:] == -7)
            for cfg in cfgs:
                cfg[0] = capacity
        for graph in graphs:
            graph.reset()
        records.append(
            dict(
                rows=capacity,
                pattern=pattern,
                exact=True,
                median_ms=[statistics.median(t) for t in times],
                samples_ms=times,
            )
        )
        print(capacity, pattern, records[-1]["median_ms"], flush=True)
a.output.write_text(
    json.dumps(
        dict(
            status="PASS",
            scope="ACTIVATE-only, dummy INT32 intermediates, same binary 16 AIV, FULL changed-state replay",
            cases=records,
        ),
        indent=2,
    )
    + "\n"
)
