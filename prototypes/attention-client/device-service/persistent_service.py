"""Bounded persistent AIV/AIC engine pair; native client wire remains unchanged."""

import ctypes as C
import json
import os
from pathlib import Path

import torch
import torch_npu


class PersistentEngine:
    def __init__(self, sources, outputs, up, down, owner, tasks=24):
        assert len(sources) == len(outputs) == 2 and owner in (0, 1)
        assert 1 <= tasks <= 24
        assert up.shape == (128, 2048, 1536) and down.shape == (128, 768, 2048)
        assert up.dtype == down.dtype == torch.bfloat16
        assert (
            int(torch_npu.get_npu_format(up))
            == int(torch_npu.get_npu_format(down))
            == 29
        )
        self.resident_moves = os.environ.get("DEVICE_SERVICE_RESIDENT_MOVES") == "1"
        self.move_quantum = int(os.environ.get("DEVICE_SERVICE_MOVE_QUANTUM", "0"))
        assert self.move_quantum in (0, 128, 256)
        self.tail_experts = int(
            os.environ.get("DEVICE_SERVICE_SEGMENT_TAIL_EXPERTS", "0")
        )
        assert 0 <= self.tail_experts <= 128
        self.up, self.down = up, down
        self.internal_pipeline = (
            os.environ.get("DEVICE_SERVICE_INTERNAL_PIPELINE") == "1"
        )
        self.segmented = (
            self.internal_pipeline or os.environ.get("DEVICE_SERVICE_SEGMENTED") == "1"
        )
        assert not (self.move_quantum and self.resident_moves)
        assert not (self.move_quantum or self.resident_moves) or self.internal_pipeline
        self.control = torch.zeros((96, 16), dtype=torch.int32, device="npu")
        self.trace = torch.full((tasks * 2, 16), -991, dtype=torch.int32, device="npu")
        self.events = torch.zeros((512, 8), dtype=torch.int64, device="npu")
        self.work_times = (
            torch.zeros((3, 512, 24, 8), dtype=torch.int64, device="npu")
            if os.environ.get("DEVICE_SERVICE_INTERNAL_TIMING") == "1"
            else None
        )
        self.slots = []
        table = []
        for _ in range(2):
            slot = [
                torch.empty((2, 32, 2048), dtype=torch.bfloat16, device="npu"),
                torch.empty((512, 2048), dtype=torch.bfloat16, device="npu"),
                torch.empty((512, 1536), dtype=torch.bfloat16, device="npu"),
                torch.empty((512, 768), dtype=torch.bfloat16, device="npu"),
                torch.empty((512, 2048), dtype=torch.bfloat16, device="npu"),
                torch.zeros((2, 264), dtype=torch.int32, device="npu"),
                torch.zeros(128, dtype=torch.int64, device="npu"),
            ]
            for w, k, n in ((up, 2048, 1536), (down, 768, 2048)):
                slot.append(
                    torch.tensor(
                        [k, n, 128, w.data_ptr(), slot[6].data_ptr(), 512],
                        dtype=torch.int64,
                        device="npu",
                    )
                )
            if self.segmented:
                # Two device-authored prefix catalogs, each relative to its own
                # packed row slice; weights retain the original expert IDs.
                slot.extend(
                    torch.zeros(128, dtype=torch.int64, device="npu") for _ in range(2)
                )
                for w, k, n in ((up, 2048, 1536), (down, 768, 2048)):
                    for part in range(2):
                        slot.append(
                            torch.tensor(
                                [
                                    k,
                                    n,
                                    128,
                                    w.data_ptr(),
                                    slot[9 + part].data_ptr(),
                                    512,
                                ],
                                dtype=torch.int64,
                                device="npu",
                            )
                        )
            self.slots.append(slot)
            table.append([p.data_ptr() for p in slot] + [0] * (16 - len(slot)))
        self.table = torch.tensor(table, dtype=torch.int64, device="npu")
        self.config = torch.tensor(
            [
                self.control.data_ptr(),
                self.table.data_ptr(),
                *outputs,
                *sources,
                tasks,
                owner,
                self.trace.data_ptr(),
                20000000,
                0,
                self.events.data_ptr(),
                0,
                self.work_times.data_ptr() if self.work_times is not None else 0,
                2 if self.internal_pipeline else int(self.segmented),
                self.tail_experts,
                self.move_quantum,
                int(self.resident_moves),
            ],
            dtype=torch.int64,
            device="npu",
        )
        self.root = Path(os.environ["PERSISTENT_BUILD"])
        self.lib = C.CDLL(str(self.root / "launch.so"))
        self.lib.load_server.argtypes = [
            C.c_char_p,
            C.c_char_p,
            C.POINTER(C.c_void_p),
            C.POINTER(C.c_void_p),
        ]
        self.lib.launch_blocks.argtypes = [C.c_void_p] * 5 + [C.c_uint32]
        self.lib.launch_cube.argtypes = [C.c_void_p] * 5 + [C.c_uint32]
        self.lib.unload_server.argtypes = [C.c_void_p]
        self.binaries = []
        self.functions = []
        self.streams = [torch.npu.Stream(), torch.npu.Stream()]
        self.graphs = []
        for i, (name, blocks) in enumerate(
            (("persistent_vector", 17), ("persistent_cube", 24))
        ):
            binary, fn = C.c_void_p(), C.c_void_p()
            assert (
                self.lib.load_server(
                    str(self.root / f"{name}.o").encode(),
                    name.encode(),
                    C.byref(binary),
                    C.byref(fn),
                )
                == 0
            )
            self.binaries.append(binary)
            self.functions.append(fn)
            graph = torch.npu.NPUGraph()
            with torch.npu.stream(self.streams[i]):
                with torch.npu.graph(graph):
                    launch = self.lib.launch_cube if i else self.lib.launch_blocks
                    assert (
                        launch(
                            fn,
                            torch.npu.current_stream().npu_stream,
                            self.config.data_ptr(),
                            0,
                            0,
                            blocks,
                        )
                        == 0
                    )
            self.graphs.append(graph)
        self.config[10] = 1
        torch.npu.synchronize()

    def replay(self):
        for stream, graph in zip(self.streams, self.graphs):
            with torch.npu.stream(stream):
                graph.replay()

    def finish(self):
        for stream in self.streams:
            stream.synchronize()
        ctrl = self.control.cpu().tolist()
        assert ctrl[0][0] == ctrl[43][0] == 1, ctrl[:3] + ctrl[43:44]
        records = self.trace[: ctrl[43][1]].cpu().tolist()
        receipt = dict(
            segmented=self.segmented,
            internal_pipeline=self.internal_pipeline,
            tail_experts=self.tail_experts,
            move_quantum=self.move_quantum,
            resident_moves=self.resident_moves,
            waves=len(records),
            pulls_during_cube=ctrl[43][2],
            trace=records,
            events=self.events[: ctrl[43][3]].cpu().tolist(),
        )

        if self.work_times is not None:
            timing = self.work_times.cpu()
            receipt["core_work"] = [
                [engine, gen + 1, core, *timing[engine, gen, core, :2].tolist()]
                for engine, gen, core in (timing[..., 1] > 0).nonzero().tolist()
            ]
            receipt["core_prefix"] = [
                [gen + 1, core, timing[1, gen, core, 2].item()]
                for gen, core in (timing[1, ..., 2] > 0).nonzero().tolist()
            ]
        return receipt

    def close(self):
        for graph in self.graphs:
            graph.reset()
        for binary in self.binaries:
            assert self.lib.unload_server(binary) == 0


def serve(api, clients, up, down, owner, output_path):
    from common import INPUT_OFFSET, recv
    from profile_capture import start, stop

    engine = PersistentEngine(
        [c["peer"] + INPUT_OFFSET for c in clients],
        [c["local"] for c in clients],
        up,
        down,
        owner,
    )
    for c in clients:
        c["pipe"].send(("server_ready",))
    assert all(recv(c["pipe"], timeout=300)[0] == "client_ready" for c in clients)
    profiler = start(f"expert{owner}")
    engine.replay()
    for c in clients:
        c["pipe"].send(("server_started",))
    receipt = engine.finish()
    for source in range(2):
        seen = [r[source] for r in receipt["trace"] if r[source]]
        assert seen == list(range(1, 25)), seen
    for c in clients:
        pipe = c["pipe"]
        assert recv(pipe)[0] == "stop"
        pipe.send(("drained",))
        assert recv(pipe)[0] == "unmapped"
        api.close_mapping(c["peer_key"])
        api.close_mapping(c["key"])
        api.free_staging(c["local"])
        pipe.send(("unmapped",))
    stop(profiler)
    engine.close()
    receipt.update(
        server=owner,
        persistent=True,
        deliberate_batch_wait=False,
        same_graph_cross_source_waves=sum(
            bool(r[0] and r[1]) for r in receipt["trace"]
        ),
    )
    Path(output_path).write_text(json.dumps(receipt, indent=2))
