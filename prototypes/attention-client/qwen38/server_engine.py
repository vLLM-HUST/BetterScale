"""Persistent Qwen38 engine: unchanged admission policy, mixed numeric catalog.

Whole up/down readiness is intentional for this first INT8 gate. Reject fine
pipeline flags rather than silently interpreting their BF16 scratch ABI.
"""

import ctypes as C
import json
from pathlib import Path

import torch
import torch_npu


class Engine:
    def __init__(
        self, build, sources, outputs, catalog, owner, *, tasks=1, open_service=False
    ):
        root = Path(build)
        abi = json.loads((root / "abi.json").read_text())
        assert (
            abi.get("kernel_timeout_us") == 1200000000
        ), "Reject microbench launch lifetime"
        assert abi["model"] == "qwen38" and abi["client_config_words"] == 17
        assert not abi["prefix_pipeline"]
        assert len(sources) == len(outputs) == 2 and 0 <= owner < 4
        assert 1 <= tasks <= 32 and 1 <= len(catalog) <= 49
        self.catalog = catalog
        self.weight_table = torch.tensor(
            [
                [t.data_ptr() if t is not None else 0 for t in layer]
                for layer in catalog
            ],
            dtype=torch.int64,
            device="npu",
        )
        assert self.weight_table.shape == (len(catalog), 4)
        for index, (up, down, su, sd) in enumerate(catalog):
            assert up.shape == (128, 2560, 1280) and down.shape == (128, 640, 2560)
            assert (
                up.dtype == down.dtype == (torch.int8 if index < 48 else torch.bfloat16)
            )
            assert torch_npu.get_npu_format(up) == torch_npu.get_npu_format(down) == 29
            if index < 48:
                assert su.shape == (128, 1280) and sd.shape == (128, 2560)
                assert su.dtype == sd.dtype == torch.float32
        self.control = torch.zeros(128, 16, dtype=torch.int32, device="npu")
        self.trace = torch.full((tasks * 2, 16), -991, dtype=torch.int32, device="npu")
        self.events = torch.zeros(512, 8, dtype=torch.int64, device="npu")
        self.slots = []
        self.auxiliary = []
        table = []
        capacity = 640
        for _ in range(2):
            slot = [
                torch.empty(2, 32, 2560, dtype=torch.bfloat16, device="npu"),
                torch.empty(capacity, 2560, dtype=torch.bfloat16, device="npu"),
                torch.empty(capacity, 1280, dtype=torch.int32, device="npu"),
                torch.empty(capacity, 640, dtype=torch.bfloat16, device="npu"),
                torch.empty(capacity, 2560, dtype=torch.int32, device="npu"),
                torch.zeros(2, 328, dtype=torch.int32, device="npu"),
                torch.zeros(128, dtype=torch.int64, device="npu"),
            ]
            for w, k, n in ((catalog[0][0], 2560, 1280), (catalog[0][1], 640, 2560)):
                slot.append(
                    torch.tensor(
                        [k, n, 128, w.data_ptr(), slot[6].data_ptr(), capacity],
                        dtype=torch.int64,
                        device="npu",
                    )
                )
            scales = [
                torch.empty(n, 8, dtype=torch.float32, device="npu")
                for n in (64, capacity, capacity)
            ]
            aux = torch.tensor(
                [t.data_ptr() for t in scales], dtype=torch.int64, device="npu"
            )
            self.auxiliary.append((scales, aux))
            self.slots.append(slot)
            table.append([p.data_ptr() for p in slot] + [0] * 6 + [aux.data_ptr()])
        self.table = torch.tensor(table, dtype=torch.int64, device="npu")
        values = [
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
            0,
            *([0] * 10),
            int(open_service),
            self.weight_table.data_ptr(),
            len(catalog),
        ]
        assert len(values) == 27
        self.config = torch.tensor(values, dtype=torch.int64, device="npu")
        self.lib = C.CDLL(str(root / "launch.so"))
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
        self.streams = []
        self.graphs = []
        for name, blocks, launch in (
            ("persistent_vector", 17, self.lib.launch_blocks),
            ("persistent_cube", 24, self.lib.launch_cube),
        ):
            binary, fn = C.c_void_p(), C.c_void_p()
            assert (
                self.lib.load_server(
                    str(root / f"{name}.o").encode(),
                    name.encode(),
                    C.byref(binary),
                    C.byref(fn),
                )
                == 0
            )
            self.binaries.append(binary)
            stream = torch.npu.Stream()
            graph = torch.npu.NPUGraph()
            with torch.npu.stream(stream):
                with torch.npu.graph(graph):
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
            self.streams.append(stream)
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
        control = self.control.cpu().tolist()
        assert control[0][0] == control[43][0] == 1, control[:3] + control[43:44]
        return dict(
            waves=control[43][1],
            completed_counts=control[43][4:6],
            admitted_promotions=control[43][6],
            trace=self.trace.cpu().tolist(),
            events=self.events[: control[43][3]].cpu().tolist(),
        )

    def close(self):
        for graph in self.graphs:
            graph.reset()
        for binary in self.binaries:
            assert self.lib.unload_server(binary) == 0
