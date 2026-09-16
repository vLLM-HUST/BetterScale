"""Bounded four-card neural service with device routing and completion.

The host grants a finite episode budget. There are no per-layer descriptors,
CPU top-k reads or server-side host batch decisions inside that episode.
"""

import ctypes as C
import json
import os
from pathlib import Path
import time
import torch
import torch_npu
import common
from common import (
    H,
    M,
    E,
    K,
    LAYERS,
    CAP,
    SHARD,
    ALIGN,
    W13_BYTES,
    W2_BYTES,
    LAYER_BYTES,
    INPUT_OFFSET,
    ALLOCATION_BYTES,
    acl_api,
    recv,
)

TASKS = 24  # two layers × twelve native forwards per client in the bounded fixture
WAVES = TASKS * 2


class Kernels:
    def __init__(self):
        build = Path(os.environ["DEVICE_SERVICE_BUILD"])
        self.path = build / "queue_service.o"
        self.lib = C.CDLL(str(build / "launch.so"))
        self.lib.load_server.argtypes = [
            C.c_char_p,
            C.c_char_p,
            C.POINTER(C.c_void_p),
            C.POINTER(C.c_void_p),
        ]
        self.lib.launch_server.argtypes = [C.c_void_p] * 5
        self.lib.unload_server.argtypes = [C.c_void_p]
        self.binaries = []

    def load(self, name):
        binary, fn = C.c_void_p(), C.c_void_p()
        rc = self.lib.load_server(
            str(self.path).encode(), name.encode(), C.byref(binary), C.byref(fn)
        )
        assert rc == 0, (name, rc)
        self.binaries.append(binary)
        return fn

    def call(self, fn, config, a, b):
        rc = self.lib.launch_server(
            fn,
            torch.npu.current_stream().npu_stream,
            config.data_ptr(),
            a.data_ptr(),
            b.data_ptr(),
        )
        assert rc == 0, rc

    def close(self):
        for binary in self.binaries:
            assert self.lib.unload_server(binary) == 0


class ClientBank:
    def __init__(self, transport, layer_id, layer, hidden):
        self.transport = transport
        self.layer = layer
        rows = hidden.shape[0]
        self.input = hidden.clone()
        self.raw = [
            torch.zeros((rows, K, H), dtype=torch.bfloat16, device="npu")
            for _ in range(2)
        ]
        self.config = torch.tensor(
            [
                transport.local + INPUT_OFFSET,
                *transport.peers,
                layer_id,
                rows,
                transport.counter.data_ptr(),
                0,
                2000000,
                *[x.data_ptr() for x in self.raw],
            ],
            device="npu",
            dtype=torch.int64,
        )
        self.body()
        torch.npu.synchronize()
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph):
            self.output = self.body()
        torch.npu.synchronize()
        self.config[6] = 1

    def body(self):
        logits, _ = self.layer.mlp.gate(self.input)
        # Import only after native platform initialization (the server process
        # must not pull in the worker/attention import cycle).
        from vllm_ascend.ops.fused_moe.experts_selector import select_experts

        weights, ids = select_experts(self.input, logits, K, False, True, num_experts=E)
        ids = ids.to(torch.int32)
        self.transport.kernels.call(self.transport.fn, self.config, self.input, ids)
        return (
            (
                (self.raw[0] + self.raw[1]).float()
                * weights.to(torch.bfloat16).float().unsqueeze(-1)
            )
            .sum(dim=1)
            .to(torch.bfloat16)
        )


class DeviceExperts:
    def __init__(self, model, links, audit):
        self.links, self.audit = links, audit
        self.api = acl_api()
        self.stream = torch.npu.current_stream()
        self.local = self.api.allocate_staging(ALLOCATION_BYTES)
        # Initialize READY before any peer can observe the source allocation.
        zeros = torch.zeros(1024, dtype=torch.int32, device="npu")
        self.api.copy(
            self.stream.npu_stream, self.local + INPUT_OFFSET, zeros.data_ptr(), 4096
        )
        self.stream.synchronize()
        for pipe in links:
            pipe.send(("pid", self.api.pid()))
        pids = [recv(p)[1] for p in links]
        self.key = self.api.export(self.local, ALLOCATION_BYTES, tuple(pids))
        for pipe in links:
            pipe.send(("key", self.key))
        self.peer_keys = [recv(p)[1] for p in links]
        self.peers = [self.api.import_memory(k) for k in self.peer_keys]
        for i, layer in enumerate(model.layers):
            experts = layer.mlp.experts
            if hasattr(experts, "routed_experts"):
                experts = experts.routed_experts
            up = torch_npu.npu_format_cast(experts.w13_weight.data, 2).contiguous()
            down = torch_npu.npu_format_cast(experts.w2_weight.data, 2).contiguous()
            assert tuple(up.shape) == (E, H, 2 * M) and tuple(down.shape) == (E, M, H)
            base = self.local + i * LAYER_BYTES
            self.api.copy(self.stream.npu_stream, base, up.data_ptr(), W13_BYTES)
            self.api.copy(
                self.stream.npu_stream, base + W13_BYTES, down.data_ptr(), W2_BYTES
            )
            self.stream.synchronize()
        for p in links:
            p.send(("weights_ready",))
        assert all(recv(p, timeout=300)[0] == "weights_loaded" for p in links)
        self.kernels = Kernels()
        self.fn = self.kernels.load("neural_client")
        self.counter = torch.zeros(8, dtype=torch.int32, device="npu")
        self.banks = {}
        self.pending = None
        self.generation = 0
        self.sealed = False

    def submit(self, identity, layer_id, layer, pending):
        assert self.pending is None
        rows = pending.normalized.shape[0]
        key = layer_id, rows
        if key not in self.banks:
            assert not self.sealed, "unprepared client graph"
            self.banks[key] = ClientBank(self, layer_id, layer, pending.normalized)
        bank = self.banks[key]
        bank.input.copy_(pending.normalized)
        bank.graph.replay()
        event = torch.npu.Event()
        event.record()
        self.generation += 1
        self.pending = bank, event, pending, time.monotonic()
        # Host records only the invocation envelope, never materializes routing.
        self.audit.append(
            dict(
                event="submit",
                generation=self.generation,
                layer=layer_id,
                rows=rows,
                time=time.monotonic(),
                routing_device=True,
            )
        )
        return self.generation

    def poll(self, handle):
        assert handle == self.generation and self.pending is not None
        bank, event, _, started = self.pending
        if not event.query():
            if time.monotonic() - started > 120:
                raise TimeoutError("device expert episode")
            return None
        torch.npu.current_stream().wait_event(event)
        self.pending = None
        self.audit.append(
            dict(
                event="retire",
                generation=handle,
                time=time.monotonic(),
                completion_device=True,
            )
        )
        return bank.output

    def close(self):
        assert self.pending is None
        torch.npu.synchronize()
        assert self.counter.cpu().tolist()[0] == TASKS
        for p in self.links:
            p.send(("stop",))
        assert all(recv(p)[0] == "drained" for p in self.links)
        for bank in self.banks.values():
            bank.graph.reset()
        self.kernels.close()
        for k in self.peer_keys:
            self.api.close_mapping(k)
        for p in self.links:
            p.send(("unmapped",))
        assert all(recv(p)[0] == "unmapped" for p in self.links)
        self.api.close_mapping(self.key)
        self.api.free_staging(self.local)


def serve(server_id, links, output_path):
    api = acl_api()
    stream = torch.npu.current_stream()
    clients = []
    weights = []
    for client_id, pipe in enumerate(links):
        kind, pid = recv(pipe)
        assert kind == "pid"
        pipe.send(("pid", api.pid()))
        local = api.allocate_staging(ALIGN)
        zeros = torch.zeros(8, dtype=torch.int32, device="npu")
        api.copy(stream.npu_stream, local, zeros.data_ptr(), 32)
        stream.synchronize()
        key = api.export(local, ALIGN, (pid,))
        kind, peer_key = recv(pipe)
        assert kind == "key"
        peer = api.import_memory(peer_key)
        pipe.send(("key", key))
        assert recv(pipe)[0] == "weights_ready"
        current = []
        for layer in range(LAYERS):
            up = torch.empty((SHARD, H, 2 * M), device="npu", dtype=torch.bfloat16)
            down = torch.empty((SHARD, M, H), device="npu", dtype=torch.bfloat16)
            base = peer + layer * LAYER_BYTES
            api.copy(
                stream.npu_stream,
                up.data_ptr(),
                base + server_id * (W13_BYTES // 2),
                W13_BYTES // 2,
            )
            api.copy(
                stream.npu_stream,
                down.data_ptr(),
                base + W13_BYTES + server_id * (W2_BYTES // 2),
                W2_BYTES // 2,
            )
            stream.synchronize()
            if client_id:
                assert torch.equal(up, weights[layer][0]) and torch.equal(
                    down, weights[layer][1]
                )
            else:
                current.append((up, down))
        if not client_id:
            weights = current
        clients.append(
            dict(pipe=pipe, local=local, key=key, peer=peer, peer_key=peer_key)
        )
        # This acknowledgement releases the source's bootstrap, not a service
        # completion. Its first client graph may publish and wait while we finish
        # preparing the server graph. Polls/timeouts bound that startup wait.
        pipe.send(("weights_loaded",))
    w13 = torch.cat([w[0] for w in weights])
    w2 = torch.cat([w[1] for w in weights])
    del weights, current, up, down
    kernels = Kernels()
    prepare = kernels.load("neural_prepare")
    complete = kernels.load("neural_complete")
    packed = torch.zeros((2 * CAP * K, H), device="npu", dtype=torch.bfloat16)
    groups = torch.zeros(2 * SHARD, device="npu", dtype=torch.int64)
    groups[-1] = 2 * CAP * K
    state = torch.zeros(8, device="npu", dtype=torch.int32)
    batch = torch.zeros(2 * (8 + CAP * K), device="npu", dtype=torch.int32)
    trace = torch.full((WAVES, 8), -991, device="npu", dtype=torch.int32)
    config = torch.tensor(
        [
            *[c["local"] for c in clients],
            *[c["peer"] + INPUT_OFFSET for c in clients],
            TASKS,
            0,
            2000000,
            state.data_ptr(),
            batch.data_ptr(),
            groups.data_ptr(),
            trace.data_ptr(),
            server_id,
        ],
        device="npu",
        dtype=torch.int64,
    )

    def body():
        packed.zero_()
        kernels.call(prepare, config, packed, state)
        up = torch_npu.npu_grouped_matmul(
            [packed],
            [w13],
            split_item=2,
            group_list=groups,
            group_type=0,
            group_list_type=0,
        )[0]
        act = torch_npu.npu_swiglu(up)
        down = torch_npu.npu_grouped_matmul(
            [act],
            [w2],
            split_item=2,
            group_list=groups,
            group_type=0,
            group_list_type=0,
        )[0]
        kernels.call(complete, config, down, state)

    body()
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(WAVES):
            body()
    stream.synchronize()
    config[5] = 1
    graph.replay()
    stream.synchronize()
    status = state.cpu().tolist()
    assert status[:4] == [TASKS, TASKS, WAVES, 0], status
    records = trace.cpu().tolist()
    assert sum(r[0] for r in records) == TASKS * 2
    seen = [[], []]
    for r in records:
        assert r[4] == 0
        for c in range(2):
            if r[2 + c]:
                seen[c].append(r[2 + c])
    assert seen == [list(range(1, TASKS + 1))] * 2
    for c in clients:
        pipe = c["pipe"]
        assert recv(pipe)[0] == "stop"
        pipe.send(("drained",))
        assert recv(pipe)[0] == "unmapped"
        api.close_mapping(c["peer_key"])
        api.close_mapping(c["key"])
        api.free_staging(c["local"])
        pipe.send(("unmapped",))
    graph.reset()
    kernels.close()
    Path(output_path).write_text(
        json.dumps(
            dict(
                server=server_id,
                status=status,
                trace=records,
                same_graph_cross_source_waves=sum(r[0] == 2 for r in records),
                host_control=False,
                graph_replays=1,
                bounded_waves=WAVES,
            ),
            indent=2,
        )
    )
