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
PARALLEL = os.environ.get("DEVICE_SERVICE_PARALLEL") == "1"


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
        self.lib.launch_blocks.argtypes = [C.c_void_p] * 5 + [C.c_uint32]
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

    def call(self, fn, config, a, b, blocks=1):
        rc = self.lib.launch_blocks(
            fn,
            torch.npu.current_stream().npu_stream,
            config.data_ptr(),
            a.data_ptr(),
            b.data_ptr(),
            blocks,
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
                int(PARALLEL),
            ],
            device="npu",
            dtype=torch.int64,
        )
        self.indices = torch.arange(rows * K, dtype=torch.int32, device="npu")
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
        if PARALLEL:
            self.transport.kernels.call(
                self.transport.collect, self.config, self.input, ids, blocks=16
            )
            self.transport.kernels.call(
                self.transport.retire, self.config, self.input, ids
            )
            return torch_npu.npu_moe_token_unpermute(
                self.raw[0].view(-1, H), self.indices, probs=weights.to(torch.bfloat16)
            )
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
        if PARALLEL:
            self.collect = self.kernels.load("neural_collect")
            self.retire = self.kernels.load("neural_retire")
        self.counter = torch.zeros(8, dtype=torch.int32, device="npu")
        self.banks = {}
        self.pending = None
        self.generation = 0
        self.sealed = False
        self.server_ready = False

    def activate(self, enqueue=None):
        if self.server_ready:
            return
        # Host initialization rendezvous only: no device polls cover compilation.
        assert all(recv(p, timeout=300)[0] == "server_ready" for p in self.links)
        if enqueue is not None:
            enqueue()
        for p in self.links:
            p.send(("client_ready",))
        assert all(recv(p, timeout=300)[0] == "server_started" for p in self.links)
        self.server_ready = True

    def submit(self, identity, layer_id, layer, pending):
        assert self.pending is None
        rows = pending.normalized.shape[0]
        key = layer_id, rows
        if key not in self.banks:
            assert not self.sealed, "unprepared client graph"
            self.banks[key] = ClientBank(self, layer_id, layer, pending.normalized)
        bank = self.banks[key]
        self.activate()
        bank.input.copy_(pending.normalized)
        # External timing brackets only the client graph (input copy excluded).
        # They do not change its publication/completion protocol.
        timing = bool(os.environ.get("DEVICE_SERVICE_TIMING"))
        start = torch.npu.Event(enable_timing=True) if timing else None
        if start is not None:
            start.record()
        bank.graph.replay()
        event = torch.npu.Event(enable_timing=timing)
        event.record()
        self.pending_timing = start, rows
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
        start, rows = self.pending_timing
        record = dict(
            event="retire",
            generation=handle,
            rows=rows,
            time=time.monotonic(),
            completion_device=True,
        )
        if start is not None:
            record["client_graph_us"] = start.elapsed_time(event) * 1000
        self.audit.append(record)
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
        if PARALLEL:
            # Poison unowned slots: owner-directed collection must never read them.
            poison = torch.full(
                (ALIGN // 4,), 0x7FC07FC0, dtype=torch.int32, device="npu"
            )
            api.copy(stream.npu_stream, local, poison.data_ptr(), ALIGN)
            stream.synchronize()
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
        # completion. A separate server_ready follows conversion and capture;
        # device timeout budgets must not be spent waiting for host compilation.
        pipe.send(("weights_loaded",))
    w13 = torch.cat([w[0] for w in weights])
    w2 = torch.cat([w[1] for w in weights])
    del weights, current, up, down
    actual_counts = os.environ.get("DEVICE_SERVICE_ACTUAL_COUNTS") == "1"
    persistent = os.environ.get("DEVICE_SERVICE_PERSISTENT") == "1"
    weight_format = os.environ.get(
        "DEVICE_SERVICE_WEIGHT_FORMAT", "NZ" if actual_counts or persistent else "ND"
    )
    assert not (actual_counts or persistent) or (PARALLEL and weight_format == "NZ")
    assert weight_format in ("ND", "NZ")
    if weight_format == "NZ":
        torch_npu.npu.config.allow_internal_format = True
        w13 = torch_npu.npu_format_cast(w13, 29)
        w2 = torch_npu.npu_format_cast(w2, 29)
        assert torch_npu.get_npu_format(w13) == 29
        assert torch_npu.get_npu_format(w2) == 29
    if persistent:
        assert int(os.environ.get("DEVICE_SERVICE_COALESCE_POLLS", "0")) == 0
        assert os.environ.get("DEVICE_SERVICE_PAIRED_CONTROL") != "1"
        from persistent_service import serve as persistent_serve

        persistent_serve(api, clients, w13, w2, server_id, output_path)
        return
    kernels = Kernels()
    prepare = kernels.load("neural_prepare")
    complete = kernels.load("neural_complete")
    if PARALLEL:
        pack = kernels.load("neural_pack")
        scatter = kernels.load("neural_scatter")
    packed = torch.zeros((2 * CAP * K, H), device="npu", dtype=torch.bfloat16)
    groups = torch.zeros(2 * SHARD, device="npu", dtype=torch.int64)
    groups[-1] = 2 * CAP * K
    state = torch.zeros(8, device="npu", dtype=torch.int32)
    batch = torch.zeros(2 * (8 + CAP * K), device="npu", dtype=torch.int32)
    trace = torch.full((WAVES, 8), -991, device="npu", dtype=torch.int32)
    paired_control = os.environ.get("DEVICE_SERVICE_PAIRED_CONTROL") == "1"
    assert not paired_control or PARALLEL
    coalesce_polls = int(os.environ.get("DEVICE_SERVICE_COALESCE_POLLS", "0"))
    assert 0 <= coalesce_polls <= 128
    assert not coalesce_polls or (PARALLEL and not paired_control)
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
            int(PARALLEL)
            | (2 if paired_control else 0)
            | (4 if actual_counts else 0)
            | (8 if coalesce_polls else 0),
            coalesce_polls,
        ],
        device="npu",
        dtype=torch.int64,
    )

    if actual_counts:
        from actual_gmm import ActualGmm

        actual_up = ActualGmm(w13, groups)
        actual_down = ActualGmm(w2, groups)
        up_buffer = torch.empty(
            (2 * CAP * K, 2 * M), device="npu", dtype=torch.bfloat16
        )
        down_buffer = torch.empty_like(packed)

    def body():
        if not actual_counts:
            packed.zero_()
        kernels.call(prepare, config, packed, state)
        if PARALLEL:
            kernels.call(pack, config, packed, state, blocks=16)
        if actual_counts:
            up = actual_up(packed, up_buffer)
            act = torch_npu.npu_swiglu(up)
            down = actual_down(act, down_buffer)
        else:
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
        if PARALLEL:
            kernels.call(scatter, config, down, state, blocks=16)
        kernels.call(complete, config, down, state)

    body()
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(WAVES):
            body()
    stream.synchronize()
    config[5] = 1
    from profile_capture import start, stop

    for c in clients:
        c["pipe"].send(("server_ready",))
    assert all(recv(c["pipe"], timeout=300)[0] == "client_ready" for c in clients)
    profiler = start(f"expert{server_id}")
    graph.replay()
    for c in clients:
        c["pipe"].send(("server_started",))
    stream.synchronize()
    status = state.cpu().tolist()
    assert status[:4] == [TASKS, TASKS, WAVES, 0], status
    records = trace.cpu().tolist()
    assert sum(r[0] for r in records) == TASKS * 2
    if actual_counts or coalesce_polls:
        # New modes must not silently run an older captured queue binary.
        expected_flags = (
            int(PARALLEL)
            | (2 if paired_control else 0)
            | (4 if actual_counts else 0)
            | (8 if coalesce_polls else 0)
        )
        assert all(
            r[6] == expected_flags and 0 <= r[5] <= coalesce_polls for r in records
        )
    if paired_control:
        # Reject a stale binary or accidentally unpaired execution, not merely
        # successful client outputs from a different batch organization.
        live_records = [r for r in records if r[0]]
        assert len(live_records) == TASKS
        assert all(r[0] == 2 and r[2] == r[3] for r in live_records)
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
    stop(profiler)
    graph.reset()
    if actual_counts:
        actual_up.close()
        actual_down.close()
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
                parallel_transport=PARALLEL,
                weight_format=weight_format,
                paired_control=paired_control,
                actual_counts=actual_counts,
                coalesce_polls=coalesce_polls,
            ),
            indent=2,
        )
    )
