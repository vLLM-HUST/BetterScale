"""Client-owned BF16 publication and two-server retirement reference."""

import time
import torch
import torch_npu
from common import *
from vllm_ascend.ops.fused_moe.experts_selector import select_experts


class RemoteExperts:
    def __init__(self, model, links, audit):
        self.links, self.audit = links, audit
        self.api = acl_api()
        self.stream = torch.npu.current_stream()
        self.local = self.api.allocate_staging(ALLOCATION_BYTES)
        for pipe in links:
            pipe.send(("pid", self.api.pid()))
        pids = [recv(p)[1] for p in links]
        self.key = self.api.export(self.local, ALLOCATION_BYTES, tuple(pids))
        for pipe in links:
            pipe.send(("key", self.key))
        self.peer_keys = [recv(p)[1] for p in links]
        self.peers = [self.api.import_memory(key) for key in self.peer_keys]
        # Preserve the native loaded expert tensors exactly, including dummy seed.
        for layer_id, layer in enumerate(model.layers):
            experts = layer.mlp.experts
            if hasattr(experts, "routed_experts"):
                experts = experts.routed_experts
            w13 = torch_npu.npu_format_cast(experts.w13_weight.data, 2).contiguous()
            w2 = torch_npu.npu_format_cast(experts.w2_weight.data, 2).contiguous()
            assert tuple(w13.shape) == (E, H, 2 * M), w13.shape
            assert tuple(w2.shape) == (E, M, H), w2.shape
            base = self.local + layer_id * LAYER_BYTES
            self.api.copy(self.stream.npu_stream, base, w13.data_ptr(), W13_BYTES)
            self.api.copy(
                self.stream.npu_stream, base + W13_BYTES, w2.data_ptr(), W2_BYTES
            )
            self.stream.synchronize()
        for pipe in links:
            pipe.send(("weights_ready",))
        assert all(recv(p)[0] == "weights_loaded" for p in links)
        self.generation = 0
        self.pending = None
        self.received = set()
        self.outputs = [
            torch.empty((CAP * K, H), device="npu", dtype=torch.bfloat16) for _ in links
        ]

    def submit(self, identity, layer_id, layer, pending):
        assert self.pending is None, "source still owned by prior generation"
        hidden = pending.normalized
        rows = hidden.shape[0]
        assert 0 < rows <= CAP
        logits, _ = layer.mlp.gate(hidden)
        weights, ids = select_experts(hidden, logits, K, False, True, num_experts=E)
        weights = weights.to(hidden.dtype)
        host_ids = ids.cpu().tolist()  # reference control plane, not performance path
        self.generation += 1
        self.received = set()
        self.pending = pending, weights, rows, time.monotonic()
        self.api.copy(
            self.stream.npu_stream,
            self.local + INPUT_OFFSET,
            hidden.data_ptr(),
            rows * H * 2,
        )
        self.stream.synchronize()  # Publish only after the payload is complete.
        for server, pipe in enumerate(self.links):
            pipe.send(("run", self.generation, layer_id, rows, host_ids))
        self.audit.append(
            dict(
                event="submit",
                generation=self.generation,
                layer=layer_id,
                rows=rows,
                owners=sorted({e // SHARD for row in host_ids for e in row}),
                time=time.monotonic(),
            )
        )
        return self.generation

    def poll(self, handle):
        assert handle == self.generation and self.pending is not None
        for server, pipe in enumerate(self.links):
            if server in self.received or not pipe.poll():
                continue
            kind, generation = recv(pipe)
            assert kind == "done" and generation == handle, (kind, generation, handle)
            self.api.copy(
                self.stream.npu_stream,
                self.outputs[server].data_ptr(),
                self.peers[server],
                RESULT_BYTES,
            )
            self.received.add(server)
        if len(self.received) != len(self.links):
            if time.monotonic() - self.pending[3] > 300:
                raise TimeoutError("expert completion timeout")
            return None
        _, weights, rows, _ = self.pending
        slots = (self.outputs[0] + self.outputs[1])[: rows * K].view(rows, K, H)
        output = (
            (slots.float() * weights.float().unsqueeze(-1))
            .sum(dim=1)
            .to(torch.bfloat16)
        )
        # Stream order consumes remote copies before any new source publication.
        self.stream.synchronize()
        self.pending = None
        self.audit.append(
            dict(
                event="retire",
                generation=handle,
                servers=sorted(self.received),
                time=time.monotonic(),
            )
        )
        return output

    def close(self):
        assert self.pending is None
        for pipe in self.links:
            pipe.send(("stop",))
        assert all(recv(p)[0] == "drained" for p in self.links)
        torch.npu.synchronize()
        for key in self.peer_keys:
            self.api.close_mapping(key)
        for pipe in self.links:
            pipe.send(("unmapped",))
        assert all(recv(p)[0] == "unmapped" for p in self.links)
        self.api.close_mapping(self.key)
        self.api.free_staging(self.local)
