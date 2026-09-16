"""Two-shard reference expert service; BF16 compute, host control, IPC data."""

import time
import torch
import torch_npu
from common import *


class ExpertGraph:
    def __init__(self, w13, w2):
        self.w13, self.w2 = w13, w2
        self.input = torch.zeros((CAP * K, H), device="npu", dtype=torch.bfloat16)
        self.groups = torch.zeros(SHARD, device="npu", dtype=torch.int64)
        self.groups[-1] = CAP * K
        self.graph = torch.npu.NPUGraph()
        self.forward()
        torch.npu.synchronize()
        with torch.npu.graph(self.graph):
            self.output = self.forward()
        torch.npu.synchronize()

    def forward(self):
        up = torch_npu.npu_grouped_matmul(
            [self.input],
            [self.w13],
            split_item=2,
            group_list=self.groups,
            group_type=0,
            group_list_type=0,
        )[0]
        activated = torch_npu.npu_swiglu(up, dim=-1)
        return torch_npu.npu_grouped_matmul(
            [activated],
            [self.w2],
            split_item=2,
            group_list=self.groups,
            group_type=0,
            group_list_type=0,
        )[0]


def serve(server_id, links, output_path):
    api = acl_api()
    stream = torch.npu.current_stream()
    clients = []
    audit = []
    for client_id, pipe in enumerate(links):
        kind, pid = recv(pipe)
        assert kind == "pid"
        pipe.send(("pid", api.pid()))
        local = api.allocate_staging(ALIGN)
        key = api.export(local, ALIGN, (pid,))
        kind, peer_key = recv(pipe)
        assert kind == "key"
        peer = api.import_memory(peer_key)
        pipe.send(("key", key))
        assert recv(pipe)[0] == "weights_ready"
        graphs = []
        for layer in range(LAYERS):
            w13 = torch.empty((SHARD, H, 2 * M), device="npu", dtype=torch.bfloat16)
            w2 = torch.empty((SHARD, M, H), device="npu", dtype=torch.bfloat16)
            base = peer + layer * LAYER_BYTES
            api.copy(
                stream.npu_stream,
                w13.data_ptr(),
                base + server_id * (W13_BYTES // 2),
                W13_BYTES // 2,
            )
            api.copy(
                stream.npu_stream,
                w2.data_ptr(),
                base + W13_BYTES + server_id * (W2_BYTES // 2),
                W2_BYTES // 2,
            )
            stream.synchronize()
            if clients:
                shared_graph = clients[0]["graphs"][layer]
                assert torch.equal(w13, shared_graph.w13) and torch.equal(
                    w2, shared_graph.w2
                ), "attention engines disagree on dummy model weights"
                graphs.append(shared_graph)
                del w13, w2
            else:
                graphs.append(ExpertGraph(w13, w2))
        hidden = torch.empty((CAP, H), device="npu", dtype=torch.bfloat16)
        result = torch.zeros((CAP * K, H), device="npu", dtype=torch.bfloat16)
        clients.append(
            dict(
                pipe=pipe,
                local=local,
                key=key,
                peer=peer,
                peer_key=peer_key,
                graphs=graphs,
                hidden=hidden,
                result=result,
                generation=0,
                stopped=False,
            )
        )
        pipe.send(("weights_loaded",))
    deadline = time.monotonic() + 900
    while not all(c["stopped"] for c in clients):
        progress = False
        for client_id, c in enumerate(clients):
            pipe = c["pipe"]
            if c["stopped"] or not pipe.poll():
                continue
            progress = True
            message = recv(pipe)
            if message[0] == "stop":
                stream.synchronize()
                pipe.send(("drained",))
                assert recv(pipe)[0] == "unmapped"
                api.close_mapping(c["peer_key"])
                api.close_mapping(c["key"])
                api.free_staging(c["local"])
                pipe.send(("unmapped",))
                c["stopped"] = True
                continue
            kind, generation, layer, rows, ids = message
            assert kind == "run" and generation == c["generation"] + 1
            assert 0 <= layer < LAYERS and 0 < rows <= CAP
            assert len(ids) == rows and all(
                len(row) == K and all(0 <= e < E for e in row) for row in ids
            )
            routes = sorted(
                (e % SHARD, row, slot)
                for row, experts in enumerate(ids)
                for slot, e in enumerate(experts)
                if e // SHARD == server_id
            )
            counts = [0] * SHARD
            for expert, _, _ in routes:
                counts[expert] += 1
            counts[-1] += CAP * K - len(routes)
            groups = torch.tensor(counts, dtype=torch.int64).cumsum(0)
            graph = c["graphs"][layer]
            api.copy(
                stream.npu_stream,
                c["hidden"].data_ptr(),
                c["peer"] + INPUT_OFFSET,
                rows * H * 2,
            )
            graph.input.zero_()
            if routes:
                row_indices = torch.tensor(
                    [r for _, r, _ in routes], device="npu", dtype=torch.int64
                )
                graph.input[: len(routes)].copy_(
                    c["hidden"].index_select(0, row_indices)
                )
            graph.groups.copy_(groups)
            graph.graph.replay()
            c["result"].zero_()
            if routes:
                slots = torch.tensor(
                    [r * K + s for _, r, s in routes], device="npu", dtype=torch.int64
                )
                c["result"].index_copy_(0, slots, graph.output[: len(routes)])
            api.copy(
                stream.npu_stream, c["local"], c["result"].data_ptr(), RESULT_BYTES
            )
            stream.synchronize()
            c["generation"] = generation
            audit.append(
                dict(
                    client=client_id,
                    generation=generation,
                    layer=layer,
                    rows=rows,
                    routed_rows=len(routes),
                    time=time.monotonic(),
                )
            )
            pipe.send(("done", generation))
        if time.monotonic() > deadline:
            raise TimeoutError("bounded expert service episode")
        if not progress:
            time.sleep(0.001)
    import json

    Path(output_path).write_text(
        json.dumps(
            dict(
                server=server_id,
                graphs=LAYERS,
                records=audit,
                host_control=True,
                real_bf16_gemm=True,
            ),
            indent=2,
        )
    )
