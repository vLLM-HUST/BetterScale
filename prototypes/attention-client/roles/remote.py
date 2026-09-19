"""Ordinary MoE-call backend; host sockets are used only for init and drain."""

import os
from pathlib import Path
import torch
import torch_npu
from common import acl_api, ALIGN
from control import CONTRACT, connect
from device_joint import Kernels


class Bank:
    def __init__(self, session, layer, rows):
        self.x = torch.zeros((rows, 2048), dtype=torch.bfloat16, device="npu")
        self.ids = torch.zeros((rows, 8), dtype=torch.int32, device="npu")
        self.probs = torch.zeros((rows, 8), dtype=torch.bfloat16, device="npu")
        self.raw = [
            torch.zeros((rows, 8, 2048), dtype=torch.bfloat16, device="npu")
            for _ in range(2)
        ]
        self.config = torch.tensor(
            [
                session.local,
                *session.peers,
                layer,
                rows,
                session.counter.data_ptr(),
                0,
                2000000,
                *[x.data_ptr() for x in self.raw],
                1,
            ],
            dtype=torch.int64,
            device="npu",
        )
        self.indices = torch.arange(rows * 8, dtype=torch.int32, device="npu")
        # Persistent IO lies outside the shared scratch pool: native residual
        # consumers may retain a returned tensor across the next MoE call.
        self.output = torch.empty_like(self.x)
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph, pool=session.pool):
            session.kernels.call(session.submit, self.config, self.x, self.ids)
            session.kernels.call(
                session.collect, self.config, self.x, self.ids, blocks=16
            )
            session.kernels.call(session.retire, self.config, self.x, self.ids)
            result = torch_npu.npu_moe_token_unpermute(
                self.raw[0].view(-1, 2048), self.indices, probs=self.probs
            )
            self.output.copy_(result)
        self.config[6] = 1


class Session:
    def __init__(self):
        self.api = acl_api()
        self.local = self.api.allocate_staging(ALIGN)
        zero = torch.zeros(ALIGN // 4, dtype=torch.int32, device="npu")
        self.api.copy(
            torch.npu.current_stream().npu_stream, self.local, zero.data_ptr(), ALIGN
        )
        torch.npu.synchronize()
        self.channels, self.peers, self.keys, self.exports = [], [], [], []
        directory = Path(os.environ["EXPERT_ROLE_DIRECTORY"])
        server_pids = []
        for owner in range(2):
            ch = connect(directory / f"expert{owner}.sock")
            ch.send(
                dict(
                    op="hello",
                    source=int(os.environ["EXPERT_SOURCE_ID"]),
                    pid=self.api.pid(),
                    contract=CONTRACT,
                )
            )
            response = ch.expect("window")
            assert response["contract"] == CONTRACT and response["owner"] == owner
            key = response["key"].encode()
            self.keys.append(key)
            self.peers.append(self.api.import_memory(key))
            server_pids.append(response["pid"])
            self.channels.append(ch)
        export = self.api.export(self.local, ALIGN, tuple(server_pids))
        self.exports.append(export)
        for ch in self.channels:
            ch.send(dict(op="source", key=export.decode()))
            ch.expect("registered")
        self.counter = torch.zeros(8, dtype=torch.int32, device="npu")
        self.kernels = Kernels()
        self.submit = self.kernels.load("neural_client")
        self.collect = self.kernels.load("neural_collect")
        self.retire = self.kernels.load("neural_retire")
        self.pool = torch.npu.graph_pool_handle()
        self.banks = {
            (layer, rows): Bank(self, layer, rows)
            for layer in range(2)
            for rows in range(1, 33)
        }
        torch.npu.synchronize()
        for ch in self.channels:
            ch.expect("ready")

    def forward(self, layer, hidden, logits):
        from vllm_ascend.ops.fused_moe.experts_selector import select_experts

        bank = self.banks[layer, hidden.shape[0]]
        probs, ids = select_experts(hidden, logits, 8, False, True, num_experts=128)
        bank.x.copy_(hidden)
        bank.ids.copy_(ids.to(torch.int32))
        bank.probs.copy_(probs.to(torch.bfloat16))
        bank.graph.replay()
        return bank.output

    def close(self):
        torch.npu.synchronize()
        count = int(self.counter.cpu()[0])
        eos = torch.zeros(8, dtype=torch.int32, device="npu")
        eos[0] = -(count + 1)
        self.api.copy(
            torch.npu.current_stream().npu_stream, self.local, eos.data_ptr(), 32
        )
        torch.npu.synchronize()
        for ch in self.channels:
            ch.send(dict(op="drain", generation=count))
        for ch in self.channels:
            ch.expect("drained")
        for key in self.keys:
            self.api.close_mapping(key)
        for ch in self.channels:
            ch.send(dict(op="unmapped"))
        for ch in self.channels:
            ch.expect("released")
            ch.close()
        for key in set(self.exports):
            self.api.close_mapping(key)
        self.api.free_staging(self.local)
        for bank in self.banks.values():
            bank.graph.reset()
        self.kernels.close()
        return count
