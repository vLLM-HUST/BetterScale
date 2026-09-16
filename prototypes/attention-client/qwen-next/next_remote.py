"""Split submit/collect banks leave shared MLP between device graph submissions."""

import os
from pathlib import Path
import torch
import torch_npu
from common import acl_api, ALIGN
from control import connect
from settings import CONTRACT, LAYERS
from device_joint import Kernels


class Bank:
    def __init__(self, session, rows):
        self.x = torch.zeros((rows, 2048), dtype=torch.bfloat16, device="npu")
        self.id_storage = torch.zeros(
            ((rows * 10 + 7) // 8 * 8,), dtype=torch.int32, device="npu"
        )
        self.ids = self.id_storage[: rows * 10].view(rows, 10)
        self.probs = torch.zeros((rows, 10), dtype=torch.bfloat16, device="npu")
        self.raw = torch.empty((rows * 10, 2048), dtype=torch.bfloat16, device="npu")
        self.output = torch.empty_like(self.x)
        self.config = torch.tensor(
            [
                session.local,
                *session.peers,
                0,
                rows,
                session.counter.data_ptr(),
                0,
                20000000,
                self.raw.data_ptr(),
            ],
            dtype=torch.int64,
            device="npu",
        )
        self.indices = torch.arange(rows * 10, dtype=torch.int32, device="npu")
        self.submit_graph, self.collect_graph = (
            torch.npu.NPUGraph(),
            torch.npu.NPUGraph(),
        )
        with torch.npu.graph(self.submit_graph, pool=session.pool):
            session.kernels.call(session.submit, self.config, self.x, self.id_storage)
        with torch.npu.graph(self.collect_graph, pool=session.pool):
            session.kernels.call(
                session.collect, self.config, self.x, self.id_storage, blocks=16
            )
            session.kernels.call(session.retire, self.config, self.x, self.id_storage)
            self.output.copy_(
                torch_npu.npu_moe_token_unpermute(
                    self.raw, self.indices, probs=self.probs
                )
            )
        self.config[8] = 1


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
        pids = []
        for owner in range(4):
            ch = connect(directory / f"expert{owner}.sock")
            ch.send(
                dict(
                    op="hello",
                    source=int(os.environ["EXPERT_SOURCE_ID"]),
                    pid=self.api.pid(),
                    contract=CONTRACT,
                )
            )
            reply = ch.expect("window")
            assert reply["contract"] == CONTRACT and reply["owner"] == owner
            key = reply["key"].encode()
            self.keys.append(key)
            self.peers.append(self.api.import_memory(key))
            pids.append(reply["pid"])
            self.channels.append(ch)
        export = self.api.export(self.local, ALIGN, tuple(pids))
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
        self.banks = {rows: Bank(self, rows) for rows in range(1, 33)}
        torch.npu.synchronize()
        for ch in self.channels:
            ch.expect("ready")

    def forward(self, layer, hidden, logits, shared):
        from vllm_ascend.ops.fused_moe.experts_selector import select_experts

        bank = self.banks[hidden.shape[0]]
        probs, ids = select_experts(hidden, logits, 10, False, True, num_experts=512)
        bank.config[5] = layer
        bank.x.copy_(hidden)
        bank.ids.copy_(ids)
        bank.probs.copy_(probs)
        bank.submit_graph.replay()
        # No device polling kernel is queued ahead of local useful work.
        shared_output = shared(hidden)
        bank.collect_graph.replay()
        # Return independent storage: bank output is reused by the next layer.
        return bank.output + shared_output

    def close(self):
        torch.npu.synchronize()
        count = int(self.counter.cpu()[0])
        assert count >= 0
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
        for key in self.exports:
            self.api.close_mapping(key)
        self.api.free_staging(self.local)
        for bank in self.banks.values():
            bank.submit_graph.reset()
            bank.collect_graph.reset()
        self.kernels.close()
        return count
