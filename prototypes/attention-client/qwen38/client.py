"""One leader per attention group; native quantization precedes input publication."""

from pathlib import Path
import os

import torch
import torch_npu
from control import connect
from wire import ALIGN, CONTRACT, Kernels, acl_api
from livemodule.arch.ascend.vllm.moe_runtime.experts_selector import select_experts


class Bank:
    def __init__(self, session, rows, quantized):
        self.input = torch.empty(
            rows, 2560, dtype=torch.int8 if quantized else torch.bfloat16, device="npu"
        )
        self.ids_storage = torch.empty(
            ((rows * 10 + 7) // 8 * 8,), dtype=torch.int32, device="npu"
        )
        self.ids = self.ids_storage[: rows * 10].view(rows, 10)
        self.probs = torch.empty(rows, 10, dtype=torch.bfloat16, device="npu")
        self.scales = torch.zeros(32, dtype=torch.float32, device="npu")
        self.raw = torch.empty(rows * 10, 2560, dtype=torch.bfloat16, device="npu")
        self.indices = torch.arange(rows * 10, dtype=torch.int32, device="npu")
        self.config = torch.tensor(
            [
                session.local,
                *session.peers,
                0 if quantized else 48,
                rows,
                session.counter.data_ptr(),
                1,
                1000000 if session.diagnostics else 20000000,
                self.raw.data_ptr(),
                0,
                0,
                0,
                0,
                0,
                self.scales.data_ptr(),
            ],
            dtype=torch.int64,
            device="npu",
        )
        assert self.config.numel() == 17


class Session:
    def __init__(self, directory, build, source=0):
        self.directory = Path(directory)
        self.diagnostics = os.environ.get("QWEN38_DIAGNOSTICS") == "1"
        self.api = acl_api()
        self.local = self.api.allocate_staging(ALIGN)
        zero = torch.zeros(ALIGN // 4, dtype=torch.int32, device="npu")
        self.api.copy(
            torch.npu.current_stream().npu_stream, self.local, zero.data_ptr(), ALIGN
        )
        torch.npu.synchronize()
        self.channels = []
        self.peers = []
        self.keys = []
        pids = []
        for owner in range(4):
            channel = connect(Path(directory) / f"expert{owner}.sock")
            channel.sock.settimeout(1200)
            channel.send(
                dict(op="hello", source=source, pid=self.api.pid(), contract=CONTRACT)
            )
            response = channel.expect("window")
            assert response["contract"] == CONTRACT
            key = response["key"].encode()
            self.keys.append(key)
            self.peers.append(self.api.import_memory(key))
            pids.append(response["pid"])
            self.channels.append(channel)
        self.export = self.api.export(self.local, ALIGN, tuple(pids))
        for channel in self.channels:
            channel.send(dict(op="source", key=self.export.decode()))
            channel.expect("registered")
        self.counter = torch.zeros(8, dtype=torch.int32, device="npu")
        self.kernels = Kernels(build)
        self.submit = self.kernels.load("neural_client")
        self.collect = self.kernels.load("neural_collect")
        self.promote = self.kernels.load("neural_promote")
        self.retire = self.kernels.load("neural_retire")
        self.banks = {
            (rows, q): Bank(self, rows, q)
            for rows in range(1, 33)
            for q in (False, True)
        }
        torch.npu.synchronize()
        for channel in self.channels:
            channel.expect("ready")

    def forward(self, layer, hidden, logits, shared, *, priority=0):
        if hidden.ndim != 2 or not 1 <= hidden.shape[0] <= 32:
            raise ValueError("First Qwen38 wire admits1..32 token rows")
        probs, ids = select_experts(hidden, logits, 10, False, True, num_experts=512)
        return self.forward_routed(layer, hidden, ids, probs, shared, priority=priority)

    def forward_routed(self, layer, hidden, ids, probs, shared, *, priority=0):
        """Same device wire with externally checked routing (also a leaf-test seam)."""
        bank = self.banks[hidden.shape[0], layer < 48]
        bank.config[5] = layer
        bank.config[15] = priority
        bank.ids.copy_(ids)
        bank.probs.copy_(probs)
        if layer < 48:
            quantized, scales = torch_npu.npu_dynamic_quant(hidden)
            bank.input.copy_(quantized)
            bank.scales[: hidden.shape[0]].copy_(scales)
        else:
            bank.input.copy_(hidden)
        if self.diagnostics:
            torch.save(
                dict(
                    layer=layer, hidden=hidden.cpu(), ids=ids.cpu(), probs=probs.cpu()
                ),
                self.directory / "last-client-input.pt",
            )
            print("submit layer", layer, "ids", ids.cpu().tolist(), flush=True)
        self.kernels.call(self.submit, bank.config, bank.input, bank.ids_storage)
        shared_result = shared(hidden)
        if priority:
            self.kernels.call(self.promote, bank.config, bank.input, bank.ids_storage)
        self.kernels.call(self.collect, bank.config, bank.input, bank.ids_storage, 16)
        if self.diagnostics:
            torch.npu.synchronize()
            value = int(self.counter.cpu()[0])
            if value < 0:
                print("failed layer", layer, "rows", hidden.shape[0], flush=True)
                for channel in self.channels:
                    channel.send(dict(op="inspect"))
                for owner, channel in enumerate(self.channels):
                    print(
                        "owner inspection",
                        owner,
                        channel.expect("inspection"),
                        flush=True,
                    )
                raise RuntimeError(
                    f"expert collect rejected layer {layer}: counter {value}"
                )
        self.kernels.call(self.retire, bank.config, bank.input, bank.ids_storage)
        routed = torch_npu.npu_moe_token_unpermute(
            bank.raw, bank.indices, probs=bank.probs
        )
        return routed + shared_result

    def close(self):
        torch.npu.synchronize()
        count = int(self.counter.cpu()[0])
        assert count >= 0
        eof = torch.zeros(8, dtype=torch.int32, device="npu")
        eof[0] = -(count + 1)
        self.api.copy(
            torch.npu.current_stream().npu_stream, self.local, eof.data_ptr(), 32
        )
        torch.npu.synchronize()
        for c in self.channels:
            c.send(dict(op="drain", generation=count))
        for c in self.channels:
            c.expect("drained")
        for key in self.keys:
            self.api.close_mapping(key)
        for c in self.channels:
            c.send(dict(op="unmapped"))
        for c in self.channels:
            c.expect("released")
            c.close()
        self.api.close_mapping(self.export)
        self.api.free_staging(self.local)
        self.kernels.close()
        return count
