"""One leader per attention group; native quantization precedes input publication."""

from pathlib import Path
import os

import torch
import torch_npu
from control import connect
from wire import ALIGN, CONTRACT, Kernels, acl_api
from livemodule.arch.ascend.vllm.moe_runtime.experts_selector import select_experts


class Bank:
    def __init__(self, session, rows, quantized, fused=False, online=False):
        self.input = torch.empty(
            rows, 2560, dtype=torch.int8 if quantized else torch.bfloat16, device="npu"
        )
        self.ids_storage = torch.empty(
            ((rows * 10 + 7) // 8 * 8 + 8,), dtype=torch.int32, device="npu"
        )
        self.ids = self.ids_storage[: rows * 10].view(rows, 10)
        self.probs_storage = torch.empty(
            (rows * 10 + 15) // 16 * 16 + 16, dtype=torch.bfloat16, device="npu"
        )
        self.probs = self.probs_storage[: rows * 10].view(rows, 10)
        self.scales = torch.zeros(
            (rows + 7) // 8 * 8, dtype=torch.float32, device="npu"
        )
        self.raw = (
            None
            if fused
            else torch.empty(rows * 10, 2560, dtype=torch.bfloat16, device="npu")
        )
        self.reduced = (
            torch.empty(rows, 2560, dtype=torch.bfloat16, device="npu")
            if fused
            else None
        )
        self.indices = torch.arange(rows * 10, dtype=torch.int32, device="npu")
        self.timings = (
            torch.zeros(48, 8, dtype=torch.int64, device="npu") if online else None
        )
        self.config = torch.tensor(
            [
                session.local,
                *(session.peers + [0] * (4 - len(session.peers))),
                0 if quantized else 48,
                rows,
                session.counter.data_ptr(),
                1,
                1000000 if session.diagnostics else 20000000,
                self.raw.data_ptr() if self.raw is not None else 0,
                self.probs.data_ptr(),
                self.reduced.data_ptr() if self.reduced is not None else 0,
                self.timings.data_ptr() if self.timings is not None else 0,
                250 if online else 0,
                0,
                self.scales.data_ptr(),
            ],
            dtype=torch.int64,
            device="npu",
        )
        assert self.config.numel() == 17
        # Events belong to warmed banks; no event/stream construction in capture.
        self.shared_input_ready = torch.npu.Event()
        self.shared_done = torch.npu.Event()


class Session:
    def __init__(self, directory, build, source=0):
        self.directory = Path(directory)
        self.diagnostics = os.environ.get("QWEN38_DIAGNOSTICS") == "1"
        self.shared_overlap = os.environ.get("QWEN38_SHARED_OVERLAP") == "1"
        self.shared_stream = torch.npu.Stream() if self.shared_overlap else None
        self.kernels = Kernels(build)
        self.parallel_pack = os.environ.get("QWEN38_PARALLEL_PACK") == "1"
        if self.parallel_pack and not self.kernels.parallel_client_pack:
            raise RuntimeError("Parallel pack requires matching client binary exports")
        self.pack = (
            self.kernels.load("neural_pack")
            if self.kernels.parallel_client_pack
            else None
        )
        self.publish = (
            self.kernels.load("neural_publish")
            if self.kernels.parallel_client_pack
            else None
        )
        self.collect_blocks = int(os.environ.get("QWEN38_COLLECT_BLOCKS", "16"))
        if self.collect_blocks not in (16, 32, 48):
            raise ValueError("collect blocks must be16,32,48")
        self.fused_collect = os.environ.get("QWEN38_FUSED_COLLECT") == "1"
        if self.fused_collect and not self.kernels.fused_client_collect:
            raise RuntimeError("Fused collect requires matching client binary exports")
        pipelined = os.environ.get("QWEN38_PIPELINED_COLLECT") == "1"
        if pipelined and (
            not self.fused_collect or not self.kernels.pipelined_client_collect
        ):
            raise RuntimeError(
                "Pipelined collect requires fused mode and a matching binary"
            )
        self.collect_fused = (
            self.kernels.load(
                "neural_collect_pipelined" if pipelined else "neural_collect_fused"
            )
            if self.kernels.fused_client_collect
            else None
        )
        self.online_collect = os.environ.get("QWEN38_ONLINE_COLLECT") == "1"
        if self.online_collect and not self.kernels.layout.route_ready:
            raise RuntimeError("Online collect requires route-ready producer ABI")
        self.collect_online = (
            self.kernels.load("neural_collect_online")
            if self.kernels.layout.route_ready
            else None
        )
        self.layout = self.kernels.layout
        contract = self.layout.contract()
        self.api = acl_api()
        self.local = self.api.allocate_staging(self.layout.source_bytes)
        zero = torch.zeros(
            self.layout.source_bytes // 4, dtype=torch.int32, device="npu"
        )
        self.api.copy(
            torch.npu.current_stream().npu_stream,
            self.local,
            zero.data_ptr(),
            self.layout.source_bytes,
        )
        torch.npu.synchronize()
        self.channels = []
        self.peers = []
        self.keys = []
        pids = []
        for owner in range(self.layout.owners):
            channel = connect(Path(directory) / f"expert{owner}.sock")
            channel.sock.settimeout(1200)
            channel.send(
                dict(op="hello", source=source, pid=self.api.pid(), contract=contract)
            )
            response = channel.expect("window")
            assert response["contract"] == contract
            key = response["key"].encode()
            self.keys.append(key)
            self.peers.append(self.api.import_memory(key))
            pids.append(response["pid"])
            self.channels.append(channel)
        self.export = self.api.export(self.local, self.layout.source_bytes, tuple(pids))
        for channel in self.channels:
            channel.send(dict(op="source", key=self.export.decode()))
            channel.expect("registered")
        self.counter = torch.zeros(8, dtype=torch.int32, device="npu")
        self.submit = self.kernels.load("neural_client")
        self.collect = self.kernels.load("neural_collect")
        self.promote = self.kernels.load("neural_promote")
        self.retire = self.kernels.load("neural_retire")
        # Banks are created by eager warmup for the actual capture buckets,
        # not all1..capacity sizes (which would reserve quadratic memory).
        self.banks = {}
        torch.npu.synchronize()
        for channel in self.channels:
            channel.expect("ready")

    def forward(self, layer, hidden, logits, shared, *, priority=0):
        if hidden.ndim != 2 or not 1 <= hidden.shape[0] <= self.layout.rows:
            raise ValueError(f"rows exceed channel capacity {self.layout.rows}")
        probs, ids = select_experts(hidden, logits, 10, False, True, num_experts=512)
        return self.forward_routed(layer, hidden, ids, probs, shared, priority=priority)

    def forward_routed(self, layer, hidden, ids, probs, shared, *, priority=0):
        """Same device wire with externally checked routing (also a leaf-test seam)."""
        if self.online_collect and not self.fused_collect:
            raise RuntimeError("Online collect requires fused output bank")
        key = (hidden.shape[0], layer < 48) + (
            (True, True)
            if self.online_collect
            else (True,) if self.fused_collect else ()
        )
        if not 1 <= key[0] <= self.layout.rows:
            raise ValueError(f"rows exceed channel capacity {self.layout.rows}")
        if key not in self.banks:
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("Warm this expert row bucket before capture")
            self.banks[key] = Bank(self, *key)
        bank = self.banks[key]
        main_stream = torch.npu.current_stream()
        if self.shared_overlap:
            # Fork BEFORE client quantization/packing. The read-only hidden input
            # is ready here; shared does not consume routing or published buffers.
            bank.shared_input_ready.record(main_stream)
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
        if self.parallel_pack:
            self.kernels.call(self.pack, bank.config, bank.input, bank.ids_storage, 16)
            self.kernels.call(self.publish, bank.config, bank.input, bank.ids_storage)
        else:
            self.kernels.call(self.submit, bank.config, bank.input, bank.ids_storage)
        if self.shared_overlap:
            with torch.npu.stream(self.shared_stream):
                self.shared_stream.wait_event(bank.shared_input_ready)
                shared_result = shared(hidden)
                bank.shared_done.record(self.shared_stream)
            # Submit precedes this join on main. Promotion therefore observes
            # BOTH this generation's READY and shared completion, never old READY.
            # Keep collect after the join in this first experiment: its busy-poll
            # AIV must not confound the isolated submit/shared overlap comparison.
            main_stream.wait_event(bank.shared_done)
            shared_result.record_stream(main_stream)
        else:
            shared_result = shared(hidden)
        if priority:
            self.kernels.call(self.promote, bank.config, bank.input, bank.ids_storage)
        self.kernels.call(
            (
                self.collect_online
                if self.online_collect
                else self.collect_fused if self.fused_collect else self.collect
            ),
            bank.config,
            bank.input,
            bank.ids_storage,
            self.collect_blocks if self.fused_collect else 16,
        )
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
        routed = (
            bank.reduced
            if self.fused_collect
            else torch_npu.npu_moe_token_unpermute(
                bank.raw, bank.indices, probs=bank.probs
            )
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
