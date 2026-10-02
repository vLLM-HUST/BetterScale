"""Rank-local copy execution. Commands share native execution RPC ordering;
receipts use a separate local wakeup channel and never wait for another wave.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import socket

import torch

from betterscale.live.runtime.page_state import PageStateStore
from betterscale.live.runtime.host_state import (
    HostStateDomainSelection,
    HostStateKey,
    HostStateSelection,
    TorchHostStateBackend,
)


class CacheWorker:
    def __init__(self, runner, host_bytes, *, max_transfers=2, waiter_initializer=None):
        from vllm.distributed import get_tensor_model_parallel_rank

        if type(max_transfers) is not int or max_transfers < 1:
            raise ValueError("positive State transfer concurrency required")
        self.max_transfers = max_transfers
        self.runner = runner
        self.rank = get_tensor_model_parallel_rank()
        self.backend = TorchHostStateBackend(memory_budget_bytes=host_bytes)
        self.page_backend = None
        self.streams = {
            kind: torch.npu.Stream(device=runner.device) for kind in ("store", "load")
        }
        self.waiters = ThreadPoolExecutor(
            max_workers=max_transfers, thread_name_prefix="state-cache",
            initializer=waiter_initializer
        )
        self.verify = {}
        self.states = tuple(
            (name, state)
            for name, state in runner._live_state_root.named_states()
            if state is not runner._live_state_root.continuation.resident_epoch
        )
        self.inflight = set()

    def submit(self, command):
        root = self.runner._live_state_root
        kind, seat, epoch = (command[k] for k in ("kind", "seat", "epoch"))
        key = HostStateKey(command["key"], 1)
        incremental = "pages" in command
        if incremental and self.page_backend is None:
            self.page_backend = PageStateStore(
                memory_budget_bytes=self.backend.memory_budget_bytes
            )
        if kind == "drop":
            (self.page_backend if incremental else self.backend).release(key)
            self.verify.pop(key, None)
            self._send(command)
            return
        if seat in self.inflight:
            raise RuntimeError("rank already has I/O on this seat")
        self.inflight.add(seat)
        stream = self.streams[kind]
        # execute_model RPC is sequenced behind all already-submitted native
        # frames. Include the separate APC writer, not just forward's stream.
        apc = getattr(self.runner, "_mtp_apc_done", None)
        if apc is not None:
            stream.wait_event(apc)
        producer = torch.npu.Event()
        producer.record(torch.npu.current_stream(self.runner.device))
        stream.wait_event(producer)
        span = command["block_size"] // root.capacity.page_tokens
        if not span or command["block_size"] % root.capacity.page_tokens:
            raise ValueError("scheduler/kernel page ratio is not integral")
        pages = tuple(b * span + k for b in command["blocks"] for k in range(span))
        selection = HostStateSelection(
            (
                HostStateDomainSelection(root.residents, (seat,)),
                HostStateDomainSelection(root.pages, pages),
            )
        )
        was_verify = (
            seat in self.runner._live_previous_verify
            if kind == "store"
            else self.verify[key]
        )
        if kind == "load":
            # This write precedes the restore completion event on the same stream.
            with torch.npu.stream(stream):
                root.continuation.resident_epoch.tensor[seat] = epoch
        if incremental:
            objects = {
                "resident:" + command["key"]: HostStateSelection(
                    (HostStateDomainSelection(root.residents, (seat,)),)
                ),
            }
            selected = (
                range(len(command["blocks"])) if kind == "store" else command["missing"]
            )
            for i in selected:
                first = command["blocks"][i] * span
                objects[command["pages"][i]] = HostStateSelection(
                    (
                        HostStateDomainSelection(
                            root.pages, tuple(range(first, first + span))
                        ),
                    )
                )
            transfer = self.page_backend.transfer(
                key,
                self.states,
                objects,
                store=kind == "store",
                stream=stream,
            )
        else:
            method = self.backend.offload if kind == "store" else self.backend.restore
            transfer = method(self.states, key, selection, stream=stream)

        def finish(producer=producer):
            try:
                torch.npu.set_device(self.runner.device)
                transfer.result()
                command["_transfer_bytes"] = transfer.byte_length
                if hasattr(transfer,"phase_seconds"):
                    command["_transfer_phases"] = dict(transfer.phase_seconds)
                if kind == "store":
                    self.verify[key] = was_verify
                else:
                    self.runner._live_resident_epochs[seat] = epoch
                    if was_verify:
                        self.runner._live_previous_verify.add(seat)
                    else:
                        self.runner._live_previous_verify.discard(seat)
                self.inflight.remove(seat)
                self._send(command)
            except BaseException as error:
                self._send(command, repr(error))

        self.waiters.submit(finish)

    def _send(self, command, error=None):
        receipt = {key: command[key] for key in ("operation", "seat", "epoch")}
        receipt.update(rank=self.rank, error=error)
        if "_transfer_bytes" in command:
            receipt["transfer_bytes"] = command["_transfer_bytes"]
        if "_transfer_phases" in command:
            receipt["transfer_phases"] = command["_transfer_phases"]
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
            channel.settimeout(30)
            channel.sendto(json.dumps(receipt).encode(), command["endpoint"])


@torch.inference_mode()
def execute_actions(runner, commands):
    if not commands:
        return
    worker = getattr(runner, "_state_cache_worker", None)
    if worker is None:
        worker = runner._state_cache_worker = CacheWorker(
            runner, commands[0]["host_bytes"]
        )
    for command in commands:
        try:
            worker.submit(command)
        except BaseException as error:
            # Never turn an uncertain copy failure into a successful rank ack.
            worker._send(command, repr(error))
            raise
