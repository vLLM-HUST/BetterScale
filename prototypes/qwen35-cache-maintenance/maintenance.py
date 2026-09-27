"""Experimental TP1 cache actions for the completed-wave Qwen research scheduler.

Single scheduler writer owns residency. The worker owns transfers, not policy.
This is not enabled by the native 35B Worker / AsyncScheduler adapter.
"""

from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass

from betterscale.live.runtime.host_state import (
    HostStateDomainSelection,
    HostStateKey,
    HostStateSelection,
    TorchHostStateBackend,
)


@dataclass(frozen=True)
class Checkpoint:
    key: HostStateKey
    tokens: tuple[int, ...]
    salt: str | None
    pages: int


@dataclass
class Action:
    number: int
    kind: str
    seat: int
    epoch: int
    checkpoint: Checkpoint
    future: object = None
    cancelled: bool = False


class CopyWorker:
    """Dedicated direction streams and completion wakeup independent of waves."""

    def __init__(self, root, *, host_bytes):
        import torch

        self.root = root
        self.backend = TorchHostStateBackend(memory_budget_bytes=host_bytes)
        self.streams = {
            k: torch.npu.Stream(device=root.live_device) for k in ("store", "load")
        }
        self.waiters = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="cache-copy"
        )
        # The destination keeps its new incarnation; all numerical continuation,
        # including all MTP candidates, is part of the payload.
        self.states = tuple(
            (n, s)
            for n, s in root.named_states()
            if s is not root.continuation.resident_epoch
        )

    def start(self, action, pages):
        import torch

        domains = [HostStateDomainSelection(self.root.residents, (action.seat,))]
        if pages:
            domains.append(HostStateDomainSelection(self.root.pages, tuple(pages)))
        selection = HostStateSelection(tuple(domains))
        stream = self.streams[action.kind]
        # Record AFTER final target/draft/continuation writers (or seat clear).
        # Caller must identify that writer stream, never guess forward exit.
        producer = torch.npu.Event()
        producer.record(torch.npu.current_stream(self.root.live_device))
        stream.wait_event(producer)
        try:
            transfer = (
                self.backend.offload if action.kind == "store" else self.backend.restore
            )(self.states, action.checkpoint.key, selection, stream=stream)
        except BaseException as error:
            if not self.backend.quarantined:
                raise
            failed = Future()
            failed.set_exception(error)
            return failed

        # Retain the producer event and transfer/payload until DMA finishes.
        def completed(producer=producer):
            torch.npu.set_device(self.root.live_device)
            transfer.result()
            return action.number, action.seat, action.epoch

        return self.waiters.submit(completed)

    def release(self, key):
        self.backend.release(key)

    def close(self):
        self.waiters.shutdown(wait=True)


class CacheMaintenance:
    """Scheduler-side command/receipt owner; no automatic eviction policy yet."""

    def __init__(self, table, worker):
        self.table, self.worker = table, worker
        self.sequence = 0
        self.pending = {}
        self.host = {}
        self.receipts = []
        self.closed = False

    @property
    def busy(self):
        return bool(self.pending)

    def _action(self, kind, seat, checkpoint):
        if self.closed:
            raise RuntimeError("maintenance closed")
        resident = self.table.seats[seat]
        if resident.request is not None or resident.io_owner is not None:
            raise RuntimeError("cache action needs an idle, unpinned seat")
        self.sequence += 1
        action = Action(self.sequence, kind, seat, resident.incarnation, checkpoint)
        resident.io_owner = action.number
        self.pending[action.number] = action
        return action

    def store(self, seat, key):
        if key in self.host or any(
            a.checkpoint.key == key for a in self.pending.values()
        ):
            raise ValueError("duplicate checkpoint identity")
        resident = self.table.seats[seat]
        if not resident.tokens:
            raise ValueError("cannot store empty resident")
        checkpoint = Checkpoint(
            key, tuple(resident.tokens), resident.cache_salt, len(resident.pages)
        )
        action = self._action("store", seat, checkpoint)
        try:
            action.future = self.worker.start(action, resident.pages)
        except BaseException:
            # Worker submission must have drained partial copies before raising.
            resident.io_owner = None
            del self.pending[action.number]
            raise
        return action.number

    def load(self, key, seat):
        checkpoint = self.host[key]
        if any(a.checkpoint.key == key for a in self.pending.values()):
            raise RuntimeError("checkpoint already transferring")
        if self.closed or seat not in self.table.idle_indices:
            raise RuntimeError("destination unavailable")
        resident = self.table.seats[seat]
        if len(self.table.free_pages) + len(resident.pages) < checkpoint.pages:
            raise RuntimeError("restore needs reserved shared pages")
        self.table.evict(seat)
        action = self._action("load", seat, checkpoint)
        resident.pages.extend(
            self.table.free_pages.pop() for _ in range(checkpoint.pages)
        )
        try:
            action.future = self.worker.start(action, resident.pages)
        except BaseException:
            resident.io_owner = None
            del self.pending[action.number]
            self.table.evict(seat)
            raise
        return action.number

    def blocks_prompt(self, tokens, salt):
        return any(
            a.kind == "load"
            and not a.cancelled
            and a.checkpoint.salt == salt
            and tuple(tokens[: len(a.checkpoint.tokens)]) == a.checkpoint.tokens
            for a in self.pending.values()
        )

    def cancel(self, number):
        # Cancellation is intent, never proof that DMA stopped using memory.
        self.pending[number].cancelled = True

    def reap(self):
        for number, action in list(self.pending.items()):
            if not action.future.done():
                continue
            receipt = action.future.result()  # failure leaves pins quarantined
            resident = self.table.seats[action.seat]
            if (
                receipt != (number, action.seat, action.epoch)
                or resident.incarnation != action.epoch
                or resident.io_owner != number
            ):
                raise RuntimeError("stale cache receipt; resources quarantined")
            checkpoint = action.checkpoint
            if action.kind == "store":
                if action.cancelled:
                    self.worker.release(checkpoint.key)
                    resident.io_owner = None  # original numerical resident stays
                else:
                    self.host[checkpoint.key] = checkpoint
                    resident.io_owner = None
                    self.table.evict(action.seat)
            else:
                resident.io_owner = None
                if action.cancelled:
                    self.table.evict(action.seat)
                else:
                    resident.tokens = list(checkpoint.tokens)
                    resident.cache_salt = checkpoint.salt
                    self.table.sequence += 1
                    resident.touched = self.table.sequence
            self.receipts.append((number, action.kind, action.cancelled))
            del self.pending[number]

    def wait(self, timeout=30):
        """Host event wait, usable with zero model work; scheduler reaps next."""
        if self.pending:
            done, _ = wait(
                [a.future for a in self.pending.values()],
                timeout=timeout,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                raise TimeoutError("cache completion deadline")

    def drop(self, key):
        if any(a.checkpoint.key == key for a in self.pending.values()):
            raise RuntimeError("host checkpoint is I/O pinned")
        self.worker.release(key)
        del self.host[key]

    def close(self):
        self.closed = True
        for action in self.pending.values():
            action.cancelled = True
        while self.pending:
            self.wait()
            self.reap()
        for key in list(self.host):
            self.drop(key)
        self.worker.close()
