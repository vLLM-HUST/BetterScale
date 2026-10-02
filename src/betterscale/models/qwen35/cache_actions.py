"""Scheduler-owned maintenance for native resident leases and pooled FA blocks.

Execution mechanism shared by explicit commands and automatic policy. Every rank
must retire the same operation before any residency change becomes visible.
"""

from dataclasses import dataclass, field
from concurrent.futures import Future
from collections import deque

from .resident_leases import Offer
from .cache_pages import PageResidency, page_keys


class CacheBusy(ValueError):
    """Recoverable admission refusal; no device or host ownership changed."""


@dataclass(frozen=True)
class Checkpoint:
    key: str
    tokens: tuple[int, ...]
    salt: str | None
    block_count: int
    byte_length: int
    pages: tuple[str, ...] = ()


@dataclass
class Pending:
    command: dict
    checkpoint: Checkpoint
    pinned_blocks: object | None = None
    cancelled: bool = False
    ranks: set[int] = field(default_factory=set)
    staged_ranks: set[int] = field(default_factory=set)
    device_released: bool = False
    transfer_bytes: dict[int, int] = field(default_factory=dict)
    transfer_phases: dict[int, dict] = field(default_factory=dict)
    completion: Future = field(default_factory=Future)


class CacheActions:
    def __init__(
        self,
        scheduler,
        ranks,
        host_bytes,
        *,
        resident_bytes,
        block_bytes,
        incremental=False,
        max_pending=None,
        two_phase_store=False,
    ):
        if ranks not in (1, 2) or host_bytes <= 0:
            raise ValueError(
                "cache maintenance requires TP1/TP2 and positive host budget"
            )
        self.scheduler = scheduler
        self.ranks = set(range(ranks))
        if resident_bytes <= 0 or block_bytes <= 0:
            raise ValueError("host admission needs declared State byte geometry")
        self.resident_bytes, self.block_bytes = resident_bytes, block_bytes
        self.host_bytes = host_bytes
        self.endpoint = None
        if max_pending is None and incremental:
            max_pending = 1
        if max_pending is not None and (
            type(max_pending) is not int or not 1 <= max_pending <= len(scheduler.residents.seats)
        ):
            raise ValueError("cache transaction capacity must fit resident seats")
        self.max_pending = max_pending
        self.two_phase_store = two_phase_store
        self.pending = {}
        self.peak_pending = 0
        self.host = {}
        self.outbox = []
        self.sequence = 0
        self.completed = deque(maxlen=256)
        self.failed = None
        self.pages = (
            PageResidency(scheduler.kv_cache_manager.block_pool)
            if incremental
            else None
        )

    def _admit(self, checkpoint=None):
        if self.max_pending is not None and len(self.pending) >= self.max_pending:
            raise CacheBusy("cache transaction capacity is still in flight")
        if checkpoint is None:
            return
        pages = set(checkpoint.pages)
        for pending in self.pending.values():
            other = pending.checkpoint
            if checkpoint.key == other.key or (pages and pages.intersection(other.pages)):
                # Keep a per-object fence through TP quorum. Independent
                # sessions proceed; a shared producer, reader or drop waits.
                raise CacheBusy("shared State object has an operation in flight")

    def _seat(self, index):
        self._admit()
        seat = self.scheduler.residents.seats[index]
        if (
            seat.owner is not None
            or seat.io_owner is not None
            or seat.fence > self.scheduler.processed_step_seq
        ):
            raise ValueError("seat still has execution, writer-fence or I/O ownership")
        return seat

    def _queue(self, kind, checkpoint, seat=None, *, blocks=None, retain=False):
        if self.failed or self.endpoint is None:
            raise RuntimeError("maintenance unavailable or failed")
        self._admit(checkpoint)
        self.sequence += 1
        command = dict(
            operation=self.sequence,
            kind=kind,
            key=checkpoint.key,
            seat=seat.index if seat else None,
            epoch=seat.epoch if seat else None,
            blocks=(
                [b.block_id for b in (blocks or seat.blocks).blocks[0]] if seat else []
            ),
            retain=retain,
            block_size=self.scheduler.block_size,
            endpoint=self.endpoint,
            host_bytes=self.host_bytes,
        )
        if kind == "store" and self.two_phase_store:
            command["two_phase_store"] = True
        if checkpoint.pages:
            command["pages"] = list(checkpoint.pages)
        if seat:
            seat.io_owner = self.sequence
        self.pending[self.sequence] = Pending(command, checkpoint)
        self.peak_pending = max(self.peak_pending, len(self.pending))
        self.outbox.append(command)
        return self.sequence

    def store(self, index, key):
        if (
            not key
            or key in self.host
            or any(p.checkpoint.key == key for p in self.pending.values())
        ):
            raise ValueError("checkpoint key must be new and nonempty")
        seat = self._seat(index)
        if not seat.tokens or seat.blocks is None:
            raise ValueError("store requires a published exact resident frontier")
        count = len(seat.blocks.blocks[0])
        checkpoint = self._checkpoint(key, seat.tokens, seat.cache_salt, count)
        if self.pages is not None:
            self.pages.remember(checkpoint.pages, seat.blocks.blocks[0])
        return self._queue("store", checkpoint, seat)

    def backup(self, index, key, tokens, salt, blocks):
        """Keep device residency; caller supplies a drained exact frontier."""
        seat = self.scheduler.residents.seats[index]
        if seat.io_owner is not None or seat.fence > self.scheduler.processed_step_seq:
            raise ValueError("backup seat has outstanding ownership")
        if seat.owner is not None:
            request = self.scheduler.requests[seat.owner]
            if request.last_sched_seq > self.scheduler.processed_step_seq:
                raise ValueError("backup writer has not retired")
        if (
            not tokens
            or blocks is None
            or key in self.host
            or any(p.checkpoint.key == key for p in self.pending.values())
        ):
            raise ValueError("backup requires a new key and exact frontier")
        count = len(blocks.blocks[0])
        checkpoint = self._checkpoint(key, tokens, salt, count)
        if self.pages is not None:
            self.pages.remember(checkpoint.pages, blocks.blocks[0])
        number = self._queue("store", checkpoint, seat, blocks=blocks, retain=True)
        for group in blocks.blocks:
            self.scheduler.kv_cache_manager.block_pool.touch(group)
        self.pending[number].pinned_blocks = blocks
        return number

    def touch(self, key):
        # dict insertion order is the host LRU; pending transfers pin entries.
        self.host[key] = self.host.pop(key)

    def load(self, key, index):
        if self.failed or self.endpoint is None:
            raise RuntimeError("maintenance unavailable or failed")
        if any(p.checkpoint.key == key for p in self.pending.values()):
            raise ValueError("checkpoint has another pending operation")
        checkpoint = self.host[key]
        self._admit(checkpoint)
        seat = self._seat(index)
        manager = self.scheduler.kv_cache_manager
        # Capacity refusal changes neither victim identity nor block ownership.
        owned = len(seat.blocks.blocks[0]) if seat.blocks else 0
        if manager.block_pool.get_num_free_blocks() + owned < checkpoint.block_count:
            raise RuntimeError("not enough shared FA blocks for restore")
        self.scheduler.residents.discard_victim(
            Offer(index, seat.epoch, False), self.scheduler.processed_step_seq
        )
        hits = self.pages.acquire(checkpoint.pages) if self.pages is not None else {}
        missing = [i for i in range(checkpoint.block_count) if i not in hits]
        fresh = iter(manager.block_pool.get_new_blocks(len(missing)))
        blocks = tuple(
            hits[i] if i in hits else next(fresh) for i in range(checkpoint.block_count)
        )
        seat.epoch += 1
        seat.blocks = manager.create_kv_cache_blocks((blocks,))
        self.touch(key)
        number = self._queue("load", checkpoint, seat)
        if checkpoint.pages:
            self.pending[number].command["missing"] = missing
        return number

    def backup_size(self, tokens, salt, count):
        if self.pages is None:
            return self.resident_bytes + count * self.block_bytes
        keys = page_keys(
            tokens, salt, self.scheduler.block_size, count, f"auto:{self.sequence + 1}"
        )
        held = {k for cp in self.host.values() for k in cp.pages}
        held.update(k for p in self.pending.values() for k in p.checkpoint.pages)
        return self.resident_bytes + len(set(keys) - held) * self.block_bytes

    def _checkpoint(self, key, tokens, salt, count):
        self._admit()
        if (
            count
            != (len(tokens) - 2 + self.scheduler.block_size)
            // self.scheduler.block_size
        ):
            raise ValueError("checkpoint pages do not cover its exact frontier")
        pages = (
            page_keys(tokens, salt, self.scheduler.block_size, count, key)
            if self.pages is not None
            else ()
        )
        checkpoint = Checkpoint(
            key,
            tuple(tokens),
            salt,
            count,
            self.resident_bytes + count * self.block_bytes,
            pages,
        )
        self._admit(checkpoint)
        held = {cp.key: cp for cp in self.host.values()}
        held.update((p.checkpoint.key, p.checkpoint) for p in self.pending.values())
        held[key] = checkpoint
        if self._host_bytes(held.values()) > self.host_bytes:
            raise ValueError("host State capacity exhausted before dispatch")
        return checkpoint

    def _host_bytes(self, checkpoints):
        checkpoints = list(checkpoints)
        if self.pages is None:
            return sum(cp.byte_length for cp in checkpoints)
        return (
            len(checkpoints) * self.resident_bytes
            + len({key for cp in checkpoints for key in cp.pages}) * self.block_bytes
        )

    @property
    def allocated_host_bytes(self):
        held = dict(self.host)
        for pending in self.pending.values():
            if pending.command["kind"] in ("store", "drop"):
                held[pending.checkpoint.key] = pending.checkpoint
        return self._host_bytes(held.values())

    def cancel(self, operation):
        if self.pending[operation].command["kind"] == "drop":
            raise ValueError("host drop is irreversible once dispatched")
        self.pending[operation].cancelled = True

    def drop(self, key):
        if any(p.checkpoint.key == key for p in self.pending.values()):
            raise ValueError("host checkpoint is pinned")
        return self._queue("drop", self.host[key])

    def blocks_prompt(self, request):
        return not request.skip_reading_prefix_cache and any(
            p.command["kind"] in ("load", "store")
            and not p.cancelled
            and p.checkpoint.salt == request.cache_salt
            and tuple(request.all_token_ids[: len(p.checkpoint.tokens)])
            == p.checkpoint.tokens
            for p in self.pending.values()
        )

    def _release_store_device(self, pending):
        """Only a complete local DRAM snapshot quorum retires device ownership.

        The caller still owns the DRAM leases until replication completion.
        Host publication is separate and must not happen here.
        """
        command = pending.command
        seat = self.scheduler.residents.seats[command["seat"]]
        if seat.epoch != command["epoch"] or seat.io_owner != command["operation"]:
            self.failed = "resident changed while cache I/O owned it"
            raise RuntimeError(self.failed)
        if not pending.cancelled and not command["retain"]:
            self.scheduler._release_resident_blocks(seat.blocks)
            seat.tokens, seat.cache_salt, seat.blocks = (), None, None
            seat.epoch += 1
        if pending.pinned_blocks is not None:
            self.scheduler._release_resident_blocks(pending.pinned_blocks)
            pending.pinned_blocks = None
        seat.io_owner = None
        pending.device_released = True

    def receive(self, receipt):
        number, rank = receipt["operation"], receipt["rank"]
        pending = self.pending.get(number)
        phase = receipt.get("phase", "committed")
        seen = pending.staged_ranks if pending and phase == "staged" else (pending.ranks if pending else set())
        if pending is None or rank not in self.ranks or rank in seen:
            self.failed = "unknown, duplicate or foreign-rank cache completion"
            raise RuntimeError(self.failed)
        command = pending.command
        if (
            receipt.get("error")
            or receipt.get("seat") != command["seat"]
            or receipt.get("epoch") != command["epoch"]
        ):
            self.failed = receipt.get("error") or "stale cache completion"
            raise RuntimeError(self.failed)  # no release on unknown DMA lifetime
        two_phase = command.get("two_phase_store", False)
        if phase not in ("staged", "committed") or (phase == "staged" and not two_phase):
            self.failed = "unexpected cache completion phase"
            raise RuntimeError(self.failed)
        if phase == "staged":
            pending.staged_ranks.add(rank)
            if pending.staged_ranks == self.ranks:
                self._release_store_device(pending)
            return
        if two_phase and rank not in pending.staged_ranks:
            self.failed = "replica completion precedes local staging"
            raise RuntimeError(self.failed)
        pending.ranks.add(rank)
        if "transfer_bytes" in receipt:
            pending.transfer_bytes[rank] = receipt["transfer_bytes"]
        if "transfer_phases" in receipt:
            pending.transfer_phases[rank] = receipt["transfer_phases"]
        if pending.ranks != self.ranks:
            return
        seat = (
            self.scheduler.residents.seats[command["seat"]]
            if command["seat"] is not None
            else None
        )
        if seat and not pending.device_released and (seat.epoch != command["epoch"] or seat.io_owner != number):
            self.failed = "resident changed while cache I/O owned it"
            raise RuntimeError(self.failed)
        kind, checkpoint = command["kind"], pending.checkpoint
        if kind == "store":
            if not pending.cancelled:
                self.host[checkpoint.key] = checkpoint
            if not pending.device_released:
                self._release_store_device(pending)
        elif kind == "load":
            if self.pages is not None:
                self.pages.remember(checkpoint.pages, seat.blocks.blocks[0])
            if pending.cancelled:
                self.scheduler._release_resident_blocks(seat.blocks)
                seat.tokens, seat.cache_salt, seat.blocks = (), None, None
                seat.epoch += 1
            else:
                seat.tokens, seat.cache_salt = checkpoint.tokens, checkpoint.salt
                self.scheduler.residents.clock += 1
                seat.touched = self.scheduler.residents.clock
            seat.io_owner = None
        elif kind == "drop":
            self.host.pop(checkpoint.key, None)
        self.completed.append(
            dict(
                operation=number,
                kind=kind,
                cancelled=pending.cancelled,
                ranks=sorted(pending.ranks),
                transfer_bytes_per_rank=dict(pending.transfer_bytes),
                transfer_phases_per_rank=dict(pending.transfer_phases),
                restored_pages=len(command.get("missing", command["blocks"]))
                if kind == "load"
                else 0,
            )
        )
        pending.completion.set_result(self.completed[-1])
        del self.pending[number]
        if kind == "store" and pending.cancelled:
            self._queue("drop", checkpoint)

    def result(self, number):
        if number in self.pending:
            return self.pending[number].completion
        return next(r for r in self.completed if r["operation"] == number)

    def take_commands(self):
        commands, self.outbox = self.outbox, []
        return commands

    def snapshot(self):
        return dict(
            host=list(self.host),
            host_entries=[
                dict(
                    key=cp.key,
                    cache_salt=cp.salt,
                    cursor=len(cp.tokens) - 1,
                    byte_length=cp.byte_length,
                )
                for cp in self.host.values()
            ],
            allocated_host_bytes=self.allocated_host_bytes,
            pending=list(self.pending),
            peak_pending=self.peak_pending,
            max_pending=self.max_pending,
            completed=list(self.completed),
            seats=[
                dict(
                    seat=s.index,
                    epoch=s.epoch,
                    cursor=s.cursor,
                    cache_salt=s.cache_salt,
                    owner=s.owner,
                    io_owner=s.io_owner,
                    blocks=[b.block_id for b in s.blocks.blocks[0]] if s.blocks else [],
                )
                for s in self.scheduler.residents.seats
            ],
        )
