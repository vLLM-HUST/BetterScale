"""Scheduler-owned maintenance for native resident leases and pooled FA blocks.

Explicit commands only; no automatic host cache policy. Every required rank
must retire the same operation before any residency change becomes visible.
"""

from dataclasses import dataclass, field
from concurrent.futures import Future

from .resident_leases import Offer


@dataclass(frozen=True)
class Checkpoint:
    key: str
    tokens: tuple[int, ...]
    salt: str | None
    block_count: int
    byte_length: int


@dataclass
class Pending:
    command: dict
    checkpoint: Checkpoint
    cancelled: bool = False
    ranks: set[int] = field(default_factory=set)
    completion: Future = field(default_factory=Future)


class CacheActions:
    def __init__(self, scheduler, ranks, host_bytes, *, resident_bytes, block_bytes):
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
        self.pending = {}
        self.host = {}
        self.outbox = []
        self.sequence = 0
        self.completed = []
        self.failed = None

    def _seat(self, index):
        seat = self.scheduler.residents.seats[index]
        if (
            seat.owner is not None
            or seat.io_owner is not None
            or seat.fence > self.scheduler.processed_step_seq
        ):
            raise ValueError("seat still has execution, writer-fence or I/O ownership")
        return seat

    def _queue(self, kind, checkpoint, seat=None):
        if self.failed or self.endpoint is None:
            raise RuntimeError("maintenance unavailable or failed")
        self.sequence += 1
        command = dict(
            operation=self.sequence,
            kind=kind,
            key=checkpoint.key,
            seat=seat.index if seat else None,
            epoch=seat.epoch if seat else None,
            blocks=([b.block_id for b in seat.blocks.blocks[0]] if seat else []),
            block_size=self.scheduler.block_size,
            endpoint=self.endpoint,
            host_bytes=self.host_bytes,
        )
        if seat:
            seat.io_owner = self.sequence
        self.pending[self.sequence] = Pending(command, checkpoint)
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
        size = self.resident_bytes + count * self.block_bytes
        if self.allocated_host_bytes + size > self.host_bytes:
            raise ValueError("host State capacity exhausted before dispatch")
        checkpoint = Checkpoint(key, seat.tokens, seat.cache_salt, count, size)
        return self._queue("store", checkpoint, seat)

    def load(self, key, index):
        if self.failed or self.endpoint is None:
            raise RuntimeError("maintenance unavailable or failed")
        if any(p.checkpoint.key == key for p in self.pending.values()):
            raise ValueError("checkpoint has another pending operation")
        checkpoint = self.host[key]
        seat = self._seat(index)
        manager = self.scheduler.kv_cache_manager
        # Capacity refusal changes neither victim identity nor block ownership.
        owned = len(seat.blocks.blocks[0]) if seat.blocks else 0
        if manager.block_pool.get_num_free_blocks() + owned < checkpoint.block_count:
            raise RuntimeError("not enough shared FA blocks for restore")
        self.scheduler.residents.discard_victim(
            Offer(index, seat.epoch, False), self.scheduler.processed_step_seq
        )
        blocks = manager.block_pool.get_new_blocks(checkpoint.block_count)
        seat.epoch += 1
        seat.blocks = manager.create_kv_cache_blocks((tuple(blocks),))
        return self._queue("load", checkpoint, seat)

    @property
    def allocated_host_bytes(self):
        held = dict(self.host)
        for pending in self.pending.values():
            if pending.command["kind"] in ("store", "drop"):
                held[pending.checkpoint.key] = pending.checkpoint
        return sum(checkpoint.byte_length for checkpoint in held.values())

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
            p.command["kind"] == "load"
            and not p.cancelled
            and p.checkpoint.salt == request.cache_salt
            and tuple(request.all_token_ids[: len(p.checkpoint.tokens)])
            == p.checkpoint.tokens
            for p in self.pending.values()
        )

    def receive(self, receipt):
        number, rank = receipt["operation"], receipt["rank"]
        pending = self.pending.get(number)
        if pending is None or rank not in self.ranks or rank in pending.ranks:
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
        pending.ranks.add(rank)
        if pending.ranks != self.ranks:
            return
        seat = (
            self.scheduler.residents.seats[command["seat"]]
            if command["seat"] is not None
            else None
        )
        if seat and (seat.epoch != command["epoch"] or seat.io_owner != number):
            self.failed = "resident changed while cache I/O owned it"
            raise RuntimeError(self.failed)
        kind, checkpoint = command["kind"], pending.checkpoint
        if kind == "store":
            if not pending.cancelled:
                self.host[checkpoint.key] = checkpoint
                self.scheduler._release_resident_blocks(seat.blocks)
                seat.tokens, seat.cache_salt, seat.blocks = (), None, None
                seat.epoch += 1
            seat.io_owner = None
        elif kind == "load":
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
            allocated_host_bytes=self.allocated_host_bytes,
            pending=list(self.pending),
            completed=self.completed,
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
