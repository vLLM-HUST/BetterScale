"""Completed-wave scheduling with pooled KV and whole-seat recompute preemption.

Ingress and execution have one writer. A normal finish retains a hot resident;
preemption discards its complete numerical identity, never just attention pages.
No offload or recurrent checkpoint is implied by keeping CPU request history.
"""

from collections import deque
from dataclasses import dataclass, field

import torch

from .generation import PageRequest, Progress, generation_steps


@dataclass
class Request:
    key: str
    order: int
    tokens: list[int]
    count: int
    stops: tuple[int, ...]
    output: list[int] = field(default_factory=list)
    progress: Progress = field(default_factory=Progress)
    protocol: object = None
    lease: object = None
    reply: object = None
    step: object = None
    ready_since: int = 0
    cached: int | None = None
    preemptions: int = 0


class Scheduler:
    def __init__(self, root, *, max_pending=256):
        self.root = root
        self.waiting = deque()
        self.active = {}
        self.keys = set()
        self.max_pending = max_pending
        self.sequence = 0
        self.pressure_drain = False
        self.stats = {
            "max_active": 0,
            "max_batch": 0,
            "completed": 0,
            "cancelled": 0,
            "preemptions": 0,
            "preemptions_with_output": 0,
            "waves": 0,
        }

    @property
    def busy(self):
        return bool(self.waiting or self.active)

    def submit(self, key, tokens, count, stops=()):
        if key in self.keys:
            raise ValueError("duplicate request identity")
        if len(self.keys) >= self.max_pending:
            raise ValueError("live request queue is full")
        if not tokens or type(count) is not int or count <= 0:
            raise ValueError("nonempty prompt and positive output count required")
        horizon = len(tokens) + count + 2
        table = self.root.residents_table
        # A request must be able to complete alone. This is validation, NOT a
        # reservation: physical pages remain shared and are acquired on demand.
        if (
            horizon > self.root.context_tokens
            or horizon > table.capacity.token_pages * table.capacity.page_tokens
        ):
            raise ValueError(
                "request horizon exceeds live context or token-page capacity"
            )
        self.sequence += 1
        self.keys.add(key)
        self.waiting.append(
            Request(key, self.sequence, list(tokens), count, tuple(stops))
        )

    @torch.inference_mode()
    def cancel(self, key):
        """Only between drained waves; cancellation invalidates the entire seat."""
        if key not in self.keys:
            return
        request = self.active.pop(key, None)
        if request is not None:
            request.protocol.close()
        else:
            self.waiting = deque(r for r in self.waiting if r.key != key)
        self.keys.remove(key)
        self.stats["cancelled"] += 1

    def _preempt(self, request):
        resident = self.root.residents_table._resident(request.lease)
        # During recompute the resident may not yet cover the saved output.
        # Only committed tokens beyond that boundary extend the CPU history.
        boundary = len(request.tokens) + len(request.output)
        if len(resident.tokens) > boundary:
            request.output.extend(resident.tokens[boundary:])
        request.protocol.close()  # drain + finish + evict; clears all State domains
        del self.active[request.key]
        request.protocol = request.lease = request.reply = request.step = None
        request.preemptions += 1
        self.stats["preemptions"] += 1
        self.stats["preemptions_with_output"] += bool(request.output)
        self.waiting.append(request)
        self.waiting = deque(sorted(self.waiting, key=lambda r: r.order))
        # Don't immediately readmit the victim and repeat the same eviction.
        # Let this surviving cohort drain before opening the next admission wave.
        self.pressure_drain = True

    def _advance(self, request, reply):
        try:
            request.step = request.protocol.send(reply)
            request.ready_since = self.stats["waves"]
            return None
        except StopIteration as done:
            result = done.value
            result["token_ids"] = request.output + result["token_ids"]
            result["cached_tokens"] = request.cached
            result["preemptions"] = request.preemptions
            del self.active[request.key]
            self.keys.remove(request.key)
            self.stats["completed"] += 1
            return result

    @torch.inference_mode()
    def tick(self):
        """One completed wave; results own their copies before publication."""
        with self.root._live_lock:
            self.root._require_active("schedule on")
            return self._tick()

    def _tick(self):
        table = self.root.residents_table
        if not self.active:
            self.pressure_drain = False
        while (
            self.waiting
            and not self.pressure_drain
            and len(self.active) < table.capacity.execution_seats
        ):
            request = self.waiting.popleft()
            prompt = request.tokens + request.output
            admission = table.acquire(prompt)
            request.lease = admission[0]
            if request.cached is None:
                request.cached = admission[1]
            request.protocol = generation_steps(
                self.root,
                prompt,
                request.count - len(request.output),
                eos_token_ids=request.stops,
                admission=admission,
                progress=request.progress,
            )
            self.active[request.key] = request
        self.stats["max_active"] = max(self.stats["max_active"], len(self.active))
        completed = {}
        # First consume every completed numerical reply. This makes all committed
        # boundaries visible before any request can be selected as a victim.
        for request in list(self.active.values()):
            if request.step is not None and request.reply is None:
                continue  # Its numerical step was not selected in the last wave.
            result = self._advance(request, request.reply)
            request.reply = None
            if result is not None:
                completed[request.key] = result
        for request in sorted(self.active.values(), key=lambda r: r.order):
            if request.key not in self.active:
                continue
            while isinstance(request.step, PageRequest):
                page = request.step
                while not table.can_reserve(page.lease, page.length):
                    # Newest request loses first; the oldest request can always
                    # progress alone. No active invocation remains at this point.
                    victim = max(self.active.values(), key=lambda r: r.order)
                    self._preempt(victim)
                    if victim is request:
                        break
                if request.key not in self.active:
                    break
                slots = table.reserve(page.lease, page.length)
                result = self._advance(request, slots)
                if result is not None:
                    completed[request.key] = result
                    break
        groups = {}
        for request in self.active.values():
            step = request.step
            groups.setdefault((step.kind, len(step.tokens)), []).append((request, step))
        if groups:
            # Advancing every family together locks staggered arrivals into
            # opposite target/draft phases. One family per wave lets pending
            # compatible work coalesce; oldest-ready first prevents starvation.
            group = min(
                groups.values(),
                key=lambda rows: (
                    min(r.ready_since for r, _ in rows),
                    -len(rows),
                    min(r.order for r, _ in rows),
                ),
            )
            offset = 0
            while offset < len(group):
                # Binary decomposition: no dummy GDN rows or padding State seat.
                batch = 2 ** ((len(group) - offset).bit_length() - 1)
                rows = group[offset : offset + batch]
                replies = self.root.batch_step([step for _, step in rows])
                for (request, _), reply in zip(rows, replies, strict=True):
                    request.reply = reply
                self.stats["max_batch"] = max(self.stats["max_batch"], batch)
                offset += batch
        self.stats["waves"] += 1
        return completed

    def snapshot(self):
        table = self.root.residents_table
        return {
            **self.stats,
            "active": len(self.active),
            "waiting": len(self.waiting),
            "token_pages": table.capacity.token_pages,
            "free_pages": len(table.free_pages),
        }

    def close(self):
        for key in tuple(self.keys):
            self.cancel(key)
