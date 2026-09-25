"""Host admission for hot resident seats and one shared token-page capacity.

Numerical State stays in the root. A lease is an execution request, not a seat
incarnation. No later recurrent state may satisfy a shorter requested prefix.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResidentLease:
    seat: int
    incarnation: int
    request: int
    generation: object


@dataclass
class Resident:
    tokens: list[int] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    incarnation: int = 0
    request: int | None = None
    touched: int = 0


class ResidentTable:
    def __init__(self, capacity, *, clear_seat):
        if capacity.token_pages is None:
            raise ValueError(
                "resident admission currently requires explicit page capacity"
            )
        self.capacity = capacity
        self.seats = [Resident() for _ in range(capacity.resident_seats)]
        self.free_pages = list(reversed(range(capacity.token_pages)))
        self.clear_seat = clear_seat
        self.generation = object()
        self.sequence = 0

    def _resident(self, lease):
        if lease.generation is not self.generation or not 0 <= lease.seat < len(
            self.seats
        ):
            raise ValueError("stale root generation")
        resident = self.seats[lease.seat]
        if (
            resident.incarnation != lease.incarnation
            or resident.request != lease.request
        ):
            raise ValueError("stale resident/request lease")
        return resident

    def acquire(self, tokens):
        if not tokens:
            raise ValueError("empty prompt")
        if (
            sum(r.request is not None for r in self.seats)
            >= self.capacity.execution_seats
        ):
            raise RuntimeError("execution capacity exhausted")
        idle = [i for i, r in enumerate(self.seats) if r.request is None]
        matches = [
            i
            for i in idle
            if self.seats[i].tokens
            and len(self.seats[i].tokens) <= len(tokens)
            and tokens[: len(self.seats[i].tokens)] == self.seats[i].tokens
        ]
        if matches:
            index = max(matches, key=lambda i: len(self.seats[i].tokens))
        else:
            empty = [i for i in idle if not self.seats[i].tokens]
            index = (
                empty[0] if empty else min(idle, key=lambda i: self.seats[i].touched)
            )
            self.evict(index)
        resident = self.seats[index]
        self.sequence += 1
        resident.request = self.sequence
        resident.touched = self.sequence
        return ResidentLease(
            index, resident.incarnation, resident.request, self.generation
        ), len(resident.tokens)

    def evict(self, index):
        resident = self.seats[index]
        if resident.request is not None:
            raise RuntimeError("cannot evict a live request")
        # Invalidate identity before numerical writes; a failed clear cannot hit.
        resident.tokens.clear()
        resident.incarnation += 1
        self.clear_seat(index)
        self.free_pages.extend(reversed(resident.pages))
        resident.pages.clear()

    def reserve(self, lease, length):
        resident = self._resident(lease)
        if length < len(resident.tokens):
            raise ValueError("reservation cannot rewind committed State")
        pages = (length + self.capacity.page_tokens - 1) // self.capacity.page_tokens
        needed = max(pages - len(resident.pages), 0)
        if needed > len(self.free_pages):
            victims = sorted(
                (i for i, r in enumerate(self.seats) if r.request is None and r.pages),
                key=lambda i: self.seats[i].touched,
            )
            if needed > len(self.free_pages) + sum(
                len(self.seats[i].pages) for i in victims
            ):
                raise RuntimeError("shared token-page capacity exhausted")
            for victim in victims:
                self.evict(victim)
                if needed <= len(self.free_pages):
                    break
        resident.pages.extend(self.free_pages.pop() for _ in range(needed))
        return [
            resident.pages[i // self.capacity.page_tokens] * self.capacity.page_tokens
            + i % self.capacity.page_tokens
            for i in range(length)
        ]

    def commit(self, lease, tokens):
        resident = self._resident(lease)
        if (
            len(resident.tokens) + len(tokens)
            > len(resident.pages) * self.capacity.page_tokens
        ):
            raise ValueError("commit exceeds reserved token pages")
        resident.tokens.extend(tokens)

    def finish(self, lease):
        resident = self._resident(lease)
        keep = (
            len(resident.tokens) + self.capacity.page_tokens - 1
        ) // self.capacity.page_tokens
        self.free_pages.extend(reversed(resident.pages[keep:]))
        del resident.pages[keep:]
        resident.request = None
        # Keep tokens/incarnation/numerical State: request finish is not eviction.
