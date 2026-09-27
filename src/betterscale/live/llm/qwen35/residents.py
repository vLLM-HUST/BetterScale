"""Host admission for hot resident seats and one shared token-page capacity.

Numerical State stays in the root. A lease is an execution request, not a seat
incarnation. No later recurrent state may satisfy a shorter requested prefix.
"""

from dataclasses import dataclass, field
from collections.abc import Sequence


@dataclass(frozen=True, eq=False)
class TokenSlots(Sequence):
    """Immutable page-map snapshot, without expanding an entire token prefix.

    Prefix views retain only page IDs. Short suffixes materialize only the new
    writes; a 256K decode wave must not rebuild 256K Python integers per row.
    """

    pages: tuple[int, ...]
    page_tokens: int
    length: int

    def __post_init__(self):
        if not 0 <= self.length <= len(self.pages) * self.page_tokens:
            raise ValueError("token view exceeds page map")
        if len(set(self.pages)) != len(self.pages) or any(p < 0 for p in self.pages):
            raise ValueError("token view requires unique nonnegative physical pages")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, stride = index.indices(self.length)
            if start == 0 and stride == 1:
                return TokenSlots(self.pages, self.page_tokens, stop)
            return [self[i] for i in range(start, stop, stride)]
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)
        return (
            self.pages[index // self.page_tokens] * self.page_tokens
            + index % self.page_tokens
        )

    def __eq__(self, other):
        return (
            isinstance(other, Sequence)
            and len(self) == len(other)
            and all(a == b for a, b in zip(self, other, strict=True))
        )

    def block_row(self):
        return list(
            self.pages[: (self.length + self.page_tokens - 1) // self.page_tokens]
        )


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
    io_owner: int | None = None
    cache_salt: str | None = None


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

    @property
    def idle_indices(self):
        return [
            i
            for i, r in enumerate(self.seats)
            if r.request is None and r.io_owner is None
        ]

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

    def can_reserve(self, lease, length):
        """Check shared pages, including reclaimable idle residents, without writes."""
        resident = self._resident(lease)
        pages = (length + self.capacity.page_tokens - 1) // self.capacity.page_tokens
        available = (
            len(resident.pages)
            + len(self.free_pages)
            + sum(len(self.seats[i].pages) for i in self.idle_indices)
        )
        return pages <= available

    def acquire(self, tokens, cache_salt=None):
        if not tokens:
            raise ValueError("empty prompt")
        if (
            sum(r.request is not None for r in self.seats)
            >= self.capacity.execution_seats
        ):
            raise RuntimeError("execution capacity exhausted")
        idle = self.idle_indices
        if not idle:
            raise RuntimeError("resident seats temporarily unavailable")
        matches = [
            i
            for i in idle
            if self.seats[i].tokens
            and self.seats[i].cache_salt == cache_salt
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
        resident.cache_salt = cache_salt
        self.sequence += 1
        resident.request = self.sequence
        resident.touched = self.sequence
        return ResidentLease(
            index, resident.incarnation, resident.request, self.generation
        ), len(resident.tokens)

    def evict(self, index):
        resident = self.seats[index]
        if resident.request is not None or resident.io_owner is not None:
            raise RuntimeError("cannot evict a live request or I/O-owned seat")
        # Invalidate identity before numerical writes; a failed clear cannot hit.
        resident.tokens.clear()
        resident.cache_salt = None
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
                (i for i in self.idle_indices if self.seats[i].pages),
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
        return TokenSlots(tuple(resident.pages), self.capacity.page_tokens, length)

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
