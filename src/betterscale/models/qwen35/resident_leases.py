"""Resident ownership only; regular-attention pages stay in the native pool."""

from dataclasses import dataclass, field


@dataclass
class Seat:
    index: int
    epoch: int = 0
    owner: str | None = None
    tokens: tuple[int, ...] = ()  # computed prefix PLUS the MTP lookahead token
    cache_salt: str | None = None
    blocks: object | None = None
    fence: int = 0
    touched: int = 0

    @property
    def cursor(self):
        return max(len(self.tokens) - 1, 0)


@dataclass(frozen=True)
class Offer:
    seat: int
    epoch: int
    warm: bool


class ResidentLeases:
    def __init__(self, seats, *, release_blocks):
        if type(seats) is not int or seats <= 0:
            raise ValueError("resident seat capacity must be positive")
        self.seats = [Seat(i) for i in range(seats)]
        self.requests = {}
        self.clock = 0
        self.release_blocks = release_blocks

    def offer(self, tokens, cache_salt, completed_step, *, allow_hit=True):
        available = [
            s for s in self.seats if s.owner is None and s.fence <= completed_step
        ]
        hits = [
            s
            for s in available
            if allow_hit
            and s.blocks is not None
            and s.cursor > 0
            and s.cache_salt == cache_salt
            and len(tokens) >= len(s.tokens)
            and tuple(tokens[: len(s.tokens)]) == s.tokens
        ]
        if hits:
            seat = max(hits, key=lambda s: (s.cursor, -s.index))
            return Offer(seat.index, seat.epoch, True)
        if not available:
            return None
        # Prefer truly empty seats over evicting a warm resident. Then LRU.
        seat = min(available, key=lambda s: (bool(s.tokens), s.touched, s.index))
        return Offer(seat.index, seat.epoch, False)

    def discard_victim(self, offer, completed_step):
        """Release a chosen cold victim before charging the new token pages."""
        seat = self.seats[offer.seat]
        if (
            offer.warm
            or seat.owner is not None
            or seat.epoch != offer.epoch
            or seat.fence > completed_step
        ):
            raise ValueError("cannot discard this resident offer")
        if seat.blocks is not None:
            self.release_blocks(seat.blocks)
            seat.tokens, seat.cache_salt, seat.blocks = (), None, None
            seat.epoch += 1
        return Offer(seat.index, seat.epoch, False)

    def claim(self, request_id, offer, completed_step):
        if request_id in self.requests:
            raise ValueError("request already owns a resident seat")
        seat = self.seats[offer.seat]
        if (
            seat.owner is not None
            or seat.epoch != offer.epoch
            or seat.fence > completed_step
        ):
            raise ValueError("stale resident offer")
        if offer.warm and (seat.blocks is None or not seat.cursor):
            raise ValueError("warm offer lost its exact checkpoint")
        if not offer.warm:
            if seat.blocks is not None:
                self.release_blocks(seat.blocks)
            seat.epoch += 1
            seat.tokens, seat.cache_salt, seat.blocks = (), None, None
        self.clock += 1
        seat.touched, seat.owner = self.clock, request_id
        self.requests[request_id] = seat.index
        return seat

    def transferred(self, request_id):
        """Native allocation has pinned the hit; drop the resident's reference."""
        seat = self.seats[self.requests[request_id]]
        if seat.blocks is not None:
            self.release_blocks(seat.blocks)
            seat.blocks = None
        seat.tokens = ()

    def retire(self, request_id, *, fence, tokens=(), cache_salt=None, blocks=None):
        """Caller supplies a known device frontier, never a rounded/visible prefix.

        A checkpoint cannot be offered before its final writer fence. An unknown
        frontier (abort/incomplete observation) is invalidated instead of guessed.
        The caller has already pinned retained blocks before native request free.
        """
        if (blocks is None) != (not tokens) or tokens and len(tokens) < 2:
            raise ValueError("checkpoint requires computed tokens, lookahead and pages")
        seat = self.seats[self.requests[request_id]]
        if seat.blocks is not None:
            raise RuntimeError("resident reference was not transferred to its request")
        del self.requests[request_id]
        seat.owner, seat.fence = None, fence
        seat.tokens, seat.cache_salt, seat.blocks = tuple(tokens), cache_salt, blocks
        self.clock += 1
        seat.touched = self.clock
        return seat

    def invalidate_hot(self):
        for seat in self.seats:
            if seat.owner is None:
                if seat.blocks is not None:
                    self.release_blocks(seat.blocks)
                seat.tokens, seat.cache_salt, seat.blocks = (), None, None

    def evict_hot(self, completed_step, *, exclude=None):
        candidates = [
            s
            for s in self.seats
            if s.owner is None
            and s.blocks is not None
            and s.fence <= completed_step
            and s.index != exclude
        ]
        if not candidates:
            return False
        seat = min(candidates, key=lambda s: s.touched)
        self.release_blocks(seat.blocks)
        seat.tokens, seat.cache_salt, seat.blocks = (), None, None
        seat.epoch += 1
        return True


@dataclass
class Frontier:
    """CPU observation of raw accepted model output, before scheduler EOS trimming."""

    tokens: list[int]
    cursor: int = 0
    known: bool = True
    prompt_length: int = field(init=False)

    def __post_init__(self):
        self.prompt_length = len(self.tokens)

    def advance(self, query_tokens, draft_tokens, sampled):
        if not self.known:
            return
        if self.cursor < self.prompt_length:
            # Intermediate prefill sampling is discarded; actual next prompt
            # tokens supply the draft lookahead through the baseline boundary hook.
            self.cursor += query_tokens
            if self.cursor > self.prompt_length or draft_tokens:
                self.known = False
                return
            if self.cursor == self.prompt_length:
                if len(sampled) != 1:
                    self.known = False
                    return
                self.tokens.extend(sampled)
        else:
            if query_tokens - draft_tokens != 1 or not sampled:
                self.known = False
                return
            self.cursor += len(sampled)
            self.tokens.extend(sampled)
        if self.cursor >= len(self.tokens):
            self.known = False

    def checkpoint(self):
        return (
            tuple(self.tokens[: self.cursor + 1])
            if self.known and self.cursor > 0
            else ()
        )
