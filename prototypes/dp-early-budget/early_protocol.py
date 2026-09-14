"""CPU-only, immutable early-wave budgets; no acceptance prediction."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Proposal:
    sequence: int
    tokens: int
    mode: int
    eligible: bool


@dataclass(frozen=True)
class Budget:
    sequence: int
    tokens: tuple[int, ...]
    mode: int
    admitted: bool


def propose(
    sequence, schedule, previous_ids, capacities, *, query_tokens=6, max_requests=2
):
    """Use only an already-issued SchedulerOutput, never next-token feedback."""
    if schedule is None:
        return Proposal(sequence, 0, 0, False)
    counts = schedule.num_scheduled_tokens
    ids = tuple(counts)
    cached = schedule.scheduled_cached_reqs
    eligible = (
        1 <= len(ids) <= max_requests
        and set(ids) == set(previous_ids)
        and all(n == query_tokens for n in counts.values())
        and sum(counts.values()) == schedule.total_num_scheduled_tokens
        and not schedule.scheduled_new_reqs
        and not schedule.finished_req_ids
        and not cached.resumed_req_ids
        and not schedule.scheduled_encoder_inputs
        and not getattr(schedule, "has_structured_output_requests", False)
        and not getattr(schedule, "preempted_req_ids", set())
        and all(
            len(schedule.scheduled_spec_decode_tokens.get(r, ())) == query_tokens - 1
            for r in ids
        )
        and set(cached.req_ids) == set(ids)
        and len(cached.num_computed_tokens) == len(ids)
        and all(n > 0 for n in cached.num_computed_tokens)
    )
    if query_tokens == 1:
        # A one-token tail of chunked prefill is not a decode query. Use the
        # native host phase receipt, without predicting device acceptance.
        outputs = getattr(cached, "num_output_tokens", ())
        eligible = eligible and len(outputs) == len(ids) and all(n > 0 for n in outputs)
    shape = capacities.get(schedule.total_num_scheduled_tokens)
    if not eligible or shape is None:
        return Proposal(sequence, 0, 0, False)
    tokens, mode = shape
    return Proposal(sequence, tokens, mode, True)


def agree(proposals, *, allowed_tokens=(6, 12)):
    """Every rank, including dummy ranks, contributes one proposal per forward."""
    if not proposals or len({p.sequence for p in proposals}) != 1:
        raise ValueError("Mismatched budget sequence")
    eligible = all(p.eligible for p in proposals)
    if eligible and (
        any(p.tokens not in allowed_tokens for p in proposals)
        or len({p.mode for p in proposals}) != 1
    ):
        raise ValueError("Invalid admitted graph envelope")
    return Budget(
        proposals[0].sequence,
        tuple(p.tokens for p in proposals),
        proposals[0].mode if eligible else 0,
        eligible,
    )


class Consumer:
    """One committed plan per actual target; draft retains native coordination."""

    def __init__(self, rank):
        self.rank = rank
        self.sequence = 0
        self.current = None
        self.used = False

    def begin(self, budget):
        if self.current is not None or budget.sequence != self.sequence:
            raise ValueError("Stale, skipped or overlapping budget")
        if not 0 <= self.rank < len(budget.tokens):
            raise ValueError("Rank outside budget group")
        self.current = budget
        self.used = False

    def resolve(self, tokens, mode, is_draft, allow_padding):
        budget = self.current
        if budget is None or is_draft or not budget.admitted:
            return None
        if (
            self.used
            or not allow_padding
            or tokens != budget.tokens[self.rank]
            or mode != budget.mode
        ):
            # A local fallback after peers bypassed their collective would hang.
            raise ValueError("Worker shape disagrees with committed DP budget")
        self.used = True
        maximum = max(budget.tokens)
        return maximum, (maximum,) * len(budget.tokens), budget.mode

    def end(self):
        if self.current is None or (self.current.admitted and not self.used):
            raise ValueError("Committed budget was not consumed")
        self.current = None
        self.sequence += 1
