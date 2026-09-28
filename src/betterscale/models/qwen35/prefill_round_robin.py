"""Greedy prefill grants with a rotating first opportunity, not time fairness.

Decode demand is reserved first. Waiting prefills join the same ring, but their
seat/fence admission remains native. Allocator failure can leave unused budget
for one attempt; rotation still advances, rather than pinning the failed head.
"""

from collections import deque


class PrefillRoundRobin:
    def __init__(self):
        self.ring = deque()
        self.grants = {}

    def plan(self, budget, decodes, prefills, *, admission_slots, minimum_grants=None):
        """Prefills: (request_id, demand, eligible, needs_admission).

        The caller supplies current members, including temporarily blocked ones.
        Eligibility never changes ownership or fences.
        """
        minimum_grants = minimum_grants or {}
        ids = {key for key, *_ in prefills}
        self.ring = deque(key for key in self.ring if key in ids)
        existing = set(self.ring)
        self.ring.extend(key for key, *_ in prefills if key not in existing)
        self.grants = {}
        for key, demand in decodes:
            grant = min(demand, budget)
            self.grants[key] = grant
            budget -= grant
        rows = {key: (demand, eligible, new) for key, demand, eligible, new in prefills}
        for key in self.ring:
            demand, eligible, new = rows[key]
            if not eligible or (new and admission_slots == 0):
                continue
            grant = min(demand, budget)
            if grant < minimum_grants.get(key, 1):
                continue
            self.grants[key] = grant
            budget -= grant
            admission_slots -= int(new)
        # Advance the START, not the last served request. Rotate failed attempts
        # too: a full-budget head rejected by native allocation must not deadlock.
        self.ring.rotate(-1)
        return dict(self.grants)

    def limit(self, request_id, proposed):
        return min(proposed, self.grants.get(request_id, 0))


def prepare(scheduler):
    """Read-only demand/admission preview for the qualified resident-State entry."""
    decodes, prefills = [], []
    minimum_grants = {}
    for request in scheduler.running:
        computed = request.num_computed_tokens
        demand = min(
            request.num_tokens_with_spec + request.num_output_placeholders - computed,
            scheduler.max_model_len - computed - scheduler.num_sampled_tokens_per_step,
        )
        eligible = not (
            (
                request.num_output_placeholders > 0
                and computed + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            )
            or scheduler.current_step + 1 < request.next_decode_eligible_step
        )
        if computed < request.num_prompt_tokens:
            prefills.append((request.request_id, max(demand, 0), eligible, False))
        elif demand > 0 and eligible:
            decodes.append((request.request_id, demand))
    for request in (*scheduler.waiting, *scheduler.skipped_waiting):
        # Grammar/remote/streaming waits retain their native promotion path.
        ready = request.status.name in ("WAITING", "PREEMPTED")
        offer = None
        if ready and not scheduler._waiting_for_resident(request):
            offer = scheduler.residents.offer(
                request.all_token_ids,
                request.cache_salt,
                scheduler.processed_step_seq,
                allow_hit=not request.skip_reading_prefix_cache,
            )
        computed = request.num_computed_tokens
        if offer is not None and offer.warm and computed == 0:
            computed = scheduler.residents.seats[offer.seat].cursor
        demand = max(request.num_tokens - computed, 0)
        # Native waiting admission can expand a one-token hot hit to MTP width.
        # Reserve that indivisible row; a later cap must not truncate padding.
        if demand == 1:
            demand = 1 + scheduler.num_spec_tokens
            minimum_grants[request.request_id] = demand
        prefills.append((request.request_id, demand, offer is not None, True))
    slots = max(
        scheduler.max_num_running_reqs
        - len(scheduler.running)
        - scheduler.num_waiting_for_streaming_input,
        0,
    )
    return scheduler._prefill_round_robin.plan(
        scheduler.max_num_scheduled_tokens,
        decodes,
        prefills,
        admission_slots=slots,
        minimum_grants=minimum_grants,
    )
