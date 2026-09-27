"""Watermark backup and two independent LRUs, owned by the scheduler thread.

No compulsory writeback on eviction. Completed checkpoints are immutable prefix
identities, not claims that a subsequently advanced device seat is still backed.
"""

from contextlib import contextmanager


class CachePolicy:
    def __init__(self, scheduler, watermark=0.7):
        if not 0 < watermark <= 1:
            raise ValueError("State cache watermark must be in (0, 1]")
        self.scheduler = scheduler
        self.cache = scheduler.cache_actions
        self.watermark = watermark
        self.round = 0
        self.last_scheduled = {}
        self.saved = {}
        self.last_backup = {}

    def frontier(self, seat):
        s = self.scheduler
        if seat.io_owner is not None or seat.fence > s.processed_step_seq:
            return None
        if seat.owner is None:
            return (seat.tokens, seat.cache_salt, seat.blocks) if seat.tokens else None
        request = s.requests[seat.owner]
        if request.last_sched_seq > s.processed_step_seq:
            return None
        frontier = s._frontiers.get((seat.owner, seat.index, seat.epoch))
        tokens = frontier.checkpoint() if frontier else ()
        if not tokens:
            return None
        blocks = s.kv_cache_manager.get_blocks(seat.owner)
        count = (len(tokens) - 2 + s.block_size) // s.block_size
        groups = tuple(tuple(group[:count]) for group in blocks.blocks)
        if any(len(group) != count for group in groups):
            return None
        return (
            tokens,
            request.cache_salt,
            s.kv_cache_manager.create_kv_cache_blocks(groups),
        )

    def under_pressure(self):
        s = self.scheduler
        seats = s.residents.seats
        occupied = sum(
            bool(
                x.owner is not None
                or x.tokens
                or x.io_owner is not None
                or x.fence > s.processed_step_seq
            )
            for x in seats
        )
        pool = s.kv_cache_manager.block_pool
        return max(occupied / len(seats), pool.get_usage()) >= self.watermark

    def candidates(self):
        if not self.under_pressure():
            return []
        candidates = []
        for seat in self.scheduler.residents.seats:
            frontier = self.frontier(seat)
            if frontier is None:
                continue
            tokens, salt, blocks = frontier
            identity = seat.epoch, tokens, salt
            if self.saved.get(seat.index) == identity:
                continue
            if any(
                cp.tokens == tokens and cp.salt == salt
                for cp in self.cache.host.values()
            ):
                self.saved[seat.index] = identity
                continue
            size = (
                self.cache.resident_bytes
                + len(blocks.blocks[0]) * self.cache.block_bytes
            )
            if size <= self.cache.host_bytes:
                candidates.append((seat, frontier, identity, size))
        return sorted(
            candidates,
            key=lambda item: (self.last_backup.get(item[0].index, -1), item[0].touched),
        )

    def needs_turn(self):
        # No spinning behind DMA or repeatedly recopying host-LRU victims.
        return not self.cache.pending and bool(self.candidates())

    def after_schedule(self, scheduled):
        self.round += 1
        s = self.scheduler
        for rid in scheduled:
            seat = s.residents.seats[s.residents.requests[rid]]
            self.last_scheduled[seat.index] = self.round
            s.residents.clock += 1
            seat.touched = s.residents.clock
        if self.cache.pending:
            return  # one bounded maintenance action, no unbounded pinned set
        for seat, (tokens, salt, blocks), identity, size in self.candidates():
            if self.round - self.last_scheduled.get(seat.index, 0) < 2:
                continue
            if self.cache.allocated_host_bytes + size > self.cache.host_bytes:
                # Oldest unpinned host object; bytes remain charged until TP quorum.
                key = next(iter(self.cache.host), None)
                if key is not None:
                    self.cache.drop(key)
                return
            key = f"auto:{self.cache.sequence + 1}"
            self.cache.backup(seat.index, key, tokens, salt, blocks)
            self.saved[seat.index] = identity
            self.last_backup[seat.index] = self.round
            return

    def restore(self, request):
        s, cache = self.scheduler, self.cache
        if request.skip_reading_prefix_cache or cache.blocks_prompt(request):
            return
        offer = s.residents.offer(
            request.all_token_ids, request.cache_salt, s.processed_step_seq
        )
        pinned = {p.checkpoint.key for p in cache.pending.values()}
        matches = [
            cp
            for cp in cache.host.values()
            if cp.key not in pinned
            and cp.salt == request.cache_salt
            and tuple(request.all_token_ids[: len(cp.tokens)]) == cp.tokens
        ]
        if not matches:
            return
        checkpoint = max(matches, key=lambda cp: len(cp.tokens))
        if offer is not None and offer.warm:
            cache.touch(checkpoint.key)
            return  # device residency wins; no needless host round trip
        if offer is None:
            return  # normal capacity/preemption path, never wait for backup
        seat = s.residents.seats[offer.seat]
        owned = len(seat.blocks.blocks[0]) if seat.blocks else 0
        if (
            s.kv_cache_manager.block_pool.get_num_free_blocks() + owned
            >= checkpoint.block_count
        ):
            cache.load(checkpoint.key, seat.index)

    @contextmanager
    def runnable(self):
        """Exclude I/O-pinned owners and restore waiters, not other ready work."""
        s = self.scheduler
        for request in list(s.waiting):
            self.restore(request)
        running = list(s.running)
        held = [
            r
            for r in running
            if s.residents.seats[s.residents.requests[r.request_id]].io_owner
            is not None
        ]
        queues = [
            (q, [r for r in q if self.cache.blocks_prompt(r)])
            for q in (s.waiting, s.skipped_waiting)
        ]
        s.running[:] = [r for r in running if r not in held]
        capacity = s.max_num_running_reqs
        s.max_num_running_reqs -= len(held)
        for q, requests in queues:
            q.remove_requests(requests)
        try:
            yield
        finally:
            s.max_num_running_reqs = capacity
            live = set(s.running)
            s.running[:] = [r for r in running if r in live or r in held] + [
                r for r in s.running if r not in running
            ]
            for q, requests in queues:
                for request in reversed(requests):
                    q.prepend_request(request)
