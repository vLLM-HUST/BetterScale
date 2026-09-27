"""Native asynchronous scheduling with resident-State admission/lifetime hooks.

The native scheduler retains allocation, preemption and async publication.
A bounded round-robin policy assigns prefill grants after reserving decode.
This adapter supplies exact hot checkpoints and leaves token pages in its pool.
"""

from dataclasses import dataclass, field, fields

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

from .resident_leases import Frontier, ResidentLeases
from .prefill_round_robin import PrefillRoundRobin, prepare
from .fair_schedule import bind


@dataclass
class StateSchedule(SchedulerOutput):
    resident_leases: dict[str, tuple[int, int]] = field(default_factory=dict)
    generation_limits: dict[str, int] = field(default_factory=dict)
    sampling_requests: set[str] = field(default_factory=set)


class LiveStateScheduler(AsyncScheduler):
    _fair_native_schedule = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.kv_cache_config.has_mamba_layers or self.connector is not None:
            raise ValueError(
                "live State requires a pure attention page pool, no connector"
            )
        if self.max_num_running_reqs != 16:
            raise ValueError("live State currently admits E16/R20")
        # Reuse the native deferred-free fence, including abort/preemption. A
        # resident cannot be advertised hot while a queued wave can advance it.
        if getattr(self, "_balance_enabled", False):
            raise ValueError(
                "Prefill round robin requires Ascend balance scheduling disabled"
            )
        self._prefill_round_robin = PrefillRoundRobin()
        if type(self)._fair_native_schedule is None:
            type(self)._fair_native_schedule = bind(
                AsyncScheduler.schedule,
                balance_enabled=getattr(self, "_balance_enabled", False),
            )
        self.defer_block_free = True
        self.residents = ResidentLeases(
            20, release_blocks=self._release_resident_blocks
        )
        self._frontiers = {}
        self._pending_hot = {}
        self._offers = {}
        self._generation_limits = {}
        manager = self.kv_cache_manager
        self._native_allocate = manager.allocate_slots
        manager.allocate_slots = self._allocate
        manager.get_computed_blocks = self._computed
        # FA-only hash hits do not establish a matching GDN/MTP checkpoint.
        # Retained exact residents own this index; native allocation/refcounts
        # remain enabled, but cannot publish incomplete tail pages as hash hits.
        manager.cache_blocks = lambda request, num_computed_tokens: None

    def _release_resident_blocks(self, blocks):
        for group in blocks.blocks:
            self.kv_cache_manager.block_pool.free_blocks(reversed(group))

    def _computed(self, request):
        if self._waiting_for_resident(request):
            self._offers.pop(request.request_id, None)
            return self.kv_cache_manager.empty_kv_cache_blocks, 0
        offer = self.residents.offer(
            request.all_token_ids,
            request.cache_salt,
            self.processed_step_seq,
            allow_hit=not request.skip_reading_prefix_cache,
        )
        self._offers[request.request_id] = offer
        cursor = self.residents.seats[offer.seat].cursor if offer and offer.warm else 0
        manager = self.kv_cache_manager
        if manager.log_stats and not request.skip_reading_prefix_cache:
            manager.prefix_cache_stats.record(
                num_tokens=request.num_tokens,
                num_hits=cursor,
                preempted=request.num_preemptions > 0,
            )
        if offer is not None and offer.warm:
            seat = self.residents.seats[offer.seat]
            return seat.blocks, seat.cursor
        return self.kv_cache_manager.empty_kv_cache_blocks, 0

    def _waiting_for_resident(self, request):
        # A known length frontier is already frozen, but the queued frame can
        # still own its FA pages. Wait for that resident's native writer fence
        # instead of needlessly sending its continuation to a cold empty seat.
        for key, (_, _, fence, salt, _) in self._pending_hot.items():
            frontier = self._frontiers.get(key)
            if (
                fence > self.processed_step_seq
                and salt == request.cache_salt
                and not request.skip_reading_prefix_cache
                and frontier is not None
                and frontier.known
                and frontier.remaining == 0
                and request.all_token_ids[: len(frontier.tokens)] == frontier.tokens
            ):
                return True
        return False

    def _allocate(self, request, *args, **kwargs):
        rid = request.request_id
        new = rid not in self.residents.requests
        if new and self._waiting_for_resident(request):
            return None
        offer = self._offers.get(rid) if new else None
        if new and offer is None:
            offer = self.residents.offer(
                request.all_token_ids,
                request.cache_salt,
                self.processed_step_seq,
                allow_hit=False,
            )
        if new and offer is None:
            return None
        if new and not offer.warm:
            offer = self.residents.discard_victim(offer, self.processed_step_seq)
            self._offers[rid] = offer
        kwargs["delay_cache_blocks"] = True
        result = self._native_allocate(request, *args, **kwargs)
        # Reclaim inactive hot residents before native scheduling preempts an
        # active request. Every retry changes actual available page capacity.
        while result is None and self.residents.evict_hot(
            self.processed_step_seq, exclude=offer.seat if offer else None
        ):
            result = self._native_allocate(request, *args, **kwargs)
        if result is None:
            return None
        if new:
            seat = self.residents.claim(rid, offer, self.processed_step_seq)
            cursor = seat.cursor if offer.warm else 0
            self.residents.transferred(rid)
            remaining = request.max_tokens - request.num_output_tokens
            self._generation_limits[rid] = remaining
            self._frontiers[rid, seat.index, seat.epoch] = Frontier(
                list(request.all_token_ids), cursor=cursor, remaining=remaining
            )
            self._offers.pop(rid, None)
        return result

    def schedule(self, *args, **kwargs):
        prepare(self)
        output = self._fair_native_schedule(*args, **kwargs)
        leases = {}
        for rid in output.num_scheduled_tokens:
            seat = self.residents.seats[self.residents.requests[rid]]
            leases[rid] = seat.index, seat.epoch
        return StateSchedule(
            **{f.name: getattr(output, f.name) for f in fields(SchedulerOutput)},
            resident_leases=leases,
            generation_limits={
                rid: self._generation_limits.pop(rid)
                for rid in leases
                if rid in self._generation_limits
            },
            sampling_requests={
                rid for rid in leases if not self.requests[rid].is_prefill_chunk
            },
        )

    def update_from_output(self, scheduler_output, model_runner_output):
        # Observe raw lists before inherited EOS/length trimming, including the
        # final queued frame of a request already removed by normal scheduling.
        for rid, query in scheduler_output.num_scheduled_tokens.items():
            lease = scheduler_output.resident_leases[rid]
            frontier = self._frontiers.get((rid, *lease))
            index = model_runner_output.req_id_to_index.get(rid)
            if frontier is None or index is None:
                continue
            sampled = model_runner_output.sampled_token_ids[index]
            drafts = len(scheduler_output.scheduled_spec_decode_tokens.get(rid, ()))
            frontier.advance(query, drafts, sampled)
        result = super().update_from_output(scheduler_output, model_runner_output)
        self._publish_completed_residents()
        return result

    def _free_request_blocks(self, request):
        rid = request.request_id
        index = self.residents.requests.get(rid)
        if index is not None:
            seat = self.residents.seats[index]
            fence = request.last_sched_seq
            key = rid, index, seat.epoch
            # Retain only the actual final device frontier, after all queued
            # writers. No terminal rollback or historical checkpoints this cut.
            if request.status in (
                RequestStatus.FINISHED_STOPPED,
                RequestStatus.FINISHED_LENGTH_CAPPED,
            ):
                blocks = self.kv_cache_manager.get_blocks(rid)
                blocks = self.kv_cache_manager.create_kv_cache_blocks(
                    tuple(tuple(group) for group in blocks.blocks)
                )
                for group in blocks.blocks:
                    self.kv_cache_manager.block_pool.touch(group)
                self._pending_hot[key] = (
                    index,
                    seat.epoch,
                    fence,
                    request.cache_salt,
                    blocks,
                )
            else:
                self._frontiers.pop(key, None)
            self.residents.retire(rid, fence=fence)
            self._offers.pop(rid, None)
            self._generation_limits.pop(rid, None)
        super()._free_request_blocks(request)
        # Handles a finish/abort outside update_from_output as well.
        self._publish_completed_residents()

    def _publish_completed_residents(self):
        for key, (index, epoch, fence, salt, blocks) in list(self._pending_hot.items()):
            if fence > self.processed_step_seq:
                continue
            seat = self.residents.seats[index]
            assert seat.owner is None and seat.epoch == epoch
            frontier = self._frontiers.pop(key, None)
            tokens = frontier.checkpoint() if frontier is not None else ()
            if tokens:
                # Trim speculative lookahead pages; retain the partial tail page.
                count = (len(tokens) - 2 + self.block_size) // self.block_size
                kept = tuple(tuple(group[:count]) for group in blocks.blocks)
                if any(len(group) != count for group in kept):
                    tokens = ()
                else:
                    for group in blocks.blocks:
                        self.kv_cache_manager.block_pool.free_blocks(
                            reversed(group[count:])
                        )
                    seat.tokens, seat.cache_salt = tokens, salt
                    seat.blocks = self.kv_cache_manager.create_kv_cache_blocks(kept)
            if not tokens:
                self._release_resident_blocks(blocks)
            del self._pending_hot[key]

    def reset_prefix_cache(self, *args, **kwargs):
        self.residents.invalidate_hot()
        for key in self._pending_hot:
            frontier = self._frontiers.get(key)
            if frontier is not None:
                frontier.known = False
        return super().reset_prefix_cache(*args, **kwargs)
