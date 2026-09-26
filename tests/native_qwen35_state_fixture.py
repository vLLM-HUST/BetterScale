"""CPU exercise of State lifetime hooks against the pinned real token-page manager."""

from collections import deque
import torch
from vllm import SamplingParams
from vllm.v1.request import Request, RequestStatus
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    FullAttentionSpec,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from betterscale.models.qwen35.resident_leases import ResidentLeases
from betterscale.models.qwen35.seat_scheduler import LiveStateScheduler

spec = FullAttentionSpec(
    block_size=128, num_kv_heads=1, head_size=4, dtype=torch.bfloat16
)
config = KVCacheConfig(
    num_blocks=16, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["fa"], spec)]
)
manager = KVCacheManager(
    config, max_model_len=1024, scheduler_block_size=128, hash_block_size=128
)
# The native queue is not mocked/reimplemented: this fixture targets its State
# hooks only. NPU serving separately exercises actual AsyncScheduler construction.
s = object.__new__(LiveStateScheduler)
s.kv_cache_manager = manager
s.block_size = 128
s.defer_block_free = True
s.deferred_frees = deque()
s.sched_step_seq = s.processed_step_seq = 1
s.residents = ResidentLeases(20, release_blocks=s._release_resident_blocks)
s._frontiers = {}
s._pending_hot = {}
s._offers = {}
s._generation_limits = {}
s._native_allocate = manager.allocate_slots


def request(name, tokens):
    return Request(name, tokens, SamplingParams(max_tokens=4), None)


a = request("A", list(range(257)))
blocks, computed = s._computed(a)
assert computed == 0
assert s._allocate(a, 257) is not None
seat = s.residents.seats[s.residents.requests["A"]]
s._frontiers["A", seat.index, seat.epoch].advance(257, 0, [999])
a.append_output_token_ids([999])
a.num_computed_tokens = 257
a.last_sched_seq = 1
a.status = RequestStatus.FINISHED_LENGTH_CAPPED
s._free_request_blocks(a)
assert seat.cursor == 257 and [b.ref_cnt for b in seat.blocks.blocks[0]] == [1, 1, 1]

b = request("B", list(range(257)) + [999, 1000])
hit, cursor = s._computed(b)
assert cursor == 257 and len(hit.blocks[0]) == 3
assert (
    s._allocate(b, 2, num_new_computed_tokens=cursor, new_computed_blocks=hit)
    is not None
)
assert s.residents.requests["B"] == 0
assert all(x.ref_cnt == 1 for x in manager.get_blocks("B").blocks[0])
frontier = s._frontiers["B", seat.index, seat.epoch]
frontier.advance(2, 0, [1001])
b.append_output_token_ids([1001])
s.sched_step_seq = 3
s.processed_step_seq = 2
b.num_computed_tokens = 260
b.last_sched_seq = 3
b.status = RequestStatus.FINISHED_STOPPED
s._free_request_blocks(b)
assert not s.residents.offer(frontier.tokens, None, 2).warm
# The existing candidates are updated in place by the final queued wave.
# Retention describes that actual frontier, not the CPU-trimmed terminal prefix.
frontier.advance(1, 0, [1002])
s.processed_step_seq = 3
s._drain_deferred_frees()
s._publish_completed_residents()
assert seat.cursor == 260
assert s.residents.offer(frontier.tokens, None, 3).warm
assert not s.residents.offer(frontier.tokens[:-1], None, 3).warm
s.residents.invalidate_hot()
assert manager.block_pool.get_num_free_blocks() == 15
# Preemption invalidates the whole seat, while native deferred freeing keeps
# its regular-attention pages alive until the last already-submitted writer.
c = request("C", list(range(129)))
s._computed(c)
assert s._allocate(c, 129) is not None
old_index = s.residents.requests["C"]
old_epoch = s.residents.seats[old_index].epoch
c.last_sched_seq = s.sched_step_seq = 4
c.status = RequestStatus.PREEMPTED
s._free_request_blocks(c)
assert "C" not in s.residents.requests
assert not s.residents.seats[old_index].tokens
assert manager.block_pool.get_num_free_blocks() == 13
# Same request ID can restart on another empty seat, not steal its pending seat.
s._computed(c)
assert s._allocate(c, 129) is not None
assert s.residents.requests["C"] != old_index
s.processed_step_seq = 4
s._drain_deferred_frees()
assert manager.block_pool.get_num_free_blocks() == 13
c.status = RequestStatus.FINISHED_ABORTED
s._free_request_blocks(c)
assert manager.block_pool.get_num_free_blocks() == 15
assert not any(seat.tokens for seat in s.residents.seats)
assert not s._frontiers and not s._pending_hot

# A length-frozen frontier must wait for its queued writer, not silently miss
# just because another empty seat is available when the next HTTP turn arrives.
d = Request("D", [1, 2, 3], SamplingParams(max_tokens=1), None)
s._computed(d)
assert s._allocate(d, 3) is not None
index = s.residents.requests["D"]
epoch = s.residents.seats[index].epoch
terminal = s._frontiers["D", index, epoch]
terminal.advance(3, 0, [4])
d.append_output_token_ids([4])
d.last_sched_seq = s.sched_step_seq = 5
d.status = RequestStatus.FINISHED_LENGTH_CAPPED
s._free_request_blocks(d)
e = request("E", [1, 2, 3, 4, 5])
assert s._computed(e)[1] == 0
assert s._allocate(e, 5) is None
terminal.advance(3, 2, [6, 7, 8])
assert terminal.checkpoint() == (1, 2, 3, 4)
s.processed_step_seq = 5
s._drain_deferred_frees()
s._publish_completed_residents()
hit, cursor = s._computed(e)
assert cursor == 3
assert s._allocate(e, 2, num_new_computed_tokens=cursor, new_computed_blocks=hit)
assert s.residents.requests["E"] == index
e.last_sched_seq = 5
e.status = RequestStatus.FINISHED_ABORTED
s._free_request_blocks(e)
assert manager.block_pool.get_num_free_blocks() == 15
print(
    "PASS: native page references, partial-tail hot handoff, final-writer fence, whole-seat preemption"
)
