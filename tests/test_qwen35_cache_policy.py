"""CPU policy/lifetime gates; native TP2 numerical evidence is separate."""

from types import SimpleNamespace as S
import pytest
from betterscale.models.qwen35.cache_policy import CachePolicy
from betterscale.models.qwen35.resident_leases import Offer, Frontier
from test_qwen35_cache_actions import setup, receipt


def fixture():
    s, c = setup(2)
    pool = s.kv_cache_manager.block_pool
    pool.touch = lambda group: None
    pool.get_usage = lambda: 1 - len(pool.free) / 8
    s.requests = {}
    p = CachePolicy(s, 0.2)
    return s, c, p


def complete(c, n):
    for rank in (1, 0):
        c.receive(receipt(c, n, rank))


def test_backup_two_round_window_retains_device_and_waits_quorum():
    s, c, p = fixture()
    seat = s.residents.seats[0]
    before = (seat.epoch, seat.tokens, seat.blocks)
    p.last_scheduled[0] = 1
    p.after_schedule({})
    p.after_schedule({})
    assert not c.pending
    p.after_schedule({})
    n = next(iter(c.pending))
    c.receive(receipt(c, n, 0))
    assert not c.host and seat.io_owner == n
    c.receive(receipt(c, n, 1))
    assert (seat.epoch, seat.tokens, seat.blocks) == before and seat.io_owner is None
    assert c.host and not p.needs_turn()
    assert s.residents.evict_hot(7)  # eviction remains independent of backup
    assert c.host


def test_two_round_absence_is_not_writer_completion():
    s, c, p = fixture()
    s.processed_step_seq = 6
    for _ in range(5):
        p.after_schedule({})
    assert not c.pending
    s.processed_step_seq = 7
    p.after_schedule({})
    assert c.pending


def test_eviction_without_host_copy_never_queues_writeback():
    s, c, p = fixture()
    assert s.residents.evict_hot(7)
    assert not c.pending and not c.host and not p.needs_turn()


def test_host_lru_drop_quorum_before_capacity_reuse_and_no_rebackup_loop():
    s, c, p = fixture()
    c.host_bytes = 320
    p.after_schedule({})
    p.after_schedule({})
    n = next(iter(c.pending))
    complete(c, n)
    key = next(iter(c.host))
    # A new exact frontier requires another copy, but current host is full.
    s.residents.seats[0].tokens = tuple(range(131))
    p.after_schedule({})
    n = next(iter(c.pending))
    assert c.pending[n].command["kind"] == "drop"
    c.receive(receipt(c, n, 0))
    assert c.allocated_host_bytes == 320
    p.after_schedule({})
    assert len(c.pending) == 1
    c.receive(receipt(c, n, 1))
    assert c.allocated_host_bytes == 0
    p.after_schedule({})
    n = next(iter(c.pending))
    complete(c, n)
    assert key not in c.host
    new = next(iter(c.host))
    complete(c, c.drop(new))
    for _ in range(4):
        p.after_schedule({})
    assert not c.pending  # no infinite cyclic rebackup after host eviction


def test_host_hit_restores_and_device_hit_does_not_move():
    s, c, p = fixture()
    p.after_schedule({})
    p.after_schedule({})
    complete(c, next(iter(c.pending)))
    req = S(
        all_token_ids=list(range(132)), cache_salt=None, skip_reading_prefix_cache=False
    )
    p.restore(req)
    assert not c.pending  # hot device first
    s.residents.evict_hot(7)
    p.restore(req)
    n = next(iter(c.pending))
    assert c.pending[n].command["kind"] == "load"
    assert c.blocks_prompt(req)
    complete(c, n)
    assert not c.blocks_prompt(req)
    assert s.residents.offer(req.all_token_ids, None, 7).warm


def test_host_drop_pins_skip_lookup_and_oversize_backup_is_not_work():
    s, c, p = fixture()
    c.host_bytes = 319
    assert not p.needs_turn()
    c.host_bytes = 320
    p.after_schedule({})
    p.after_schedule({})
    complete(c, next(iter(c.pending)))
    key = next(iter(c.host))
    s.residents.evict_hot(7)
    c.drop(key)
    req = S(
        all_token_ids=list(range(132)), cache_salt=None, skip_reading_prefix_cache=False
    )
    p.restore(req)
    assert len(c.pending) == 1


def test_live_owner_backup_requires_drain_and_pins_native_pages():
    s, c, p = fixture()
    seat = s.residents.seats[0]
    blocks = seat.blocks
    seat.blocks = None
    seat.tokens = ()
    seat.owner = "a"
    s.residents.requests["a"] = 0
    req = S(last_sched_seq=8, cache_salt=None)
    s.requests["a"] = req
    f = Frontier(list(range(130)), cursor=129)
    s._frontiers = {("a", 0, seat.epoch): f}
    s.kv_cache_manager.get_blocks = lambda rid: blocks
    assert p.frontier(seat) is None
    req.last_sched_seq = 7
    pins = []
    s.kv_cache_manager.block_pool.touch = lambda group: pins.extend(group)
    p.after_schedule({})
    p.after_schedule({})
    n = next(iter(c.pending))
    assert len(pins) == 2 and seat.io_owner == n
    complete(c, n)
    assert seat.owner == "a" and seat.blocks is None and seat.io_owner is None
    assert c.host


def test_host_lru_touch_moves_entry_and_pending_load_keeps_it_pinned():
    s, c, p = fixture()
    c.store(0, "old")
    complete(c, 1)
    c.host["new"] = c.host["old"]
    c.touch("old")
    assert list(c.host) == ["new", "old"]
    n = c.load("old", 2)
    with pytest.raises(ValueError, match="pinned"):
        c.drop("old")
    complete(c, n)


def test_runnable_skips_copy_owner_but_preserves_capacity_and_queue_order():
    from vllm.v1.core.sched.request_queue import FCFSRequestQueue

    s, c, p = fixture()

    class Request:
        def __init__(self, rid, tokens):
            self.request_id = rid
            self.all_token_ids = tokens
            self.cache_salt = None
            self.skip_reading_prefix_cache = False

    a = Request("a", [999])
    b = Request("b", [888])
    wait = Request("wait", list(range(131)))
    seat = s.residents.seats[1]
    seat.owner = "a"
    seat.io_owner = 99
    s.residents.requests = {"a": 1, "b": 2}
    s.running = [a, b]
    s.waiting = FCFSRequestQueue([wait])
    s.skipped_waiting = FCFSRequestQueue()
    s.max_num_running_reqs = 16
    c.store(0, "inflight")
    with p.runnable():
        assert s.running == [b] and not s.waiting and s.max_num_running_reqs == 15
    assert (
        s.running == [a, b]
        and list(s.waiting) == [wait]
        and s.max_num_running_reqs == 16
    )


def test_real_page_pool_retains_active_backup_through_abort():
    import torch
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        FullAttentionSpec,
    )
    from betterscale.models.qwen35.cache_actions import CacheActions
    from betterscale.models.qwen35.resident_leases import ResidentLeases

    spec = FullAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=4, dtype=torch.bfloat16
    )
    config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["fa"], spec)],
    )
    manager = KVCacheManager(
        config, max_model_len=1024, scheduler_block_size=128, hash_block_size=128
    )

    def release(blocks):
        for group in blocks.blocks:
            manager.block_pool.free_blocks(reversed(group))

    s = S(
        kv_cache_manager=manager,
        processed_step_seq=7,
        block_size=128,
        requests={"a": S(last_sched_seq=7)},
        _release_resident_blocks=release,
    )
    s.residents = ResidentLeases(2, release_blocks=release)
    seat = s.residents.seats[0]
    seat.owner = "a"
    s.residents.requests["a"] = 0
    blocks = manager.create_kv_cache_blocks(
        (tuple(manager.block_pool.get_new_blocks(2)),)
    )
    c = CacheActions(s, 2, 320, resident_bytes=64, block_bytes=128)
    c.endpoint = "/test/unused"
    n = c.backup(0, "active", tuple(range(130)), None, blocks)
    assert [b.ref_cnt for b in blocks.blocks[0]] == [2, 2]
    s.residents.retire("a", fence=7)
    release(blocks)  # request's original ownership, as native abort would do
    assert [b.ref_cnt for b in blocks.blocks[0]] == [1, 1]
    assert s.residents.offer([1, 2], None, 7).seat == 1
    c.cancel(n)
    c.receive(receipt(c, n, 0))
    assert [b.ref_cnt for b in blocks.blocks[0]] == [1, 1]
    c.receive(receipt(c, n, 1))
    assert [b.ref_cnt for b in blocks.blocks[0]] == [0, 0]
    assert manager.block_pool.get_num_free_blocks() == 7
    assert c.allocated_host_bytes == 320  # cancelled host payload pending drop
    complete(c, next(iter(c.pending)))
    assert c.allocated_host_bytes == 0


def test_automatic_receipt_history_is_bounded():
    s, c, p = fixture()
    for i in range(300):
        c.completed.append(dict(operation=i))
    assert len(c.snapshot()["completed"]) == 256
    assert c.result(299) == dict(operation=299)


def test_independent_capacity_watermarks():
    s, c, p = fixture()
    p.watermark = 0.7
    assert not p.under_pressure()
    s.kv_cache_manager.block_pool.get_usage = lambda: 0.71
    assert p.under_pressure()
    s.kv_cache_manager.block_pool.get_usage = lambda: 0
    s.residents.seats[1].owner = "b"
    s.residents.seats[2].owner = "c"
    assert p.under_pressure()
