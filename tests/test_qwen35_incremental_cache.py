"""Real native-pool ownership and CPU State byte witnesses, no NPU imports."""

from types import SimpleNamespace as S

import pytest
import torch
from test_qwen35_cache_actions import receipt

from betterscale.live.runtime.host_state import (
    HostStateDomainSelection,
    HostStateSelection,
)
from betterscale.live.runtime.page_state import PageStateStore
from betterscale.models.qwen35.cache_actions import CacheActions
from betterscale.models.qwen35.cache_pages import page_keys
from betterscale.models.qwen35.resident_leases import ResidentLeases


def setup(pages=100):
    from vllm.v1.core.block_pool import BlockPool

    pool = BlockPool(num_gpu_blocks=pages + 1, enable_caching=True, hash_block_size=128)
    manager = S(block_pool=pool, create_kv_cache_blocks=lambda groups: S(blocks=groups))

    def release(blocks):
        pool.free_blocks(reversed(blocks.blocks[0]))

    s = S(
        kv_cache_manager=manager,
        processed_step_seq=7,
        block_size=128,
        _release_resident_blocks=release,
    )
    s.residents = ResidentLeases(3, release_blocks=release)
    c = CacheActions(
        s, 2, 1 << 28, resident_bytes=64, block_bytes=128, incremental=True
    )
    c.endpoint = "/unused"
    s.cache_actions = c
    seat = s.residents.seats[0]
    seat.epoch, seat.fence = 3, 7
    seat.tokens = tuple(range(pages * 128 + 1))
    seat.blocks = manager.create_kv_cache_blocks((tuple(pool.get_new_blocks(pages)),))
    return s, c, pool


def done(c, n):
    for rank in (1, 0):
        c.receive(receipt(c, n, rank))


def test_100_pages_20_overwritten_restore_only_20_without_eviction_of_80_hits():
    s, c, pool = setup()
    n = c.store(0, "A")
    done(c, n)
    assert pool.get_num_free_blocks() == 100
    b = pool.get_new_blocks(20)
    assert len(c.pages.by_key) == 80
    pool.free_blocks(b)
    n = c.load("A", 2)
    command = c.pending[n].command
    assert len(command["missing"]) == 20
    assert len(command["blocks"]) == 100
    assert pool.get_num_free_blocks() == 0
    c.receive(receipt(c, n, 0))
    assert not s.residents.seats[2].tokens
    c.receive(receipt(c, n, 1))
    assert s.residents.seats[2].cursor == 12800
    assert len(c.pages.by_key) == 100
    assert c.completed[-1]["restored_pages"] == 20


def test_incremental_backup_charges_shared_prefix_once_and_lru_keeps_live_refs():
    s, c, _pool = setup(4)
    seat = s.residents.seats[0]
    seat.tokens = tuple(range(500))
    n = c.backup(0, "A", seat.tokens, None, seat.blocks)
    done(c, n)
    first = c.allocated_host_bytes
    # Advance inside a tail block: three sealed pages shared, final page new.
    tokens = seat.tokens + (500,)
    n = c.backup(0, "B", tokens, None, seat.blocks)
    assert c.allocated_host_bytes == first + 64 + 128
    assert c.pending[n].checkpoint.pages[:3] == c.host["A"].pages[:3]
    assert c.pending[n].checkpoint.pages[3] != c.host["A"].pages[3]
    done(c, n)
    n = c.drop("A")
    assert c.allocated_host_bytes == first + 64 + 128
    c.receive(receipt(c, n, 0))
    assert c.allocated_host_bytes == first + 64 + 128
    c.receive(receipt(c, n, 1))
    assert c.allocated_host_bytes == first


def test_abort_restore_keeps_pages_pinned_until_quorum_and_reuse_invalidates():
    s, c, pool = setup(4)
    done(c, c.store(0, "A"))
    n = c.load("A", 2)
    assert c.pending[n].command["missing"] == []
    c.cancel(n)
    c.receive(receipt(c, n, 0))
    assert pool.get_num_free_blocks() == 0
    c.receive(receipt(c, n, 1))
    assert pool.get_num_free_blocks() == 4
    assert not s.residents.seats[2].tokens
    pool.get_new_blocks(1)
    assert len(c.pages.by_key) == 3


def test_sealed_identity_includes_salt_prefix_and_draft_lookahead_not_tail_version():
    tokens = tuple(range(257))
    a = page_keys(tokens, "salt", 128, 2, "A")
    b = page_keys(tokens, "salt", 128, 2, "B")
    assert a[0] == b[0] and a[1] != b[1]
    for changed, salt in [
        (tokens, "other"),
        ((99,) + tokens[1:], "salt"),
        (tokens[:128] + (999,) + tokens[129:], "salt"),
    ]:
        assert page_keys(changed, salt, 128, 2, "C")[0] != a[0]
    assert page_keys(tuple(range(129)), None, 128, 1, "A")[0].startswith("tail:")


def state(tensor, domain):
    return S(
        tensor=tensor,
        domain=domain,
        storage_dtype=tensor.dtype,
        block_shape=tuple(tensor.shape[1:]),
        num_blocks=len(tensor),
        physical_blocks_per_logical_block=1,
        leading_physical_blocks=0,
        logical_block_bytes=tensor[0].numel() * tensor.element_size(),
    )


def select(domain, *ids):
    return HostStateSelection((HostStateDomainSelection(domain, ids),))


def test_real_bytes_incremental_manifest_partial_restore_and_exact_host_charge():
    residents, pages = object(), object()
    gdn = state(torch.arange(8, dtype=torch.float32).reshape(4, 2), residents)
    fa = state(torch.arange(16, dtype=torch.float32).reshape(8, 2), pages)
    states = [("gdn", gdn), ("fa", fa)]
    store = PageStateStore(memory_budget_bytes=40)
    a = {
        "resident:A": select(residents, 0),
        "page0": select(pages, 0),
        "tail:A": select(pages, 1),
    }
    transfer = store.transfer("A", states, a, store=True, stream=None)
    assert transfer.byte_length == 24
    transfer.result()
    old = fa.tensor[:2].clone()
    fa.tensor[1].add_(100)
    gdn.tensor[0].add_(10)
    b = {
        "resident:B": select(residents, 0),
        "page0": select(pages, 0),
        "tail:B": select(pages, 1),
    }
    transfer = store.transfer("B", states, b, store=True, stream=None)
    assert transfer.byte_length == 16  # new GDN + dirty tail only
    transfer.result()
    assert store.committed_bytes == 40
    # Only page1 was lost; page0 stays untouched (poison proves no recopy).
    fa.tensor[0].fill_(-1)
    transfer = store.transfer(
        "A",
        states,
        {"resident:A": select(residents, 2), "tail:A": select(pages, 5)},
        store=False,
        stream=None,
    )
    assert transfer.byte_length == 16
    with pytest.raises(ValueError, match="pending"):
        store.release("A")
    transfer.result()
    assert torch.equal(fa.tensor[5], old[1])
    assert fa.tensor[0].tolist() == [-1, -1]
    assert gdn.tensor[2].tolist() == [0, 1]
    store.release("A")
    assert store.committed_bytes == 24
    store.release("B")
    assert store.committed_bytes == 0


def test_host_capacity_refusal_is_before_any_manifest_or_copy():
    d = object()
    st = state(torch.arange(8).reshape(4, 2), d)
    store = PageStateStore(memory_budget_bytes=16)
    with pytest.raises(ValueError, match="capacity"):
        store.transfer(
            "A",
            [("x", st)],
            {"a": select(d, 0), "b": select(d, 1)},
            store=True,
            stream=None,
        )
    assert not store.manifests and not store.pending
    assert store.committed_bytes == 0


def test_overlapping_incremental_actions_refuse_before_changing_device_ownership():
    s, c, pool = setup(4)
    n = c.store(0, "A")
    with pytest.raises(ValueError, match="in flight"):
        c.store(1, "B")
    assert s.residents.seats[0].io_owner == n
    assert pool.get_num_free_blocks() == 0
    done(c, n)
    n = c.load("A", 1)
    with pytest.raises(ValueError, match="pending operation"):
        c.load("A", 2)
    assert s.residents.seats[2].blocks is None
    done(c, n)


def test_device_write_invalidates_only_the_mutable_suffix():
    s, c, pool = setup(4)
    seat = s.residents.seats[0]
    done(c, c.backup(0, "A", seat.tokens, None, seat.blocks))
    keys = c.host["A"].pages
    c.pages.invalidate(seat.blocks.blocks[0][3:])
    assert keys[:3] == tuple(c.pages.by_key)
    assert keys[3] not in c.pages.by_key
    assert pool.get_num_free_blocks() == 0  # invalidation is metadata, not free
    assert c.allocated_host_bytes == 64 + 4 * 128


def test_page_transfer_result_is_idempotent():
    domain = object()
    st = state(torch.arange(4).reshape(2, 2), domain)
    store = PageStateStore(memory_budget_bytes=32)
    handle = store.transfer(
        "A", [("x", st)], {"a": select(domain, 0)}, store=True, stream=None
    )
    assert handle.result() == handle.result() == "A"
    assert store.committed_bytes == 16
    store.release("A")
    assert store.committed_bytes == 0


def test_native_allocation_hook_invalidates_write_suffix_before_dispatch():
    from betterscale.models.qwen35.seat_scheduler import LiveStateScheduler

    s, c, pool = setup(4)
    seat = s.residents.seats[0]
    done(c, c.backup(0, "A", seat.tokens, None, seat.blocks))
    keys = c.host["A"].pages
    s.residents.requests["running"] = 0
    seat.owner = "running"
    s.kv_cache_manager.get_blocks = lambda rid: seat.blocks
    s._native_allocate = lambda *args, **kwargs: seat.blocks
    req = S(request_id="running", num_computed_tokens=385)
    assert LiveStateScheduler._allocate(s, req, 3) is seat.blocks
    assert set(c.pages.by_key) == set(keys[:3])
    assert pool.get_num_free_blocks() == 0


def test_weak_cached_pages_use_lru_not_native_unhashed_free_prepend():
    s, c, pool = setup(4)
    seat = s.residents.seats[0]
    original = seat.blocks.blocks[0]
    keys = page_keys(seat.tokens, None, 128, 4, "A")
    c.pages.remember(keys, original)
    # Two cold objects retire in order; later retirement cannot jump ahead.
    pool.free_blocks(original[:2])
    pool.free_blocks(original[2:])
    assert [b.block_id for b in pool.get_new_blocks(2)] == [
        b.block_id for b in original[:2]
    ]
    assert set(c.pages.by_key) == set(keys[2:])


def test_uncertain_completion_retains_payload_and_reserved_capacity():
    domain = object()
    st = state(torch.arange(4).reshape(2, 2), domain)
    store = PageStateStore(memory_budget_bytes=32)
    transfer = store.transfer(
        "A", [("x", st)], {"a": select(domain, 0)}, store=True, stream=None
    )

    def failed_wait():
        raise RuntimeError("unknown DMA lifetime")

    transfer.event = S(synchronize=failed_wait)
    with pytest.raises(RuntimeError, match="unknown DMA"):
        transfer.result()
    assert store.transfers["A"] is transfer
    assert store.backend._reserved_bytes == 16
    assert store.pending == {"A"}
    with pytest.raises(ValueError, match="pending DMA"):
        store.release("A")


def test_partial_enqueue_drains_before_unwind_or_quarantines(monkeypatch):
    from betterscale.live.runtime.page_state import _CopyBackend

    backend = _CopyBackend(memory_budget_bytes=32)
    calls = []

    def fail_enqueue(*args, **kwargs):
        calls.append("enqueue")
        raise RuntimeError("partial enqueue")

    monkeypatch.setattr(backend, "_enqueue_copies", fail_enqueue)
    lanes = (("fa", S(tensor=S(device=S(type="npu"))), (0,)),)
    payloads = {"fa": object()}
    stream = S(synchronize=lambda: calls.append("drain"))
    with pytest.raises(RuntimeError, match="partial enqueue"):
        backend._copy_lanes(lanes, payloads, to_host=True, stream=stream)
    assert calls == ["enqueue", "drain"]
    assert not backend.quarantined

    def failed_drain():
        raise RuntimeError("drain failed")

    stream.synchronize = failed_drain
    with pytest.raises(RuntimeError, match="drain failed"):
        backend._copy_lanes(lanes, payloads, to_host=True, stream=stream)
    assert backend.quarantined == [(lanes, payloads, stream)]


@pytest.mark.parametrize("incremental", [False, True])
def test_policy_serializes_returning_host_hits_without_blocking_cold_work(incremental):
    from vllm.v1.core.sched.request_queue import FCFSRequestQueue
    from betterscale.models.qwen35.cache_policy import CachePolicy

    class Request(S):
        __hash__ = object.__hash__
        __eq__ = object.__eq__

    s, c, pool = setup(8)
    if not incremental:
        c.pages = None
    seat = s.residents.seats[0]
    pool.free_blocks(seat.blocks.blocks[0][4:])
    seat.blocks = s.kv_cache_manager.create_kv_cache_blocks((seat.blocks.blocks[0][:4],))
    seat.tokens = tuple(range(513))
    done(c, c.store(0, "A"))
    seat = s.residents.seats[0]
    seat.tokens = tuple(range(1000, 1129))
    seat.blocks = s.kv_cache_manager.create_kv_cache_blocks(
        (tuple(pool.get_new_blocks(1)),)
    )
    done(c, c.store(0, "B"))
    a = Request(all_token_ids=list(range(514)), cache_salt=None,
          skip_reading_prefix_cache=False)
    b = Request(all_token_ids=list(range(1000, 1130)), cache_salt=None,
          skip_reading_prefix_cache=False)
    cold = Request(all_token_ids=[9999], cache_salt=None,
             skip_reading_prefix_cache=False)
    s.waiting = FCFSRequestQueue([a, b, cold])
    s.skipped_waiting = FCFSRequestQueue()
    s.running = []
    s.max_num_running_reqs = 16
    policy = CachePolicy(s)
    with policy.runnable():
        assert list(s.waiting) == [cold]
        assert len(c.pending) == 1
    assert list(s.waiting) == [a, b, cold]
    n = next(iter(c.pending))
    c.receive(receipt(c, n, 0))
    with policy.runnable():
        assert list(s.waiting) == [cold]
        assert len(c.pending) == 1
    c.receive(receipt(c, n, 1))
    with policy.runnable():
        assert list(s.waiting) == [a, cold]
        assert len(c.pending) == 1
    done(c, next(iter(c.pending)))
    assert {id(r) for r in s.waiting} == {id(a), id(b), id(cold)}
    assert s.residents.offer(a.all_token_ids, None, 7).warm
    assert s.residents.offer(b.all_token_ids, None, 7).warm
    with policy.runnable():
        assert {id(r) for r in s.waiting} == {id(a), id(b), id(cold)}
        assert not c.pending


@pytest.mark.parametrize("incremental", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_automatic_restore_waits_for_native_execution_capacity(incremental, streaming):
    from vllm.v1.core.sched.request_queue import FCFSRequestQueue
    from betterscale.models.qwen35.cache_policy import CachePolicy

    class Request(S):
        __hash__ = object.__hash__
        __eq__ = object.__eq__

    s, c, _ = setup(4)
    if not incremental:
        c.pages = None
    done(c, c.store(0, "A"))
    a = Request(request_id="a", all_token_ids=list(range(514)),
                cache_salt=None, skip_reading_prefix_cache=False)
    busy = Request(request_id="busy")
    s.residents.requests["busy"] = 2
    s.residents.seats[2].owner = "busy"
    s.running = [] if streaming else [busy]
    s.num_waiting_for_streaming_input = int(streaming)
    s.max_num_running_reqs = 1
    s.waiting = FCFSRequestQueue([a])
    s.skipped_waiting = FCFSRequestQueue()
    policy = CachePolicy(s)
    with policy.runnable():
        assert not c.pending
        assert list(s.waiting) == [a]
    s.running = []
    s.num_waiting_for_streaming_input = 0
    with policy.runnable():
        assert len(c.pending) == 1
        assert not s.waiting
