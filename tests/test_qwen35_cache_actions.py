"""CPU ownership and real native EngineCore maintenance-only wakeup gates."""

from collections import deque
import json
import queue
import socket
from threading import Event, Thread
from types import SimpleNamespace as S
import pytest
from betterscale.models.qwen35.cache_actions import CacheActions
from betterscale.models.qwen35.resident_leases import ResidentLeases, Offer


class Pool:
    def __init__(self):
        self.free = list(range(1, 9))

    def get_num_free_blocks(self):
        return len(self.free)

    def get_new_blocks(self, n):
        assert n <= len(self.free)
        return [S(block_id=self.free.pop(0)) for _ in range(n)]

    def free_blocks(self, group):
        self.free.extend(b.block_id for b in group)


def setup(ranks=1):
    pool = Pool()
    manager = S(block_pool=pool, create_kv_cache_blocks=lambda blocks: S(blocks=blocks))

    def release(blocks):
        pool.free_blocks(blocks.blocks[0])

    s = S(
        kv_cache_manager=manager,
        processed_step_seq=7,
        block_size=128,
        _release_resident_blocks=release,
    )
    s.residents = ResidentLeases(3, release_blocks=release)
    seat = s.residents.seats[0]
    seat.epoch, seat.tokens, seat.fence = 3, tuple(range(130)), 7
    seat.blocks = manager.create_kv_cache_blocks((tuple(pool.get_new_blocks(2)),))
    cache = CacheActions(s, ranks, 1 << 28, resident_bytes=64, block_bytes=128)
    s.cache_actions = cache
    cache.endpoint = "/test/unused"
    return s, cache


def receipt(cache, n, rank=0, **extra):
    command = cache.pending[n].command
    return dict(
        operation=n, seat=command["seat"], epoch=command["epoch"], rank=rank, **extra
    )


def test_async_writer_and_io_fences_are_separate():
    s, cache = setup()
    s.processed_step_seq = 6
    with pytest.raises(ValueError, match="writer-fence"):
        cache.store(0, "a")
    s.processed_step_seq = 7
    n = cache.store(0, "a")
    assert cache.take_commands()[0]["operation"] == n
    assert not cache.outbox and cache.pending and "a" not in cache.host
    assert s.residents.offer(list(range(131)), None, 7).seat != 0
    assert not s.residents.evict_hot(7)
    with pytest.raises(ValueError):
        s.residents.claim("bad", Offer(0, 3, True), 7)
    assert s.kv_cache_manager.block_pool.get_num_free_blocks() == 6
    cache.receive(receipt(cache, n))
    assert (
        "a" in cache.host and s.kv_cache_manager.block_pool.get_num_free_blocks() == 8
    )
    n = cache.load("a", 2)
    assert not s.residents.seats[2].tokens
    cache.receive(receipt(cache, n))
    assert s.residents.offer(list(range(131)), None, 7) == Offer(
        2, s.residents.seats[2].epoch, True
    )


def test_tp2_needs_both_distinct_rank_receipts():
    s, cache = setup(2)
    n = cache.store(0, "a")
    cache.receive(receipt(cache, n, 1))
    assert s.residents.seats[0].io_owner == n and "a" not in cache.host
    cache.receive(receipt(cache, n, 0))
    n = cache.load("a", 2)
    cache.receive(receipt(cache, n, 0))
    assert not s.residents.seats[2].tokens
    cache.receive(receipt(cache, n, 1))
    assert s.residents.seats[2].cursor == 129


@pytest.mark.parametrize("kind", ["duplicate", "epoch", "failure", "foreign"])
def test_invalid_rank_completion_never_recycles(kind):
    s, cache = setup(2)
    n = cache.store(0, "a")
    if kind == "duplicate":
        cache.receive(receipt(cache, n))
        value = receipt(cache, n)
    elif kind == "epoch":
        value = receipt(cache, n)
        value["epoch"] += 1
    elif kind == "failure":
        value = receipt(cache, n, error="DMA failed")
    else:
        value = receipt(cache, n, rank=3)
    with pytest.raises(RuntimeError):
        cache.receive(value)
    assert s.residents.seats[0].io_owner == n and s.residents.seats[0].blocks


def test_cancel_waits_for_all_ranks_then_drops_host_copy():
    s, cache = setup(2)
    n = cache.store(0, "a")
    cache.cancel(n)
    cache.receive(receipt(cache, n, 0))
    assert s.residents.seats[0].io_owner == n
    cache.receive(receipt(cache, n, 1))
    assert s.residents.seats[0].tokens and s.residents.seats[0].io_owner is None
    assert "a" not in cache.host
    drop = next(iter(cache.pending))
    assert cache.pending[drop].command["kind"] == "drop"
    for rank in (1, 0):
        cache.receive(receipt(cache, drop, rank))
    assert not cache.pending


def test_real_native_engine_idle_wait_wakes_without_model_wave(monkeypatch):
    from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
    from betterscale.models.qwen35.cache_engine import install

    for name in (
        "_process_engine_step",
        "has_work",
        "_handle_client_request",
        "shutdown",
        "state_cache",
        "_betterscale_cache_wakeup",
    ):
        monkeypatch.setattr(
            EngineCoreProc, name, getattr(EngineCoreProc, name, None), raising=False
        )
    monkeypatch.setattr(EngineCoreProc, "_betterscale_cache_wakeup", False)
    install()
    s, cache = setup()
    cache.endpoint = None
    s.has_requests = lambda: bool(cache.outbox)
    core = object.__new__(EngineCoreProc)
    core.scheduler, core.batch_queue, core.engines_running = s, deque(), False
    core.input_queue, core.output_queue, core.aborts_queue = (
        queue.Queue(),
        queue.Queue(),
        queue.Queue(),
    )
    core.process_input_queue_block = True
    core.shutdown_state = EngineShutdownState.RUNNING
    core._idle_state_callbacks = []
    core.post_step = lambda _: None
    core.step_fn = lambda: (cache.take_commands() and {} or {}, False)
    assert not core.has_work()
    try:
        n = core.state_cache(dict(kind="store", seat=0, key="a"))
        core._process_engine_step()
        assert not core.has_work() and cache.pending
        ready, returned = Event(), Event()

        def pump():
            ready.set()
            core._process_input_queue()
            returned.set()

        thread = Thread(target=pump, daemon=True)
        thread.start()
        assert ready.wait(1)
        assert not returned.wait(0.02)
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
            channel.sendto(json.dumps(receipt(cache, n)).encode(), cache.endpoint)
        assert returned.wait(2)
        thread.join(2)
        core._process_engine_step()
        assert "a" in cache.host and not cache.pending
    finally:
        core._state_cache_inbox.close()


def test_host_capacity_counts_inflight_and_cancelled_store_until_drop_ack():
    s, cache = setup(2)
    cache.host_bytes = 320
    n = cache.store(0, "A")
    assert cache.allocated_host_bytes == 320
    cache.cancel(n)
    for rank in (0, 1):
        cache.receive(receipt(cache, n, rank))
    assert cache.allocated_host_bytes == 320  # host drop is still queued
    with pytest.raises(ValueError, match="capacity"):
        cache.store(0, "B")
    drop = next(iter(cache.pending))
    with pytest.raises(ValueError, match="irreversible"):
        cache.cancel(drop)
    for rank in (0, 1):
        cache.receive(receipt(cache, drop, rank))
    assert cache.allocated_host_bytes == 0
    cache.store(0, "B")


def test_restore_capacity_refusal_leaves_victim_and_pool_unchanged():
    s, cache = setup()
    n = cache.store(0, "A")
    cache.receive(receipt(cache, n))
    s.kv_cache_manager.block_pool.get_new_blocks(8)
    before = cache.snapshot()
    with pytest.raises(RuntimeError, match="shared FA"):
        cache.load("A", 2)
    assert cache.snapshot() == before
