"""CPU protocol tests with deliberately delayed worker receipts, no NPU init."""

from concurrent.futures import Future
from types import SimpleNamespace
import pytest
import torch

from maintenance import CacheMaintenance
from betterscale.live.llm.qwen35.residents import ResidentTable
from betterscale.live.runtime.host_state import (
    HostStateKey,
    HostStateDomainSelection,
    HostStateSelection,
    TorchHostStateBackend,
)


class Worker:
    def __init__(self):
        self.actions = {}
        self.released = []

    def start(self, action, pages):
        future = Future()
        self.actions[action.number] = (action, future)
        return future

    def complete(self, n, receipt=None):
        action, future = self.actions[n]
        future.set_result(receipt or (n, action.seat, action.epoch))

    def release(self, key):
        self.released.append(key)

    def close(self):
        pass


def setup():
    table = ResidentTable(
        SimpleNamespace(
            token_pages=4, resident_seats=3, execution_seats=2, page_tokens=4
        ),
        clear_seat=lambda _: None,
    )
    lease, _ = table.acquire([1, 2, 3], "salt")
    table.reserve(lease, 3)
    table.commit(lease, [1, 2, 3])
    table.finish(lease)
    worker = Worker()
    cache = CacheMaintenance(table, worker)
    return table, worker, cache


def test_store_and_load_only_publish_after_receipt():
    table, worker, cache = setup()
    key = HostStateKey("A", 1)
    pages = tuple(table.seats[0].pages)
    n = cache.store(0, key)
    assert 0 not in table.idle_indices and key not in cache.host
    with pytest.raises(RuntimeError):
        table.evict(0)
    lease, cached = table.acquire([1, 2, 3, 4], "salt")
    assert lease.seat != 0 and cached == 0
    cache.reap()
    assert tuple(table.seats[0].pages) == pages
    worker.complete(n)
    cache.wait()
    cache.reap()
    assert not table.seats[0].tokens and pages[0] in table.free_pages
    n = cache.load(key, 2)
    assert not table.seats[2].tokens and 2 not in table.idle_indices
    with pytest.raises(RuntimeError):
        cache.drop(key)
    worker.complete(n)
    cache.reap()
    assert table.seats[2].tokens == [1, 2, 3]
    assert table.seats[2].cache_salt == "salt"
    cache.close()


@pytest.mark.parametrize("kind", ["store", "load"])
def test_cancel_retains_pins_until_io_finishes(kind):
    table, worker, cache = setup()
    key = HostStateKey("A", 1)
    n = cache.store(0, key)
    if kind == "load":
        worker.complete(n)
        cache.reap()
        n = cache.load(key, 0)
    cache.cancel(n)
    cache.reap()
    assert table.seats[0].io_owner == n
    worker.complete(n)
    cache.reap()
    assert table.seats[0].io_owner is None
    assert bool(table.seats[0].tokens) == (kind == "store")
    cache.close()


def test_stale_or_failed_receipts_quarantine_not_free():
    table, worker, cache = setup()
    n = cache.store(0, HostStateKey("A", 1))
    worker.complete(n, (n, 0, 999))
    with pytest.raises(RuntimeError, match="stale cache receipt"):
        cache.reap()
    assert table.seats[0].io_owner == n and table.seats[0].pages


def test_pinned_pages_not_reclaimable_and_no_execution_slot_charge():
    table, worker, cache = setup()
    n = cache.store(0, HostStateKey("A", 1))
    a, _ = table.acquire([4])
    b, _ = table.acquire([5])
    assert not table.can_reserve(a, 16)
    assert table.can_reserve(b, 12)
    worker.complete(n)
    cache.reap()
    assert table.can_reserve(a, 16)
    cache.close()


def test_restore_no_capacity_is_transactional():
    table, worker, cache = setup()
    key = HostStateKey("A", 1)
    n = cache.store(0, key)
    worker.complete(n)
    cache.reap()
    lease, _ = table.acquire([8])
    table.reserve(lease, 16)
    before = (table.seats[2].incarnation, list(table.free_pages))
    with pytest.raises(RuntimeError, match="shared pages"):
        cache.load(key, 2)
    assert before == (table.seats[2].incarnation, table.free_pages)
    cache.close()


def test_host_backend_copies_span_offsets_and_reserves_inflight_budget():
    tensor = torch.arange(26, dtype=torch.float32).reshape(13, 2)
    domain = object()
    state = SimpleNamespace(
        tensor=tensor,
        domain=domain,
        storage_dtype=tensor.dtype,
        block_shape=(2,),
        num_blocks=4,
        leading_physical_blocks=1,
        physical_blocks_per_logical_block=3,
        logical_block_bytes=24,
    )
    selection = lambda *ids: HostStateSelection(
        (HostStateDomainSelection(domain, ids),)
    )
    backend = TorchHostStateBackend(memory_budget_bytes=48)
    key = HostStateKey("A", 1)
    original = tensor.clone()
    first = backend.offload([("lane", state)], key, selection(2, 0), stream=None)
    # Completion not consumed yet: those bytes still count against capacity.
    with pytest.raises(Exception, match="host budget exhausted"):
        backend.offload(
            [("lane", state)], HostStateKey("B", 1), selection(1), stream=None
        )
    first.result()
    restored = backend.restore([("lane", state)], key, selection(3, 1), stream=None)
    with pytest.raises(Exception):
        backend.release(key)
    restored.result()
    assert torch.equal(tensor[10:13], original[7:10])
    assert torch.equal(tensor[4:7], original[1:4])
    assert torch.equal(tensor[0], original[0])
    backend.release(key)
    assert backend.committed_bytes == 0


def test_scheduler_reaps_without_compute_and_skips_loading_prefix():
    from contextlib import nullcontext
    from betterscale.live.llm.qwen35.scheduler import Scheduler

    table, worker, cache = setup()
    root = SimpleNamespace(
        residents_table=table,
        context_tokens=32,
        _live_lock=nullcontext(),
        _require_active=lambda _: None,
    )
    scheduler = Scheduler(root, maintenance=cache)
    key = HostStateKey("A", 1)
    n = cache.store(0, key)
    assert scheduler.busy
    worker.complete(n)
    assert scheduler.tick() == {} and not scheduler.busy
    n = cache.load(key, 1)
    request = scheduler.submit("resume", [1, 2, 3, 4], 2, cache_salt="salt")
    assert scheduler.tick() == {} and request.lease is None
    # Cancel the waiting request, then process the load with no model waves.
    scheduler.cancel("resume")
    worker.complete(n)
    scheduler.tick()
    assert table.seats[1].tokens == [1, 2, 3] and not scheduler.busy
    scheduler.close()


def test_worker_failure_keeps_ownership_and_submission_failure_rolls_back():
    table, worker, cache = setup()

    def fail(*args):
        raise RuntimeError("submission drained")

    original = worker.start
    worker.start = fail
    with pytest.raises(RuntimeError, match="submission drained"):
        cache.store(0, HostStateKey("A", 1))
    assert table.seats[0].io_owner is None and not cache.pending
    worker.start = original
    n = cache.store(0, HostStateKey("A", 1))
    worker.actions[n][1].set_exception(RuntimeError("unknown DMA outcome"))
    with pytest.raises(RuntimeError, match="unknown DMA"):
        cache.reap()
    assert table.seats[0].io_owner == n and 0 not in table.idle_indices


@pytest.mark.parametrize("drain_fails", [False, True])
def test_partial_copy_failure_drains_only_copy_stream_or_quarantines(
    monkeypatch, drain_fails
):
    from contextlib import nullcontext

    backend = TorchHostStateBackend(memory_budget_bytes=64)

    class Stream:
        def __init__(self):
            self.drained = 0

        def synchronize(self):
            self.drained += 1
            if drain_fails:
                raise RuntimeError("drain failed")

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(Stream=Stream, stream=lambda _: nullcontext()),
        raising=False,
    )

    def partial(*args, **kwargs):
        raise RuntimeError("partial submission")

    monkeypatch.setattr(backend, "_enqueue_copies", partial)
    stream = Stream()
    lane = SimpleNamespace(tensor=SimpleNamespace(device=SimpleNamespace(type="npu")))
    payloads = {"owned": object()}
    with pytest.raises(
        RuntimeError, match="drain failed" if drain_fails else "partial submission"
    ):
        backend._copy_lanes(
            (("lane", lane, (0,)),), payloads, to_host=True, stream=stream
        )
    assert stream.drained == 1
    assert bool(backend.quarantined) == drain_fails
    if drain_fails:
        assert backend.quarantined[0][1] is payloads


def test_scheduler_loading_request_does_not_head_of_line_block_others(monkeypatch):
    from test_live_qwen_scheduler import BatchRoot
    from betterscale.live.llm.qwen35.scheduler import Scheduler

    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(synchronize=lambda _: None), raising=False
    )
    root = BatchRoot()
    table = root.residents_table
    lease, _ = table.acquire([1, 2, 3], "salt")
    table.reserve(lease, 3)
    table.commit(lease, [1, 2, 3])
    table.finish(lease)
    worker = Worker()
    cache = CacheMaintenance(table, worker)
    key = HostStateKey("A", 1)
    n = cache.store(0, key)
    worker.complete(n)
    cache.reap()
    n = cache.load(key, 0)
    scheduler = Scheduler(root, maintenance=cache)
    blocked = scheduler.submit("blocked", [1, 2, 3, 4], 2, cache_salt="salt")
    other = scheduler.submit("other", [20, 21, 22], 2)
    scheduler.tick()
    assert blocked.lease is None and other.lease.seat == 1
    scheduler.cancel("blocked")
    worker.complete(n)
    scheduler.close()


def test_scheduler_page_pressure_waits_for_store_without_recompute_preemption(
    monkeypatch,
):
    from test_live_qwen_scheduler import BatchRoot
    from betterscale.live.llm.qwen35.scheduler import Scheduler

    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(synchronize=lambda _: None), raising=False
    )
    root = BatchRoot(pages=4)
    table = root.residents_table
    lease, _ = table.acquire([1, 2, 3])
    table.reserve(lease, 3)
    table.commit(lease, [1, 2, 3])
    table.finish(lease)
    worker = Worker()
    cache = CacheMaintenance(table, worker)
    n = cache.store(0, HostStateKey("A", 1))
    scheduler = Scheduler(root, maintenance=cache)
    scheduler.submit("waiting-for-pages", list(range(20, 33)), 2)
    # Incremental prefill can use the three free pages first. Stop precisely
    # when its next PageRequest is waiting for the offloaded fourth page.
    for _ in range(50):
        before = len(root.batches)
        scheduler.tick()
        if before == len(root.batches):
            break
    assert scheduler.active and before == len(root.batches)
    assert scheduler.stats["preemptions"] == 0
    worker.complete(n)
    for _ in range(50):
        scheduler.tick()
        if not scheduler.busy:
            break
    assert not scheduler.busy and scheduler.stats["preemptions"] == 0
    scheduler.close()
