"""Controlled protocol/pressure evidence, not NPU or model equivalence evidence."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from test_live_qwen_generation import ProtocolRoot

from betterscale.live.llm.qwen35 import Capacity
from betterscale.live.llm.qwen35.residents import ResidentTable
from betterscale.live.llm.qwen35.scheduler import Scheduler


class BatchRoot(ProtocolRoot):
    context_tokens = 32
    _live_lock = nullcontext()

    def __init__(self, pages=8, bad_draft=False):
        super().__init__(bad_draft)
        self.residents_table = ResidentTable(
            Capacity(2, 2, page_tokens=4, token_pages=pages), clear_seat=self.clear_seat
        )
        self.batches = []

    def _require_active(self, _):
        pass

    def batch_step(self, steps):
        self.batches.append((steps[0].kind, len(steps)))
        return [self.result(s.tokens, draft=s.kind == "draft") for s in steps]


@pytest.fixture(autouse=True)
def no_npu():
    with patch.object(
        torch, "npu", SimpleNamespace(synchronize=lambda _: None), create=True
    ):
        yield


def drain(scheduler):
    results = {}
    for _ in range(500):
        if not scheduler.busy:
            return results
        results.update(scheduler.tick())
    raise AssertionError("scheduler made no bounded progress")


@pytest.mark.parametrize("bad_draft", [False, True])
def test_real_grouping_turnover_and_hot_resume(bad_draft):
    root = BatchRoot(bad_draft=bad_draft)
    scheduler = Scheduler(root)
    scheduler.submit("a", [1, 2, 3], 6)
    scheduler.submit("b", [20, 21], 3)
    results = drain(scheduler)
    assert results["a"]["token_ids"] == [4, 5, 6, 7, 8, 9]
    assert results["b"]["token_ids"] == [22, 23, 24]
    assert scheduler.stats["max_batch"] == scheduler.stats["max_active"] == 2
    scheduler.submit("c", list(range(1, 10)), 2)
    c = drain(scheduler)["c"]
    assert c["cached_tokens"] == 9 and c["seat"] == results["a"]["seat"]
    assert c["token_ids"] == [10, 11]


def test_pressure_discards_entire_seat_and_recomputes_without_duplicate_output():
    root = BatchRoot(pages=4)
    scheduler = Scheduler(root)
    scheduler.submit("a", [1, 2, 3], 10)
    scheduler.submit("b", [20, 21, 22], 10)
    results = drain(scheduler)
    assert results["a"]["token_ids"] == list(range(4, 14))
    assert results["b"]["token_ids"] == list(range(23, 33))
    assert results["b"]["preemptions"] > 0
    assert scheduler.stats["preemptions"] > 0
    assert scheduler.stats["preemptions_with_output"] > 0
    table = root.residents_table
    assert all(r.request is None for r in table.seats)
    assert sum(len(r.pages) for r in table.seats) + len(table.free_pages) == 4


def test_cancel_after_speculative_writer_invalidates_lease_and_all_pages():
    root = BatchRoot()
    scheduler = Scheduler(root)
    scheduler.submit("a", [1, 2, 3], 8)
    for _ in range(9):
        scheduler.tick()
    request = scheduler.active["a"]
    lease = request.lease
    scheduler.cancel("a")
    resident = root.residents_table.seats[lease.seat]
    assert resident.tokens == [] and resident.pages == [] and resident.request is None
    with pytest.raises(ValueError, match="stale"):
        root.residents_table.reserve(lease, 1)
    assert not scheduler.busy


def test_arrival_after_execution_started_and_queued_cancellation():
    root = BatchRoot()
    scheduler = Scheduler(root)
    scheduler.submit("a", [1], 12)
    scheduler.tick()
    scheduler.submit("b", [20], 3)
    scheduler.submit("c", [30], 3)
    scheduler.cancel("c")
    results = drain(scheduler)
    assert set(results) == {"a", "b"}
    assert scheduler.stats["max_active"] == 2
    assert scheduler.stats["cancelled"] == 1


def test_async_ingress_dispatches_same_waves_to_a_mirror_and_accepts_concurrency():
    import asyncio

    from betterscale.live.llm.qwen35.ingress import Ingress

    async def run():
        root, mirror_root = BatchRoot(), BatchRoot()
        mirror = Scheduler(mirror_root)
        remote_results = {}

        def broadcast(commands):
            for command in commands:
                if command[0] == "submit":
                    mirror.submit(*command[1:])
                else:
                    mirror.cancel(command[1])
            remote_results.update(mirror.tick())

        ingress = Ingress(root, broadcast)
        async with ingress.lifespan(None):
            actual = await asyncio.gather(
                ingress.execute([1, 2], 5, ()), ingress.execute([20, 21], 5, ())
            )
        assert actual == [remote_results["1"], remote_results["2"]]
        assert ingress.scheduler.stats["max_batch"] == 2
        assert ingress.scheduler.stats == mirror.stats
        mirror.close()

    asyncio.run(run())


def test_c16_r20_uses_real_batches_without_per_seat_context_reservation():
    root = BatchRoot()
    for value in vars(root.continuation).values():
        value.tensor = torch.zeros(
            (20, *value.tensor.shape[1:]), dtype=value.tensor.dtype
        )
    root.residents_table = ResidentTable(
        Capacity(16, 20, page_tokens=4, token_pages=64), clear_seat=root.clear_seat
    )
    scheduler = Scheduler(root)
    for i in range(16):
        scheduler.submit(str(i), [i + 1, i + 2], 6)
    results = drain(scheduler)
    assert scheduler.stats["max_batch"] == scheduler.stats["max_active"] == 16
    for i in range(16):
        assert results[str(i)]["token_ids"] == list(range(i + 3, i + 9))
    assert scheduler.stats["preemptions"] == 0
    # Only the committed 8-token histories remain, rather than 20*max_context.
    assert sum(len(r.pages) for r in root.residents_table.seats) == 32
    assert sum(bool(r.tokens) for r in root.residents_table.seats) == 16


def test_staggered_target_and_draft_phases_coalesce_instead_of_locking_apart():
    root = BatchRoot()
    scheduler = Scheduler(root)
    scheduler.submit("early", [1, 2, 3, 4, 5], 8)
    scheduler.tick()  # target1 completes before the second request arrives
    scheduler.submit("later", [20, 21, 22, 23, 24], 8)
    scheduler.tick()
    scheduler.tick()
    assert scheduler.stats["max_batch"] == 2  # Before either prefill finishes.
    results = drain(scheduler)
    assert scheduler.stats["max_batch"] == 2
    assert results["early"]["token_ids"] == list(range(6, 14))
    assert results["later"]["token_ids"] == list(range(25, 33))
    # Both target and shifted draft phases actually share batches.
    assert ("target", 2) in root.batches and ("draft", 2) in root.batches
