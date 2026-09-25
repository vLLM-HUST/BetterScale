"""Request leases must not accidentally free, alias or rewind hot model State."""

import pytest
from betterscale.live.llm.qwen35 import Capacity
from betterscale.live.llm.qwen35.residents import ResidentTable


def test_hot_hit_empty_first_and_explicit_eviction():
    cleared = []
    table = ResidentTable(
        Capacity(2, 3, page_tokens=4, token_pages=4), clear_seat=cleared.append
    )
    a, hit = table.acquire([1, 2])
    assert (a.seat, hit) == (0, 0)
    assert table.reserve(a, 2) == [0, 1]
    table.commit(a, [1, 2])
    table.finish(a)
    b, hit = table.acquire([9])
    assert (b.seat, hit) == (1, 0)
    assert table.reserve(b, 1) == [4]
    table.commit(b, [9])
    table.finish(b)
    c, hit = table.acquire([1, 2, 3])
    assert (c.seat, hit) == (0, 2)
    assert table.reserve(c, 3) == [0, 1, 2]
    table.commit(c, [3])
    table.finish(c)
    # An earlier recurrent boundary is NOT a prefix hit on seat0.
    d, hit = table.acquire([1])
    assert (d.seat, hit) == (2, 0)
    table.reserve(d, 1)
    table.commit(d, [1])
    table.finish(d)
    e, hit = table.acquire([8])
    assert (e.seat, hit) == (1, 0)
    assert table.seats[0].tokens == [1, 2, 3]
    assert cleared == [0, 1, 2, 1]
    with pytest.raises(ValueError, match="stale"):
        table.reserve(b, 1)


def test_shared_capacity_not_resident_max_context_and_atomic_refusal():
    table = ResidentTable(
        Capacity(2, 20, page_tokens=4, token_pages=2), clear_seat=lambda _: None
    )
    a, _ = table.acquire([1])
    b, _ = table.acquire([2])
    assert table.reserve(a, 4) == [0, 1, 2, 3]
    assert table.reserve(b, 4) == [4, 5, 6, 7]
    with pytest.raises(RuntimeError, match="page capacity"):
        table.reserve(a, 5)
    assert table.seats[0].pages == [0] and table.seats[1].pages == [1]
    with pytest.raises(RuntimeError, match="execution capacity"):
        table.acquire([3])
    with pytest.raises(RuntimeError, match="live request"):
        table.evict(a.seat)


def test_finish_trims_only_uncommitted_lookahead_pages_and_retires_lease():
    table = ResidentTable(
        Capacity(1, 2, page_tokens=4, token_pages=3), clear_seat=lambda _: None
    )
    a, _ = table.acquire([1])
    table.reserve(a, 9)
    table.commit(a, [1])
    table.finish(a)
    assert table.seats[0].pages == [0] and sorted(table.free_pages) == [1, 2]
    with pytest.raises(ValueError, match="stale"):
        table.finish(a)
    other = ResidentTable(table.capacity, clear_seat=lambda _: None)
    with pytest.raises(ValueError, match="generation"):
        other.reserve(a, 1)


def test_failed_eviction_cannot_publish_old_identity():
    table = ResidentTable(Capacity(1, 1, token_pages=1), clear_seat=lambda _: None)
    a, _ = table.acquire([1])
    table.reserve(a, 1)
    table.commit(a, [1])
    table.finish(a)

    def failed_clear(_):
        raise RuntimeError("clear failed")

    table.clear_seat = failed_clear
    with pytest.raises(RuntimeError, match="clear failed"):
        table.acquire([2])
    assert table.seats[0].tokens == []
    assert table.seats[0].incarnation != a.incarnation


def test_page_pressure_evicts_idle_state_even_when_an_empty_seat_exists():
    cleared = []
    table = ResidentTable(
        Capacity(1, 3, page_tokens=4, token_pages=1), clear_seat=cleared.append
    )
    a, _ = table.acquire([1])
    table.reserve(a, 4)
    table.commit(a, [1, 2, 3, 4])
    table.finish(a)
    b, _ = table.acquire([8])
    assert b.seat == 1
    assert table.reserve(b, 1) == [0]
    assert table.seats[0].tokens == [] and table.seats[0].pages == []
    assert cleared == [0, 1, 0]
