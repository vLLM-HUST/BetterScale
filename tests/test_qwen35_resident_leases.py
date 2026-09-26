import pytest
from betterscale.models.qwen35.resident_leases import ResidentLeases


def test_hot_seat_stays_put_and_unrelated_request_uses_empty_seat():
    releases = []
    pool = ResidentLeases(3, release_blocks=releases.append)
    first = pool.claim("A", pool.offer([1, 2], None, 0), 0)
    pool.retire("A", fence=1, tokens=[1, 2, 3], blocks="pages-A")
    second = pool.claim("B", pool.offer([8, 9], None, 1), 1)
    assert first.index == 0 and second.index == 1
    offer = pool.offer([1, 2, 3, 4], None, 1)
    assert offer.warm and offer.seat == 0
    resumed = pool.claim("C", offer, 1)
    assert resumed.epoch == first.epoch
    pool.transferred("C")
    assert releases == ["pages-A"]


def test_no_shorter_prefix_or_wrong_mtp_lookahead_or_salt_hit():
    pool = ResidentLeases(2, release_blocks=lambda blocks: None)
    pool.claim("A", pool.offer([1, 2], None, 0), 0)
    pool.retire("A", fence=2, tokens=[1, 2, 3], cache_salt="one", blocks="pages")
    for tokens, salt in [([1, 2], "one"), ([1, 2, 8], "one"), ([1, 2, 3], "two")]:
        assert not pool.offer(tokens, salt, 2).warm
    assert not pool.offer([1, 2, 3], "one", 1).warm
    assert pool.offer([1, 2, 3], "one", 2).warm


def test_preempted_seat_waits_for_writer_and_new_owner_gets_new_epoch():
    pool = ResidentLeases(1, release_blocks=lambda blocks: None)
    first = pool.offer([1, 2], None, 0)
    seat = pool.claim("A", first, 0)
    epoch = seat.epoch
    pool.retire("A", fence=3)
    assert pool.offer([8, 9], None, 2) is None
    with pytest.raises(ValueError, match="stale"):
        pool.claim("B", first, 3)
    next_seat = pool.claim("B", pool.offer([8, 9], None, 3), 3)
    assert next_seat.epoch == epoch + 1


def test_eviction_returns_pages_to_native_pool_once():
    releases = []
    pool = ResidentLeases(1, release_blocks=releases.append)
    pool.claim("A", pool.offer([1, 2], None, 0), 0)
    pool.retire("A", fence=0, tokens=[1, 2], blocks="pages")
    pool.claim("B", pool.offer([8, 9], None, 0), 0)
    assert releases == ["pages"]
    pool.transferred("B")
    assert releases == ["pages"]


def test_raw_frontier_keeps_post_eos_tokens_and_mtp_lookahead():
    from betterscale.models.qwen35.resident_leases import Frontier

    f = Frontier([1, 2, 3, 4])
    f.advance(2, 0, [])
    assert f.checkpoint() == (1, 2, 3)  # third prompt token is the lookahead
    f.advance(2, 0, [5])
    assert f.checkpoint() == (1, 2, 3, 4, 5)
    f.advance(
        3, 2, [6, 7, 8]
    )  # scheduler may stop at6; physical State must not roll back
    assert f.cursor == 7 and f.checkpoint() == tuple(range(1, 9))
    f.advance(3, 2, [9])  # all speculative tokens rejected; anchor still advances
    assert f.cursor == 8 and f.checkpoint() == tuple(range(1, 10))


def test_missing_frontier_evidence_fails_closed():
    from betterscale.models.qwen35.resident_leases import Frontier

    f = Frontier([1, 2])
    f.advance(2, 0, [])
    assert not f.known and f.checkpoint() == ()


def test_selected_victim_releases_capacity_before_new_allocation():
    releases = []
    pool = ResidentLeases(1, release_blocks=releases.append)
    pool.claim("A", pool.offer([1, 2], None, 0), 0)
    pool.retire("A", fence=0, tokens=[1, 2], blocks="pages")
    offer = pool.offer([8, 9], None, 0)
    refreshed = pool.discard_victim(offer, 0)
    assert releases == ["pages"]
    with pytest.raises(ValueError, match="stale"):
        pool.claim("B", offer, 0)
    pool.claim("B", refreshed, 0)
    assert releases == ["pages"]
