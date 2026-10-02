from types import SimpleNamespace
from threading import Lock
import pytest
from rank_state_runtime import owner_for, RankRuntime

MAPPING = [6, 0, 7, 1, 4, 2, 5, 3]


@pytest.mark.parametrize("group", range(4))
@pytest.mark.parametrize("rank", range(2))
def test_physical_owner_is_same_for_independent_p_and_global_d(group, rank):
    p = dict(pd_rank_role="P", pd_rank_instance=group, pd_rank_dp=0)
    d = dict(pd_rank_role="D", pd_rank_instance=0, pd_rank_dp=group)
    physical = 2 * group + rank
    assert owner_for(p, rank, rank, MAPPING, f"{2*group},{2*group+1}") == (
        ("P", group, rank), physical, MAPPING[physical])
    assert owner_for(d, rank, physical, MAPPING, "0,1,2,3,4,5,6,7") == (
        ("D", group, rank), physical, MAPPING[physical])


def test_wrong_visibility_cannot_silently_cross_wire_rank_shards():
    config = dict(pd_rank_role="P", pd_rank_instance=1, pd_rank_dp=0)
    with pytest.raises(ValueError, match="physical placement"):
        owner_for(config, 0, 0, MAPPING, "0,1")


def test_release_keeps_route_if_incoming_state_is_still_pending():
    routes = {"cp": 2}
    replica = SimpleNamespace(routes=routes, lock=Lock())
    def pending(key):
        raise RuntimeError("pending")
    receiver = SimpleNamespace(drop=pending)
    runtime = RankRuntime(None, receiver, replica, None, None)
    with pytest.raises(RuntimeError, match="pending"):
        runtime.release("cp")
    assert routes == {"cp": 2}
    receiver.drop = lambda key: None
    runtime.release("cp")
    runtime.release("cp")
    assert not routes


def test_close_refuses_referenced_checkpoints_before_stopping_listener():
    events = []
    pool = SimpleNamespace(lock=Lock(), groups={"cp": ("object",)}, close=lambda: events.append("pool"))
    replica = SimpleNamespace(lock=Lock(), registered={}, quarantined=[])
    runtime = RankRuntime(pool, None, replica,
        SimpleNamespace(close=lambda: events.append("control")), SimpleNamespace(quarantined=[]))
    with pytest.raises(RuntimeError, match="not retired"):
        runtime.close()
    assert not events
    pool.groups.clear()
    runtime.close()
    assert events == ["control", "pool"]
