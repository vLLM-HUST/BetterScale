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


def test_native_engine_identity_is_fenced_and_path_survives_spawn(monkeypatch):
    import os
    import sys
    from rank_state_runtime import transfer_engine
    site="/workspace/pd-kv-layout-results/store-cpu-venv/lib/python3.12/site-packages"
    monkeypatch.setattr(sys,"path",list(sys.path))
    monkeypatch.setenv("PYTHONPATH","/pinned/runtime")
    cls=object()
    module=SimpleNamespace(__file__=site+"/mooncake/engine.so",TransferEngine=cls)
    monkeypatch.setitem(sys.modules,"mooncake",SimpleNamespace(engine=module))
    monkeypatch.setitem(sys.modules,"mooncake.engine",module)
    assert transfer_engine() is cls
    assert transfer_engine() is cls
    assert sys.path.count(site)==1 and sys.path[0]==site
    assert os.environ["PYTHONPATH"]==site+os.pathsep+"/pinned/runtime"
    module.__file__="/usr/local/lib/mooncake/engine.so"
    with pytest.raises(RuntimeError,match="unqualified Mooncake"):
        transfer_engine()


def test_close_releases_arena_only_after_all_pool_objects_retire():
    events = []
    pool = SimpleNamespace(lock=Lock(), groups={}, close=lambda: events.append("pool"))
    replica = SimpleNamespace(lock=Lock(), registered={}, quarantined=[])
    runtime = RankRuntime(pool, None, replica,
        SimpleNamespace(close=lambda: events.append("control")),
        SimpleNamespace(quarantined=[]),
        SimpleNamespace(close=lambda: events.append("arena")))
    runtime.close()
    assert events == ["control", "pool", "arena"]


def test_startup_reservation_is_owned_before_model_load_and_not_repeated(monkeypatch):
    import json,sys,threading
    import state_numa,rank_pinned_arena
    from rank_state_runtime import prepare_host_arena
    events=[];main=threading.get_ident()
    monkeypatch.setenv("BETTERSCALE_PD_STATE_NUMA",json.dumps(MAPPING))
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES","4,5")
    monkeypatch.setenv("BETTERSCALE_PD_PINNED_ARENA","1")
    monkeypatch.setenv("BETTERSCALE_PD_COMPRESS_RESIDENT","0")
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(
        npu=SimpleNamespace(set_device=lambda device:events.append("device"))))
    def bind(physical,node):
        assert threading.get_ident()!=main
        assert (physical,node)==(5,2)
        events.append("numa")
    monkeypatch.setattr(state_numa,"bind_thread",bind)
    def reserve(budget,node):
        assert (budget,node)==(4<<30,2)
        events.append("reserved")
        return SimpleNamespace(stats=lambda:dict(reserved_bytes=budget))
    monkeypatch.setattr(rank_pinned_arena,"RankPinnedArena",reserve)
    runner=SimpleNamespace(device=SimpleNamespace(index=1))
    config=dict(pd_rank_role="P",pd_rank_instance=2,pd_rank_dp=0,state_cache_host_bytes=4<<30)
    prepare_host_arena(runner,1,config)
    assert events==["numa","device","reserved"]
    assert runner._pd_reserved_arena.stats()["reserved_bytes"]==4<<30
    with pytest.raises(RuntimeError,match="already"):
        prepare_host_arena(runner,1,config)
