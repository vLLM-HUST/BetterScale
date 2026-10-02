import asyncio
import pytest
from online_coordinator import Admission
from online_node import validate


def test_long_context_admission_is_pages_not_just_seats():
    async def run():
        gate=Admission(1044,16)
        held=[await gate.acquire(262144) for _ in range(8)]
        assert held==[129]*8 and gate.free==12 and gate.slots==8
        ninth=asyncio.create_task(gate.acquire(262144))
        await asyncio.sleep(0)
        assert not ninth.done()
        await gate.release(held.pop())
        held.append(await asyncio.wait_for(ninth,1))
        for n in held:await gate.release(n)
        assert (gate.free,gate.slots)==(1044,16)
    asyncio.run(run())


def test_only_narrow_online_operations_admitted():
    body=dict(instance=0,op="cache",args=dict(owner=3,command=dict(kind="snapshot")))
    assert validate("D",body)[1]=="cache"
    with pytest.raises(ValueError):validate("P",body)
    body["args"]["command"]={"kind":"arbitrary_worker_method"}
    with pytest.raises(ValueError):validate("D",body)
    for bad in (True,-1,4):
        body["args"]=dict(owner=bad,command=dict(kind="snapshot"))
        with pytest.raises(ValueError):validate("D",body)


def test_prefill_affinity_reuses_ready_permit_but_never_waits_for_busy_owner(tmp_path):
    from online_coordinator import Coordinator
    async def run():
        c=Coordinator(tmp_path/"directory.db","http://10.244.1.16:55581","http://10.244.2.32:55586",p_per_instance=1)
        c.p_affinity["warm"]=2
        assert await c.acquire_p("warm")==2
        assert await c.acquire_p("another")==0
        c.p_affinity["busy"]=2
        assert await asyncio.wait_for(c.acquire_p("busy"),.1)==1
        assert c.p_available.qsize()==1
    asyncio.run(run())


def test_profile_control_is_bounded():
    from online_node import validate
    import pytest
    assert validate("D",dict(instance=0,op="profile",args=dict(owner=0,start=True)))[1]=="profile"
    for args in (dict(owner=1,start=True),dict(owner=0,start=1),dict(owner=0,start=True,path="/tmp/arbitrary")):
        with pytest.raises(ValueError):validate("D",dict(instance=0,op="profile",args=args))


@pytest.mark.parametrize("d_version,accepted",[("zstd-resident-v1",True),("raw-v2",False)])
def test_wire_version_mismatch_fails_before_admission(tmp_path,monkeypatch,d_version,accepted):
    import online_coordinator as module
    from pd_limits import context_limit
    class Peer:
        def __init__(self,url,*args):self.kind="P" if "1.16" in url else "D"
        async def health(self):
            kind=self.kind
            info=dict(state_wire="zstd-resident-v1" if kind=="P" else d_version,
                capacities=[dict(block_size=2048,max_requests=16,free_blocks=1044)]*4)
            return dict(ready=True,kind=kind,context_limit=context_limit(),
                actors=[dict(alive=True,quarantined=False,info=info)]*(4 if kind=="P" else 1))
    monkeypatch.setattr(module,"Peer",Peer)
    async def run():
        c=module.Coordinator(tmp_path/"directory.db","http://10.244.1.16:55581","http://10.244.2.32:55586")
        try:
            if accepted:assert await c.start() is c
            else:
                with pytest.raises(RuntimeError,match="wire version"):await c.start()
        finally:await c.close()
    asyncio.run(run())


def test_audit_control_is_boolean_and_actor_wide():
    assert validate("D",dict(instance=0,op="audit",args=dict(owner=0,enabled=False)))[1]=="audit"
    for args in (dict(owner=1,enabled=False),dict(owner=0,enabled=0),dict(owner=0,enabled=True,extra=1)):
        with pytest.raises(ValueError):validate("D",dict(instance=0,op="audit",args=args))


def test_worker_audit_mode_changes_only_without_inflight_io():
    import ast
    from pathlib import Path
    from types import SimpleNamespace as NS
    tree=ast.parse(Path(__file__).with_name("online_entry.py").read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="Worker")
    fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=="pd_object_audit")
    namespace={};exec(compile(ast.Module(body=[fn],type_ignores=[]),"audit-method","exec"),namespace)
    method=namespace["pd_object_audit"]
    cache=NS(inflight={1},rank=1,page_backend=NS(verify=True))
    worker=NS(model_runner=NS(_state_cache_worker=cache),
        vllm_config=NS(parallel_config=NS(data_parallel_rank=2)))
    with pytest.raises(RuntimeError):method(worker,False)
    assert cache.page_backend.verify is True
    cache.inflight.clear()
    assert method(worker,False)==dict(owner=2,rank=1,previous=True,enabled=False)
    assert cache.page_backend.verify is False
    with pytest.raises(ValueError):method(worker,1)
