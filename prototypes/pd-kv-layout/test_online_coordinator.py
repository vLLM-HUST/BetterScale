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
