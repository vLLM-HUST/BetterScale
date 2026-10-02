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
