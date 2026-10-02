import asyncio
import hashlib
import json
import pytest
from online_coordinator import Coordinator, IDENTITY
from session import Conflict


def fixture(tmp_path):
    c=Coordinator(tmp_path/"eviction.db","http://10.244.1.16:55581","http://10.244.2.32:55586")
    c.rank_private=True
    c.directory.create("s",IDENTITY,"P")
    data=json.dumps(dict(key="cp",tokens=[1,2],salt="s",pages=[],block_count=1,byte_length=100)).encode()
    manifest=hashlib.sha256(data).hexdigest()
    c.directory.publish(c.directory.claim("s","P",IDENTITY),manifest,"P")
    with c.directory.transaction() as db:
        db.execute("CREATE TABLE rank_manifests(id TEXT PRIMARY KEY,payload BLOB,p_owner INTEGER,d_owner INTEGER)")
        db.execute("INSERT INTO rank_manifests VALUES(?,?,?,?)",(manifest,data,2,3))
    c.p_affinity["s"]=2;c.d_affinity["s"]=3
    drops=[]
    async def prepare(kind,index,command):
        assert command==dict(kind="drop",key="cp")
        drops.append((kind,index))
        return len(drops)
    async def cache(kind,index,command):
        assert command["kind"]=="wait"
        return dict(operation=command["operation"],kind="drop",cancelled=False,ranks=[0,1])
    c.prepare=prepare;c.cache=cache
    return c,manifest,drops


def test_eviction_fences_next_turn_and_preserves_sticky_owners(tmp_path):
    async def run():
        c,manifest,drops=fixture(tmp_path)
        entered=asyncio.Event();release=asyncio.Event();original=c.cache
        async def cache(*args):
            entered.set();await release.wait()
            return await original(*args)
        c.cache=cache
        async def turn(session,prompt,n,future,generated,on_tokens):
            assert c.directory.current(session) is None
            assert drops==[("P",2),("D",3)]
            future.set_result("admitted");generated.set_result("admitted")
        c.turn=turn
        evict=asyncio.create_task(c.evict_idle("s"))
        await entered.wait()
        submit=asyncio.create_task(c.submit("s",[1,2],1))
        await asyncio.sleep(0)
        assert "s" not in c.inflight and not submit.done()
        release.set()
        assert await evict is True
        assert await submit=="admitted"
        assert c.p_affinity["s"]==2 and c.d_affinity["s"]==3
        await asyncio.sleep(0)
        assert await c.evict_idle("s") is False
        with c.directory.transaction() as db:
            assert db.execute("SELECT count(*) FROM rank_manifests").fetchone()[0]==0
        await c.close()
    asyncio.run(run())


def test_active_session_is_never_a_pressure_victim(tmp_path):
    async def run():
        c,manifest,drops=fixture(tmp_path)
        c.inflight["s"]=asyncio.get_running_loop().create_future()
        with pytest.raises(Conflict,match="active"):
            await c.evict_idle("s")
        assert c.directory.current("s")==manifest and not drops
        c.inflight.pop("s")
        c.directory.claim("s","P",IDENTITY)
        with pytest.raises(Conflict,match="idle"):
            await c.evict_idle("s")
        assert not drops
    asyncio.run(run())


@pytest.mark.parametrize("bad",[
    dict(operation=1,kind="drop",cancelled=False,ranks=[0]),
    dict(operation=1,kind="drop",cancelled=True,ranks=[0,1]),
    dict(operation=2,kind="drop",cancelled=False,ranks=[0,1]),
    dict(operation=1,kind="store",cancelled=False,ranks=[0,1]),
])
def test_incomplete_retirement_fails_closed_not_a_cold_miss(tmp_path,bad):
    async def run():
        c,manifest,drops=fixture(tmp_path)
        async def cache(*args):return bad
        c.cache=cache
        with pytest.raises(RuntimeError,match="quorum"):
            await c.evict_idle("s")
        assert c.failure and c.directory.current("s")==manifest
        assert drops==[("P",2)] and not c.evictions
        with c.directory.transaction() as db:
            assert db.execute("SELECT count(*) FROM rank_manifests").fetchone()[0]==1
        with pytest.raises(RuntimeError):
            await c.submit("s",[1,2],1)
    asyncio.run(run())


def test_directory_rejects_stale_retirement(tmp_path):
    c,manifest,_=fixture(tmp_path)
    with c.directory.transaction() as db:
        epoch=db.execute("SELECT epoch FROM sessions WHERE id='s'").fetchone()[0]
    newer=c.directory.claim("s","P",IDENTITY)
    with pytest.raises(Conflict):
        c.directory.forget_idle("s",epoch,manifest)
    assert c.directory.current("s")==manifest
    c.directory.publish(newer,"new","P")
    with pytest.raises(Conflict):
        c.directory.forget_idle("s",epoch,manifest)
    assert c.directory.current("s")=="new"


def test_eviction_started_while_ingress_waits_is_also_fenced(tmp_path):
    async def run():
        c,_,_=fixture(tmp_path)
        c.request_slots=asyncio.Semaphore(0)
        submit=asyncio.create_task(c.submit("s",[1,2],1))
        await asyncio.sleep(0)
        entered=asyncio.Event();release=asyncio.Event();original=c.cache
        async def cache(*args):
            entered.set();await release.wait();return await original(*args)
        c.cache=cache
        evict=asyncio.create_task(c.evict_idle("s"))
        await entered.wait()
        c.request_slots.release()
        await asyncio.sleep(0)
        assert "s" not in c.inflight
        async def turn(session,prompt,n,future,generated,on_tokens):
            assert c.directory.current(session) is None
            future.set_result(True);generated.set_result(True)
        c.turn=turn
        release.set()
        assert await evict and await submit
        await c.close()
    asyncio.run(run())


def test_oversized_host_admission_rejects_only_that_request(tmp_path):
    from online_host_cache import HostCapacityError
    from types import SimpleNamespace
    async def run():
        c,_,_=fixture(tmp_path)
        async def acquire(*args):raise HostCapacityError("test capacity")
        c.host_cache=SimpleNamespace(acquire=acquire,changed=asyncio.Event())
        with pytest.raises(HostCapacityError):
            await c.submit("s",[1,2],2)
        await asyncio.sleep(0)
        assert c.failure is None and not c.inflight and not c.host_waiters
        assert c.request_slots._value==128
        await c.close()
    asyncio.run(run())
