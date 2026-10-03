import asyncio
from types import SimpleNamespace
import pytest
from online_host_cache import HostCache,page_count
from betterscale.models.qwen35.cache_pages import page_keys


def make(limit=384):
    c=SimpleNamespace(p_affinity={},d_affinity={},inflight={},host_waiters=set(),failure=None)
    h=HostCache(c,{(k,i):limit for k in ("P","D") for i in range(4)},64,64)
    c.evicted=[]
    async def evict(session):
        assert not h.entries[session].active
        c.evicted.append(session);h.forgot(session);return True
    c.evict_idle=evict
    return c,h


def place(c,name,p=0,d=0):
    c.p_affinity[name]=p
    if d is not None:c.d_affinity[name]=d


def checkpoint(tokens):
    return dict(tokens=tokens,block_count=page_count(len(tokens)))


@pytest.mark.parametrize("base",[0,2,2048,2049,4097])
@pytest.mark.parametrize("extra",[0,1,2048])
@pytest.mark.parametrize("n",[1,2,4097])
def test_incremental_peak_covers_actual_three_generation_page_union(base,extra,n):
    from online_host_cache import Entry
    c,h=make(1<<30)
    old=[17]*base
    prompt=(old or [17,17])+[9]*extra
    entry=Entry((("P",0),),checkpoint(old),64+page_count(base)*64) if base else None
    for diverged in (False,True):
        current=list(prompt)
        if diverged:current[0]=22
        generations=[]
        if base:generations.append((old,"original","base"))
        salt="other" if base and current[:base]!=old else "original"
        p=current+[7];generations.append((p,salt,"P"))
        if n>1:generations.append((p+[8]*(n-1),salt,"D"))
        pages=set()
        for tokens,salt,key in generations:
            pages.update(page_keys(tokens,salt,2048,page_count(len(tokens)),key))
        actual=len(generations)*64+len(pages)*64
        assert h.peak(entry,current,n)>=actual


def test_idle_lru_reclaims_only_after_retirement_and_keeps_other_owners_free():
    async def run():
        c,h=make()
        for name in ("a","b"):
            place(c,name);await h.acquire(name,[1,2],2);h.finish(name,checkpoint([1,2,7,8]))
        entered=asyncio.Event();release=asyncio.Event();original=c.evict_idle
        async def evict(name):
            entered.set();await release.wait();return await original(name)
        c.evict_idle=evict
        place(c,"c")
        task=asyncio.create_task(h.acquire("c",[1,2],2))
        await entered.wait()
        assert h.used["P",0]==256 and not task.done()
        place(c,"independent",1,1)
        await h.acquire("independent",[1,2],2)
        assert h.entries["independent"].active
        release.set();await task
        assert c.evicted==["a"] and h.used["P",0]==384
        h.finish("c",checkpoint([1,2,7,8]))
        assert h.used["D",0]==256
    asyncio.run(run())


def test_active_reservation_waits_without_evicting_then_wakes_on_commit():
    async def run():
        c,h=make()
        for name in ("a","b"):place(c,name)
        await h.acquire("a",[1,2],2)
        waiting=asyncio.create_task(h.acquire("b",[1,2],2))
        await asyncio.sleep(0)
        assert not waiting.done() and not c.evicted
        h.finish("a",checkpoint([1,2,7,8]))
        await waiting
        assert h.used["P",0]==384 and not c.evicted
    asyncio.run(run())


def test_queued_old_session_can_lose_cache_but_not_execution_state():
    async def run():
        c,h=make(300)
        place(c,"a")
        await h.acquire("a",[1,2],2);h.finish("a",checkpoint([1,2,7,8]))
        c.inflight["a"]=object();c.host_waiters.add("a")
        await h.acquire("a",[1,2,7,8,9],2)
        assert c.evicted==["a"] and h.entries["a"].active
        assert h.used["P",0]==256
    asyncio.run(run())


def test_p_only_session_first_decode_adds_complete_d_charge():
    async def run():
        c,h=make(1024)
        place(c,"a",d=None)
        await h.acquire("a",[1,2],1);h.finish("a",checkpoint([1,2,7]))
        assert h.used["P",0]==128 and h.used["D",0]==0
        c.d_affinity["a"]=2
        await h.acquire("a",[1,2,7,9],2)
        assert h.used["D",2]==h.used["P",0]==384
        h.finish("a",checkpoint([1,2,7,9,7,8]))
        assert h.used["D",2]==128
    asyncio.run(run())


def test_oversize_and_unreserved_commit_are_not_silently_admitted():
    async def run():
        c,h=make(128);place(c,"a")
        with pytest.raises(ValueError,match="exceeds"):
            await h.acquire("a",[1,2],2)
        assert not h.entries and not any(h.used.values())
        await h.acquire("a",[1,2],1)
        with pytest.raises(RuntimeError,match="exceeded"):
            h.finish("a",checkpoint([1]*4097))
    asyncio.run(run())
