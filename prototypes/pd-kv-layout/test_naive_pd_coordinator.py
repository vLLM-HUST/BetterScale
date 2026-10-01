"""CPU ownership/failure tests; fake engines do not qualify model numerics."""
import asyncio
import hashlib
from pathlib import Path

import pytest
import naive_pd_coordinator as pd


class FakePeer:
    def __init__(self, kind):
        self.kind=kind;self.cache={};self.resident={};self.calls=[]
        self.fail=False;self.block=None

    async def get(self,key):
        if key not in self.cache:raise pd.RemoteError(404,"miss")
        return self.cache[key]

    async def put(self,key,data):
        assert hashlib.sha256(data).hexdigest()==key
        self.cache[key]=data

    async def rpc(self,instance,op,**args):
        await asyncio.sleep(0)
        self.calls.append((instance,op,args))
        if op=="generate_batch":
            if self.block:await self.block.wait()
            if self.fail:raise RuntimeError("injected uncertain generation")
            results=[]
            for item in args["items"]:
                slot=(instance,item["owner"],item["salt"])
                old=self.resident.get(slot,[])
                assert not old or item["tokens"][:len(old)]==old
                ids=[17]*item["n"]
                self.resident[slot]=item["tokens"]+ids
                results.append(dict(token_ids=ids,cached=max(0,len(old)-1)))
            return results
        slot=(instance,args["owner"],args["salt"])
        if op=="import":
            self.resident[slot]=pd.unpack(await self.get(args["key"]))["header"]["tokens"]
        elif op=="drop":self.resident.pop(slot,None)
        elif op=="export":
            assert self.resident[slot]==args["tokens"]
            data=pd.pack(dict(header=dict(identity=pd.IDENTITY,tokens=args["tokens"],
                                          cursor=len(args["tokens"])-1),shards=[{},{}]))
            key=hashlib.sha256(data).hexdigest();await self.put(key,data)
            return dict(key=key,bytes=len(data),cursor=len(args["tokens"])-1)


async def start(path):
    c=pd.Coordinator(path,"unusedP","unusedD",verify_imports=True)
    c.p=FakePeer("P");c.d=FakePeer("D")
    c.tasks=[asyncio.create_task(c._prefill(i)) for i in range(4)]
    c.tasks.append(asyncio.create_task(c._decode()))
    return c


def test_roundtrip_owners_cache_and_single_token(tmp_path):
    async def scenario():
        c=await start(tmp_path/"directory.db")
        try:
            names=[f"s{i}" for i in range(8)]
            first=await asyncio.gather(*(c.submit(s,[1,2,3],8) for s in names))
            second=await asyncio.gather(*(c.submit(s,r["full_tokens"]+[4],4)
                                         for s,r in zip(names,first)))
            one=await c.submit("one",[1,2,3],1)
            assert len(one["trace"])==1
            assert {r["owner"] for r in first}==set(range(4))
            assert {r["trace"][0]["instance"] for r in first}==set(range(4))
            for a,b in zip(first,second):
                assert a["owner"]==b["owner"]
                assert b["trace"][0]["restored"]
                assert b["trace"][0]["cached"]==len(a["full_tokens"])-1
                key=c.directory.current(b["session"])
                assert c.p.cache[key]==c.d.cache[key]
            assert not c.p.resident and not c.d.resident
            with c.directory.transaction() as db:
                assert db.execute("SELECT COUNT(*) FROM sessions WHERE active!=0 OR owner!='P'").fetchone()[0]==0
        finally:await c.close()
    asyncio.run(scenario())


def test_uncertain_failure_settles_every_active_caller_and_fences(tmp_path):
    async def scenario():
        c=await start(tmp_path/"directory.db");c.p.block=asyncio.Event()
        requests=[asyncio.create_task(c.submit(f"s{i}",[1,2],8)) for i in range(7)]
        await asyncio.sleep(.02)
        c.p.fail=True;c.p.block.set()
        results=await asyncio.wait_for(asyncio.gather(*requests,return_exceptions=True),2)
        assert all(isinstance(r,Exception) for r in results)
        assert c.failure
        with c.directory.transaction() as db:
            assert db.execute("SELECT COUNT(*) FROM sessions WHERE active=1").fetchone()[0]==4
        with pytest.raises(RuntimeError):await c.submit("new",[1],2)
        await c.close()
    asyncio.run(scenario())


def test_cancelled_caller_does_not_release_session(tmp_path):
    async def scenario():
        c=await start(tmp_path/"directory.db");c.p.block=asyncio.Event()
        request=asyncio.create_task(c.submit("same",[1,2],4))
        await asyncio.sleep(.02);request.cancel()
        with pytest.raises(asyncio.CancelledError):await request
        with pytest.raises(pd.Conflict):await c.submit("same",[1,2],4)
        c.p.block.set()
        for _ in range(100):
            if not c.inflight:break
            await asyncio.sleep(.01)
        assert not c.inflight
        await c.close()
    asyncio.run(scenario())


def test_cache_miss_cold_recompute_and_divergent_prompt(tmp_path):
    async def scenario():
        c=await start(tmp_path/"directory.db")
        try:
            first=await c.submit("s",[1,2],4)
            c.p.cache.clear();c.d.cache.clear()
            cold=await c.submit("s",first["full_tokens"]+[3],2)
            assert not cold["trace"][0]["restored"]
            divergent=await c.submit("s",[8,9],2)
            assert not divergent["trace"][0]["restored"]
        finally:await c.close()
    asyncio.run(scenario())


def test_state_readback_corruption_fences_destination(tmp_path):
    async def scenario():
        c=await start(tmp_path/"directory.db")
        original=c.d.rpc
        async def corrupt(instance,op,**args):
            result=await original(instance,op,**args)
            if op=="export":
                blob=pd.unpack(c.d.cache[result["key"]])
                blob["shards"][0]["corrupt"]=True
                data=pd.pack(blob);key=hashlib.sha256(data).hexdigest()
                c.d.cache[key]=data;result["key"]=key
            return result
        c.d.rpc=corrupt
        with pytest.raises(RuntimeError,match="readback differs"):
            await c.submit("broken",[1,2],3)
        with c.directory.transaction() as db:
            row=db.execute("SELECT active,owner FROM sessions").fetchone()
            assert row==(1,f"D{pd.owner_for('broken')}")
        await c.close()
    asyncio.run(scenario())
