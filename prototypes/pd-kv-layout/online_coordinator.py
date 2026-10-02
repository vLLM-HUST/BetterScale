"""Single-writer online PD control: no whole-State payloads or global DP drain."""
import asyncio
import fcntl
import hashlib
import json
import math
import re
import sqlite3
import time
import uuid
import aiohttp
from naive_pd_coordinator import Peer,owner_for
from online_objects import PeerObjectSink
from session import Directory,Conflict
from pd_limits import context_limit

IDENTITY="qwen35-online-target-v2"

class Admission:
    def __init__(self,blocks=1044,seats=16):
        self.blocks,self.seats=blocks,seats
        self.free,self.slots=blocks,seats
        self.condition=asyncio.Condition()
    async def acquire(self,tokens):
        count=math.ceil((tokens+2)/2048)
        if count>self.blocks:raise ValueError("Request exceeds owner KV budget")
        async with self.condition:
            await self.condition.wait_for(lambda:self.free>=count and self.slots>0)
            self.free-=count;self.slots-=1
        return count
    async def release(self,count):
        async with self.condition:
            self.free+=count;self.slots+=1;self.condition.notify_all()

class Coordinator:
    def __init__(self,path,p_url,d_url,trace=None):
        self.directory=Directory(path);self.urls=(p_url,d_url);self.trace=trace
        self.inflight={};self.tasks=set();self.failure=None;self.client=None;self.lock=None
        self.p_available=asyncio.Queue()
        for i in range(4):self.p_available.put_nowait(i)
        self.maintenance={(kind,i):asyncio.Lock() for kind in ("P","D") for i in range(4)}
        self.admission=[Admission() for _ in range(4)]
        self.sink=PeerObjectSink(self.urls)

    async def start(self):
        self.lock=open(self.directory.path+".controller.lock","a")
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with self.directory.transaction() as db:
            if db.execute("SELECT 1 FROM sessions WHERE active!=0 OR owner!=? LIMIT 1",("P",)).fetchone():
                raise RuntimeError("Unfinished ownership requires explicit recovery")
        self.client=aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1860))
        self.p,self.d=[Peer(url,self.client,self.trace) for url in self.urls]
        for peer,kind,count in ((self.p,"P",4),(self.d,"D",1)):
            h=await peer.health()
            if (not h["ready"] or h["kind"]!=kind or h["context_limit"]!=context_limit()
                    or len(h["actors"])!=count or any(not a["alive"] or a["quarantined"] for a in h["actors"])):
                raise RuntimeError("Online node topology/context mismatch")
            if kind=="D":
                capacities=h["actors"][0]["info"]["capacities"]
                if len(capacities)!=4 or any(c["block_size"]!=2048 or c["max_requests"]!=16 for c in capacities):
                    raise RuntimeError("Unexpected owner capacity")
                self.admission=[Admission(c["free_blocks"],c["max_requests"]) for c in capacities]
        return self

    async def cache(self,kind,index,command):
        peer=self.p if kind=="P" else self.d
        return await peer.rpc(index if kind=="P" else 0,"cache",owner=0 if kind=="P" else index,command=command)

    async def prepare(self,kind,index,command):
        deadline=time.monotonic()+300
        while True:
            value=await self.cache(kind,index,command)
            if value is not None:return value
            if time.monotonic()>deadline:raise TimeoutError("State admission/frontier deadline")
            await asyncio.sleep(.02)

    async def load(self,kind,index,checkpoint):
        async with self.maintenance[kind,index]:
            await self.prepare(kind,index,dict(kind="adopt",checkpoint=checkpoint))
            op=await self.prepare(kind,index,dict(kind="load_match",key=checkpoint["key"]))
            return op if isinstance(op,dict) else await self.cache(kind,index,dict(kind="wait",operation=op))

    async def save(self,kind,index,tokens,salt):
        async with self.maintenance[kind,index]:
            key=uuid.uuid4().hex
            op=await self.prepare(kind,index,dict(kind="store_match",tokens=tokens,salt=salt,key=key))
            receipt=await self.cache(kind,index,dict(kind="wait",operation=op))
            checkpoint=await self.cache(kind,index,dict(kind="describe",key=key))
            data=json.dumps(checkpoint,separators=(",",":")).encode()
            manifest=hashlib.sha256(data).hexdigest()
            await asyncio.to_thread(self.sink.put,manifest,data)
            return manifest,checkpoint,receipt

    async def submit(self,session,prompt,n):
        if self.failure:raise RuntimeError(self.failure)
        if (not isinstance(session,str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}",session)
                or not isinstance(prompt,list) or not prompt or any(type(t) is not int or not 0<=t<248320 for t in prompt)
                or type(n) is not int or not 1<=n<=512 or len(prompt)+n>context_limit()):raise ValueError("Invalid request")
        if session in self.inflight:raise Conflict("Session already active")
        if len(self.inflight)>=128:raise RuntimeError("Online request bound exceeded")
        future=asyncio.get_running_loop().create_future();self.inflight[session]=future
        def done(value):
            self.inflight.pop(session,None)
            if not value.cancelled():value.exception()
        future.add_done_callback(done)
        task=asyncio.create_task(self.turn(session,list(prompt),n,future));self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return await asyncio.shield(future)

    async def turn(self,session,prompt,n,future):
        started=time.perf_counter();owner=owner_for(session);events=[]
        reserved=None;p_instance=None
        try:
            # Bound complete D footprint before creating source work. P workers
            # remain shared; independent owners and in-flight turns are online.
            if n>1:reserved=await self.admission[owner].acquire(len(prompt)+n)
            p_instance=await self.p_available.get()
            if self.failure:raise RuntimeError(self.failure)
            try:self.directory.create(session,IDENTITY,"P")
            except sqlite3.IntegrityError:pass
            lease=self.directory.claim(session,"P",IDENTITY)
            salt="online:"+session
            if lease.base:
                cp=json.loads(await asyncio.to_thread(self.sink.get,lease.base))
                if prompt[:len(cp["tokens"])]==cp["tokens"]:
                    loaded=await self.load("P",p_instance,cp)
                    events.append(dict(stage="P-load",receipt=loaded))
                else:
                    # Divergence is a new numerical/cache lineage, not permission
                    # to overwrite immutable versions of the old prefix.
                    salt="online:"+uuid.uuid4().hex
            if lease.base and "cp" in locals() and prompt[:len(cp["tokens"])]==cp["tokens"]:salt=cp["salt"]
            value=await self.p.rpc(p_instance,"generate",owner=0,tokens=prompt,salt=salt,n=1)
            tokens=value["full_tokens"]
            manifest,cp,saved=await self.save("P",p_instance,tokens,salt)
            events.append(dict(stage="P",instance=p_instance,cached=value["cached"],save=saved,
                               elapsed=time.perf_counter()-started))
            self.directory.publish(lease,manifest,"P" if n==1 else f"D{owner}")
            self.p_available.put_nowait(p_instance);p_instance=None
            if n>1:
                lease=self.directory.claim(session,f"D{owner}",IDENTITY)
                loaded=await self.load("D",owner,cp)
                value=await self.d.rpc(0,"generate",owner=owner,tokens=tokens,salt=salt,n=n-1)
                tokens=value["full_tokens"]
                manifest,cp,saved=await self.save("D",owner,tokens,salt)
                self.directory.publish(lease,manifest,"P")
                events.append(dict(stage="D",owner=owner,cached=value["cached"],load=loaded,
                    save=saved,arrivals=value["arrivals"],elapsed=time.perf_counter()-started))
            if not future.done():future.set_result(dict(session=session,owner=owner,
                token_ids=tokens[len(prompt):],full_tokens=tokens,trace=events,seconds=time.perf_counter()-started))
        except BaseException as error:
            if self.failure is None:self.failure=str(error)
            for f in self.inflight.values():
                if not f.done():f.set_exception(RuntimeError("Online PD failed closed: "+str(error)))
            # Other admitted tasks are interrupted; no uncertain lease revoked.
            for task in self.tasks:
                if task is not asyncio.current_task():task.cancel()
        finally:
            if p_instance is not None:self.p_available.put_nowait(p_instance)
            if reserved is not None and not self.failure:await self.admission[owner].release(reserved)

    async def close(self):
        for task in self.tasks:task.cancel()
        await asyncio.gather(*self.tasks,return_exceptions=True)
        if self.client:await self.client.close()
        if self.lock:self.lock.close()
