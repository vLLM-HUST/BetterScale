"""Single-controller, wave-gated P4/D4 coordination over explicit peer URLs.

Uses the existing SQLite single-writer directory. This is not distributed
consensus, automatic failover, online State loading, or an OpenAI API server.
"""
import asyncio
from dataclasses import dataclass
import hashlib
import fcntl
import math
import re
import sqlite3
import time
import zlib

import aiohttp
from naive_pool_node import MAX_STATE,pack,unpack
from model_checkpoint import IDENTITY
from session import Directory,Conflict
from pd_limits import context_limit

class RemoteError(RuntimeError):
    def __init__(self,status,message):
        self.status=status
        super().__init__(message)

class Peer:
    def __init__(self,url,client,trace=None):
        self.url=url.rstrip("/");self.client=client;self.trace=trace

    def _record(self,op,start,size,**fields):
        if self.trace:
            end=time.perf_counter()
            self.trace(dict(peer=self.url,op=op,start=start,end=end,seconds=end-start,
                            bytes=size,**fields))

    async def health(self):
        async with self.client.get(self.url+"/health") as response:
            if response.status!=200:raise RemoteError(response.status,"Node not healthy")
            return await response.json()

    async def rpc(self,instance,op,**args):
        start=time.perf_counter()
        async with self.client.post(self.url+"/rpc",
                data=pack(dict(instance=instance,op=op,args=args))) as response:
            data=await response.read()
            if response.status!=200:
                try:message=unpack(data)["error"]
                except (ValueError,KeyError,TypeError):message=data.decode(errors="replace")
                self._record(op,start,len(data),instance=instance,status=response.status)
                raise RemoteError(response.status,str(message)[:4096])
            value=unpack(data)
            self._record(op,start,len(data),instance=instance,owner=args.get("owner"),
                         requests=len(args.get("items",[])))
            return value

    async def get(self,key):
        start=time.perf_counter()
        async with self.client.get(self.url+"/checkpoint/"+key) as response:
            if response.status!=200:
                raise RemoteError(response.status,(await response.text())[:4096])
            data=await response.read()
            if not 0<len(data)<=MAX_STATE or hashlib.sha256(data).hexdigest()!=key:
                raise RuntimeError("Remote checkpoint content identity mismatch")
            self._record("get",start,len(data),key=key)
            return data

    async def put(self,key,data):
        start=time.perf_counter()
        async with self.client.put(self.url+"/checkpoint/"+key,data=data) as response:
            if response.status!=200:
                raise RemoteError(response.status,(await response.read())[:4096].decode(errors="replace"))
            result=await response.json()
            if result.get("key")!=key or result.get("bytes")!=len(data):
                raise RemoteError(response.status,"Destination checkpoint not acknowledged")
            self._record("put",start,len(data),key=key)

@dataclass
class Job:
    session:str
    prompt:list
    n:int
    owner:int
    future:object
    started:float
    trace:list
    tokens:list|None=None
    lease:object=None

def owner_for(session):
    return zlib.crc32(session.encode())%4

def select_wave(candidates,capacities,max_batch):
    """Reserve each request's complete generation footprint before importing."""
    if capacities is None:raise RuntimeError("Missing D capacity receipt")
    counts=[0]*4;blocks=[0]*4;selected=[];deferred=[]
    for job in candidates:
        owner=job.owner;cap=capacities[owner]
        # Keep native MTP lookahead allocation despite target-only proposals.
        need=math.ceil((len(job.tokens)+job.n-1+2)/cap["block_size"])
        if need>cap["free_blocks"]:raise ValueError("Request exceeds idle owner KV capacity")
        if (len(selected)>=max_batch or counts[owner]>=cap["max_requests"]
                or blocks[owner]+need>cap["free_blocks"]):
            deferred.append(job)
        else:
            selected.append(job);counts[owner]+=1;blocks[owner]+=need
    return selected,deferred

class Coordinator:
    def __init__(self,path,p_url,d_url,*,max_batch=8,verify_imports=False,trace=None):
        if not 1<=max_batch<=64:raise ValueError("D wave limit must be1..64")
        self.directory=Directory(path)
        self.urls=p_url,d_url
        self.max_batch=max_batch
        self.verify_imports=verify_imports
        self.capacities=None
        self.trace=trace
        self.controller_lock=None
        self.inflight={};self.pending=asyncio.Queue(maxsize=128)
        self.ready=asyncio.Queue(maxsize=128)
        self.failure=None;self.tasks=[];self.client=None

    async def start(self):
        self.controller_lock=open(self.directory.path+".controller.lock","a")
        try:
            fcntl.flock(self.controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.directory.transaction() as db:
                if db.execute("SELECT 1 FROM sessions WHERE active!=0 OR owner!='P' LIMIT 1").fetchone():
                    raise RuntimeError("Directory has unfinished ownership; explicit recovery required")
        except BaseException:
            self.controller_lock.close();self.controller_lock=None
            raise
        self.client=aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1860))
        self.p,self.d=[Peer(url,self.client,self.trace) for url in self.urls]
        try:
            for peer,kind,count in ((self.p,"P",4),(self.d,"D",1)):
                status=await peer.health()
                if (status.get("kind")!=kind or not status.get("ready")
                        or status.get("context_limit")!=context_limit()
                        or len(status.get("actors",[]))!=count
                        or any(not a["alive"] or a["quarantined"] for a in status["actors"])):
                    raise RuntimeError("Node topology/context/health does not match controller")
                if kind=="D":
                    self.capacities=status["actors"][0]["info"]["capacities"]
                    if len(self.capacities)!=4 or any(
                            c["context_limit"]!=context_limit() or c["block_size"]<=0
                            or c["free_blocks"]<math.ceil(context_limit()/c["block_size"])
                            or c["max_requests"]!=16 for c in self.capacities):
                        raise RuntimeError("D owner cannot admit one full-context request")
        except BaseException:
            await self.client.close()
            self.controller_lock.close();self.controller_lock=None
            raise
        self.tasks=[asyncio.create_task(self._prefill(i)) for i in range(4)]
        self.tasks.append(asyncio.create_task(self._decode()))
        return self

    async def close(self):
        self._fail([], RuntimeError("Coordinator closed; unfinished leases remain fenced"))
        for task in self.tasks:task.cancel()
        await asyncio.gather(*self.tasks,return_exceptions=True)
        if self.client:await self.client.close()
        if self.controller_lock:
            self.controller_lock.close();self.controller_lock=None

    async def submit(self,session,prompt,n):
        if self.failure:raise RuntimeError("Coordinator stopped after uncertain execution")
        if (not isinstance(session,str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}",session)
                or type(n) is not int or not 1<=n<=512
                or not isinstance(prompt,list) or not prompt or len(prompt)+n>context_limit()
                or any(type(t) is not int or not 0<=t<248320 for t in prompt)):
            raise ValueError("Invalid bounded request")
        if session in self.inflight:raise Conflict("Session already has an active turn")
        if self.pending.full():raise RuntimeError("Admission queue full")
        future=asyncio.get_running_loop().create_future()
        job=Job(session,list(prompt),n,owner_for(session),future,time.perf_counter(),[])
        self.inflight[session]=job
        def completed(value):
            self.inflight.pop(session,None)
            if not value.cancelled():value.exception()
        future.add_done_callback(completed)
        self.pending.put_nowait(job)
        # Caller cancellation does not revoke remote work or free its session.
        return await asyncio.shield(future)

    async def _copy(self,source,dest,key):
        data=await source.get(key)
        await dest.put(key,data)

    async def _import(self,peer,instance,owner,key,salt):
        original=unpack(await peer.get(key)) if self.verify_imports else None
        await peer.rpc(instance,"import",owner=owner,key=key,salt=salt)
        if original is not None:
            check=await peer.rpc(instance,"export",owner=owner,
                                 tokens=original["header"]["tokens"],salt=salt)
            restored=unpack(await peer.get(check["key"]))
            if (restored["shards"]!=original["shards"]
                    or any(restored["header"][k]!=original["header"][k]
                           for k in ("identity","cursor","tokens"))):
                raise RuntimeError("Post-H2D target State readback differs")
        return self.verify_imports

    def _fail(self,jobs,error):
        if self.failure is None:self.failure=str(error)
        # Settle every admitted caller, including work held by another worker.
        for job in list(self.inflight.values()):
            if not job.future.done():
                # Do not raise the same exception object through many futures:
                # Python would keep appending each caller's traceback to it.
                job.future.set_exception(RuntimeError("PD controller failed closed: "+str(error)))
        current=asyncio.current_task()
        for task in self.tasks:
            if task is not current:task.cancel()
        # Never revoke a lease whose remote mutation may still be running.
        for queue in (self.pending,self.ready):
            while not queue.empty():queue.get_nowait()

    async def _prefill(self,instance):
        while True:
            job=await self.pending.get()
            try:
                if self.failure:raise RuntimeError("Coordinator failed closed")
                try:self.directory.create(job.session,IDENTITY,"P")
                except sqlite3.IntegrityError:pass
                job.lease=self.directory.claim(job.session,"P",IDENTITY)
                salt="pd:"+job.session
                restored=False
                if job.lease.base:
                    try:
                        old=unpack(await self.d.get(job.lease.base))["header"]["tokens"]
                        if job.prompt[:len(old)]==old:
                            await self._import(self.p,instance,0,job.lease.base,salt)
                            restored=True
                    except RemoteError as error:
                        if error.status!=404:raise
                        # Full prompt remains available: a cache miss recomputes,
                        # never fabricates a warm State hit.
                result=(await self.p.rpc(instance,"generate_batch",items=[
                    dict(owner=0,tokens=job.prompt,salt=salt,n=1)]))[0]
                job.tokens=job.prompt+result["token_ids"]
                checkpoint=await self.p.rpc(instance,"export",owner=0,tokens=job.tokens,salt=salt)
                await self.p.rpc(instance,"drop",owner=0,salt=salt)
                await self._copy(self.p,self.d,checkpoint["key"])
                job.trace.append(dict(stage="P",instance=instance,restored=restored,
                    exact_state_readback=restored and self.verify_imports,
                    cached=result["cached"],checkpoint=checkpoint,
                    elapsed=time.perf_counter()-job.started))
                next_owner="P" if job.n==1 else f"D{job.owner}"
                self.directory.publish(job.lease,checkpoint["key"],next_owner)
                job.lease=None
                if job.n==1:
                    job.future.set_result(self._result(job))
                else:
                    await self.ready.put(job)
            except Exception as error:
                self._fail([job],error)
                return

    def _result(self,job):
        return dict(session=job.session,owner=job.owner,
                    token_ids=job.tokens[len(job.prompt):],full_tokens=job.tokens,
                    trace=job.trace,seconds=time.perf_counter()-job.started)

    async def _decode(self):
        deferred=[]
        while True:
            first=deferred.pop(0) if deferred else await self.ready.get()
            await asyncio.sleep(.025)
            candidates=[first]+deferred;deferred=[]
            while not self.ready.empty():candidates.append(self.ready.get_nowait())
            try:
                jobs,deferred=select_wave(candidates,self.capacities,self.max_batch)
            except Exception as error:
                self._fail(candidates,error)
                return
            try:
                if self.failure:raise RuntimeError("Coordinator failed closed")
                for job in jobs:
                    job.lease=self.directory.claim(job.session,f"D{job.owner}",IDENTITY)
                    await self._import(self.d,0,job.owner,job.lease.base,"pd:"+job.session)
                values=await self.d.rpc(0,"generate_batch",items=[
                    dict(owner=j.owner,tokens=j.tokens,salt="pd:"+j.session,n=j.n-1) for j in jobs])
                for job,result in zip(jobs,values,strict=True):
                    job.tokens+=result["token_ids"]
                    checkpoint=await self.d.rpc(0,"export",owner=job.owner,
                                                tokens=job.tokens,salt="pd:"+job.session)
                    await self.d.rpc(0,"drop",owner=job.owner,salt="pd:"+job.session)
                    await self._copy(self.d,self.p,checkpoint["key"])
                    self.directory.publish(job.lease,checkpoint["key"],"P")
                    job.lease=None
                    job.trace.append(dict(stage="D",owner=job.owner,wave_size=len(jobs),
                        exact_state_readback=self.verify_imports,
                        cached=result["cached"],checkpoint=checkpoint,
                        elapsed=time.perf_counter()-job.started))
                    job.future.set_result(self._result(job))
            except Exception as error:
                self._fail(jobs+deferred,error)
                return

