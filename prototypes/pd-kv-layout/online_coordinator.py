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
    def __init__(self,path,p_url,d_url,trace=None,*,p_per_instance=16,same_host_d=False,sticky_owners=True):
        self.directory=Directory(path);self.urls=(p_url,d_url);self.trace=trace
        self.same_host_d=same_host_d
        self.inflight={};self.generated={};self.tasks=set();self.failure=None;self.client=None;self.lock=None
        self.sticky_owners=sticky_owners
        self.p_affinity={};self.d_affinity={}
        self.d_changed=asyncio.Condition()
        if sticky_owners:
            with self.directory.transaction() as db:
                db.execute("CREATE TABLE IF NOT EXISTS rank_placements(session TEXT PRIMARY KEY,p_owner INTEGER,d_owner INTEGER)")
                rows=list(db.execute("SELECT session,p_owner,d_owner FROM rank_placements"))
                self.p_affinity={session:p for session,p,d in rows if p is not None}
                self.d_affinity={session:d for session,p,d in rows if d is not None}
            if any(type(i) is not int or not 0<=i<4 for i in (*self.p_affinity.values(),*self.d_affinity.values())):
                raise ValueError("Invalid persisted attention owner")
        if type(p_per_instance) is not int or not 1<=p_per_instance<=16:
            raise ValueError("P pipeline admits one to sixteen requests per TP2 instance")
        self.p_limit=p_per_instance
        self.p_admission=[Admission(seats=p_per_instance) for _ in range(4)]
        self.p_changed=asyncio.Condition()
        self.maintenance={(kind,i):asyncio.Lock() for kind in ("P","D") for i in range(4)}
        self.admission=[Admission() for _ in range(4)]
        self.sink=PeerObjectSink(self.urls)
        self.rank_private=False

    async def start(self):
        self.lock=open(self.directory.path+".controller.lock","a")
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with self.directory.transaction() as db:
            if db.execute("SELECT 1 FROM sessions WHERE active!=0 OR owner!=? LIMIT 1",("P",)).fetchone():
                raise RuntimeError("Unfinished ownership requires explicit recovery")
            if self.sticky_owners and db.execute(
                    "SELECT 1 FROM sessions s LEFT JOIN rank_placements p ON s.id=p.session "
                    "WHERE s.manifest IS NOT NULL AND p.p_owner IS NULL LIMIT 1").fetchone():
                raise RuntimeError("Existing session lacks sticky placement; explicit migration required")
        self.client=aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1860))
        self.p,self.d=[Peer(url,self.client,self.trace) for url in self.urls]
        wire_versions=set()
        for peer,kind,count in ((self.p,"P",4),(self.d,"D",1)):
            h=await peer.health()
            if (not h["ready"] or h["kind"]!=kind or h["context_limit"]!=context_limit()
                    or len(h["actors"])!=count or any(not a["alive"] or a["quarantined"] for a in h["actors"])):
                raise RuntimeError("Online node topology/context mismatch")
            wire_versions.update(a["info"].get("state_wire","raw-v2") for a in h["actors"])
            if kind=="P":
                capacities=[a["info"]["capacities"] for a in h["actors"]]
                if any(len(cs)!=1 or cs[0]["block_size"]!=2048 or cs[0]["max_requests"]!=16 for cs in capacities):
                    raise RuntimeError("Unexpected P capacity receipt")
                self.p_admission=[Admission(cs[0]["free_blocks"],min(self.p_limit,cs[0]["max_requests"])) for cs in capacities]
            if kind=="D":
                capacities=h["actors"][0]["info"]["capacities"]
                if len(capacities)!=4 or any(c["block_size"]!=2048 or c["max_requests"]!=16 for c in capacities):
                    raise RuntimeError("Unexpected owner capacity")
                self.admission=[Admission(c["free_blocks"],c["max_requests"]) for c in capacities]
        if len(wire_versions)!=1 or not wire_versions<={"raw-v2","zstd-resident-v1","mtp-prefix-raw-v2","mtp-prefix-zstd-resident-v1","rank-private-v1"}:
            raise RuntimeError("P/D State wire version mismatch")
        self.rank_private=wire_versions=={"rank-private-v1"}
        if self.rank_private:
            if not self.sticky_owners:
                raise ValueError("private rank State requires sticky placement")
            with self.directory.transaction() as db:
                db.execute("CREATE TABLE IF NOT EXISTS rank_manifests("
                           "id TEXT PRIMARY KEY, payload BLOB NOT NULL, p_owner INTEGER, d_owner INTEGER)")
        return self

    async def acquire_p(self,session,tokens=1):
        count=math.ceil((tokens+2)/2048)
        if count>max(a.blocks for a in self.p_admission):
            raise ValueError("Request exceeds P KV budget")
        async with self.p_changed:
            owner=self.p_affinity.get(session) if self.sticky_owners else None
            if owner is not None and count>self.p_admission[owner].blocks:
                raise ValueError("Request exceeds sticky P owner KV budget")
            def eligible():
                owner=self.p_affinity.get(session) if self.sticky_owners else None
                return [i for i,a in enumerate(self.p_admission)
                        if (owner is None or i==owner) and a.slots>0 and a.free>=count]
            await self.p_changed.wait_for(eligible)
            choices=eligible();preferred=self.p_affinity.get(session)
            chosen=preferred if preferred in choices else min(choices,
                key=lambda i:(-self.p_admission[i].free,self.p_admission[i].seats-self.p_admission[i].slots))
            if self.sticky_owners and session not in self.p_affinity:
                # Placement precedes State creation. Changing this owner later
                # requires an explicit migration, never a capacity-only reroute.
                with self.directory.transaction() as db:
                    db.execute("INSERT INTO rank_placements(session,p_owner) VALUES(?,?) "
                               "ON CONFLICT(session) DO UPDATE SET p_owner=excluded.p_owner",(session,chosen))
            a=self.p_admission[chosen];a.free-=count;a.slots-=1
            self.p_affinity[session]=chosen
            return chosen,count

    async def acquire_d(self,session,tokens):
        if not self.sticky_owners:
            owner=owner_for(session)
            return owner,await self.admission[owner].acquire(tokens)
        count=math.ceil((tokens+2)/2048)
        if count>max(a.blocks for a in self.admission):
            raise ValueError("Request exceeds D KV budget")
        async with self.d_changed:
            owner=self.d_affinity.get(session)
            if owner is not None and count>self.admission[owner].blocks:
                raise ValueError("Request exceeds sticky D owner KV budget")
            def eligible():
                owner=self.d_affinity.get(session)
                return [i for i,a in enumerate(self.admission)
                        if (owner is None or i==owner) and a.slots>0 and a.free>=count]
            await self.d_changed.wait_for(eligible)
            chosen=min(eligible(),key=lambda i:(-self.admission[i].free,
                self.admission[i].seats-self.admission[i].slots))
            if session not in self.d_affinity:
                with self.directory.transaction() as db:
                    db.execute("INSERT INTO rank_placements(session,d_owner) VALUES(?,?) "
                               "ON CONFLICT(session) DO UPDATE SET d_owner=excluded.d_owner",(session,chosen))
                self.d_affinity[session]=chosen
            a=self.admission[chosen];a.free-=count;a.slots-=1
            return chosen,count

    async def release_d(self,index,count):
        if not self.sticky_owners:return await self.admission[index].release(count)
        async with self.d_changed:
            a=self.admission[index];a.free+=count;a.slots+=1
            self.d_changed.notify_all()

    async def release_p(self,index,count):
        async with self.p_changed:
            a=self.p_admission[index];a.free+=count;a.slots+=1
            self.p_changed.notify_all()

    async def cache(self,kind,index,command):
        peer=self.p if kind=="P" else self.d
        start=time.perf_counter()
        result=await peer.rpc(index if kind=="P" else 0,"cache",owner=0 if kind=="P" else index,command=command)
        if self.trace:
            self.trace(dict(op="cache-control",kind=kind,index=index,action=command["kind"],
                            operation=command.get("operation"),start=start,end=time.perf_counter(),
                            receipt=result if command["kind"]=="wait" else None))
        return result

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

    async def read_manifest(self,manifest):
        if not self.rank_private:
            return json.loads(await asyncio.to_thread(self.sink.get,manifest))
        with self.directory.transaction() as db:
            row=db.execute("SELECT payload FROM rank_manifests WHERE id=?",(manifest,)).fetchone()
        if row is None:raise KeyError("private checkpoint metadata missing")
        if hashlib.sha256(row[0]).hexdigest()!=manifest:
            raise ValueError("private checkpoint metadata checksum mismatch")
        return json.loads(row[0])

    async def retire_manifest(self,manifest):
        if not self.rank_private:return
        with self.directory.transaction() as db:
            row=db.execute("SELECT payload,p_owner,d_owner FROM rank_manifests WHERE id=?",(manifest,)).fetchone()
        if row is None:raise KeyError("private checkpoint retirement metadata missing")
        checkpoint=json.loads(row[0])
        for kind,index in (("P",row[1]),("D",row[2])):
            if index is None:continue
            async with self.maintenance[kind,index]:
                op=await self.prepare(kind,index,dict(kind="drop",key=checkpoint["key"]))
            await self.cache(kind,index,dict(kind="wait",operation=op))
        with self.directory.transaction() as db:
            db.execute("DELETE FROM rank_manifests WHERE id=?",(manifest,))

    async def save(self,kind,index,tokens,salt,*,peer_group=None):
        async with self.maintenance[kind,index]:
            key=uuid.uuid4().hex
            command=dict(kind="store_match",tokens=tokens,salt=salt,key=key)
            if self.rank_private:command["peer_group"]=peer_group
            op=await self.prepare(kind,index,command)
        receipt=await self.cache(kind,index,dict(kind="wait",operation=op))
        checkpoint=await self.cache(kind,index,dict(kind="describe",key=key))
        data=json.dumps(checkpoint,separators=(",",":")).encode()
        manifest=hashlib.sha256(data).hexdigest()
        if self.rank_private:
            if peer_group is not None:
                peer_kind="D" if kind=="P" else "P"
                async with self.maintenance[peer_kind,peer_group]:
                    await self.prepare(peer_kind,peer_group,dict(kind="adopt",checkpoint=checkpoint))
            p_owner=index if kind=="P" else peer_group
            d_owner=index if kind=="D" else peer_group
            with self.directory.transaction() as db:
                db.execute("INSERT INTO rank_manifests VALUES(?,?,?,?)",
                           (manifest,data,p_owner,d_owner))
        else:
            await asyncio.to_thread(self.sink.put,manifest,data)
        return manifest,checkpoint,receipt

    async def submit(self,session,prompt,n,*,on_tokens=None,output_ready=False):
        if self.failure:raise RuntimeError(self.failure)
        if (not isinstance(session,str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}",session)
                or not isinstance(prompt,list) or not prompt or any(type(t) is not int or not 0<=t<248320 for t in prompt)
                or type(n) is not int or not 1<=n<=context_limit() or len(prompt)+n>context_limit()):raise ValueError("Invalid request")
        if session in self.inflight:
            generated=self.generated[session]
            if not generated.done():raise Conflict("Session already generating")
            # A completed response may be followed immediately by another turn.
            # Wait only this session's commit, never the global DP wave.
            await asyncio.shield(self.inflight[session])
            if self.failure:raise RuntimeError(self.failure)
            if session in self.inflight:raise Conflict("Session admission raced")
        if len(self.inflight)>=128:raise RuntimeError("Online request bound exceeded")
        future=asyncio.get_running_loop().create_future();self.inflight[session]=future
        generated=asyncio.get_running_loop().create_future();self.generated[session]=generated
        generated.add_done_callback(lambda value:None if value.cancelled() else value.exception())
        def done(value):
            self.inflight.pop(session,None);self.generated.pop(session,None)
            if not value.cancelled():value.exception()
        future.add_done_callback(done)
        task=asyncio.create_task(self.turn(session,list(prompt),n,future,generated,on_tokens));self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return await asyncio.shield(generated if output_ready else future)

    async def turn(self,session,prompt,n,future,generated,on_tokens):
        started=time.perf_counter();owner=owner_for(session);events=[]
        reserved=None;p_instance=None;p_reserved=None
        retired=[]
        async def generate(peer,instance,**args):
            if on_tokens is None:return await peer.rpc(instance,"generate",**args)
            from online_stream import generate as streamed
            delays=[];stages={name:[] for name in ("actor_to_node","node_queue","node_to_coordinator")}
            def progress(item):
                if peer is self.d and self.same_host_d:
                    now=time.perf_counter_ns()
                    delays.append((now-item["arrival_ns"])/1e6)
                    if "node_arrival_ns" in item and "node_send_ns" in item:
                        points=(item["arrival_ns"],item["node_arrival_ns"],item["node_send_ns"],now)
                        for name,a,b in zip(stages,points,points[1:]):stages[name].append((b-a)/1e6)
                on_tokens(item)
            result=await streamed(peer,instance,progress,**args)
            if self.trace and delays:
                ordered=sorted(delays)
                self.trace(dict(op="D-stream-return",session=session,owner=owner,
                    request_id=result["request_id"],same_host=True,chunks=len(delays),
                    p50_ms=ordered[len(ordered)//2],p95_ms=ordered[int((len(ordered)-1)*.95)],
                    max_ms=ordered[-1],stages_ms={name:dict(
                        p50=sorted(values)[len(values)//2],
                        p95=sorted(values)[int((len(values)-1)*.95)],max=max(values))
                        for name,values in stages.items() if values},
                    scope="actor yield to coordinator token callback; not device cadence"))
            return result
        def output_complete(tokens,cached):
            if self.trace:self.trace(dict(op="output-ready",session=session,owner=owner,
                start=started,end=time.perf_counter(),prompt_tokens=len(prompt),output_tokens=n))
            if not generated.done():generated.set_result(dict(session=session,owner=owner,
                token_ids=tokens[len(prompt):],full_tokens=tokens,cached=cached,
                seconds=time.perf_counter()-started))
        try:
            # Bound complete D footprint before creating source work. P workers
            # remain shared; independent owners and in-flight turns are online.
            if n>1:owner,reserved=await self.acquire_d(session,len(prompt)+n)
            p_instance,p_reserved=await self.acquire_p(session,len(prompt)+1)
            if self.failure:raise RuntimeError(self.failure)
            try:self.directory.create(session,IDENTITY,"P")
            except sqlite3.IntegrityError:pass
            lease=self.directory.claim(session,"P",IDENTITY)
            salt="online:"+session
            if lease.base:
                cp=await self.read_manifest(lease.base)
                retired.append(lease.base)
                if prompt[:len(cp["tokens"])]==cp["tokens"]:
                    loaded=await self.load("P",p_instance,cp)
                    events.append(dict(stage="P-load",receipt=loaded))
                else:
                    # Divergence is a new numerical/cache lineage, not permission
                    # to overwrite immutable versions of the old prefix.
                    salt="online:"+uuid.uuid4().hex
            if lease.base and "cp" in locals() and prompt[:len(cp["tokens"])]==cp["tokens"]:salt=cp["salt"]
            value=await generate(self.p,p_instance,owner=0,tokens=prompt,salt=salt,n=1)
            tokens=value["full_tokens"];prompt_cached=value["cached"]
            if n==1:output_complete(tokens,prompt_cached)
            save_args={"peer_group":self.d_affinity.get(session) if n==1 else owner} if self.rank_private else {}
            manifest,cp,saved=await self.save("P",p_instance,tokens,salt,**save_args)
            events.append(dict(stage="P",instance=p_instance,cached=value["cached"],save=saved,
                               elapsed=time.perf_counter()-started))
            self.directory.publish(lease,manifest,"P" if n==1 else f"D{owner}")
            await self.release_p(p_instance,p_reserved);p_instance=None
            if n>1:
                retired.append(manifest)
                lease=self.directory.claim(session,f"D{owner}",IDENTITY)
                loaded=await self.load("D",owner,cp)
                value=await generate(self.d,0,owner=owner,tokens=tokens,salt=salt,n=n-1)
                tokens=value["full_tokens"]
                output_complete(tokens,prompt_cached)
                save_args={"peer_group":self.p_affinity[session]} if self.rank_private else {}
                manifest,cp,saved=await self.save("D",owner,tokens,salt,**save_args)
                self.directory.publish(lease,manifest,"P")
                events.append(dict(stage="D",owner=owner,cached=value["cached"],load=loaded,
                    save=saved,request_id=value["request_id"],actor_start_ns=value["start_ns"],arrivals=value["arrivals"],elapsed=time.perf_counter()-started))
            if self.rank_private:
                for obsolete in dict.fromkeys(retired):
                    await self.retire_manifest(obsolete)
            if self.trace:self.trace(dict(op="turn-committed",session=session,owner=owner,
                start=started,end=time.perf_counter(),events=events))
            if not future.done():future.set_result(dict(session=session,owner=owner,
                token_ids=tokens[len(prompt):],full_tokens=tokens,trace=events,seconds=time.perf_counter()-started))
        except BaseException as error:
            if self.failure is None:self.failure=str(error)
            for f in (*self.inflight.values(),*self.generated.values()):
                if not f.done():f.set_exception(RuntimeError("Online PD failed closed: "+str(error)))
            # Other admitted tasks are interrupted; no uncertain lease revoked.
            for task in self.tasks:
                if task is not asyncio.current_task():task.cancel()
        finally:
            if p_instance is not None and not self.failure:await self.release_p(p_instance,p_reserved)
            if reserved is not None and not self.failure:await self.release_d(owner,reserved)

    async def close(self):
        for task in self.tasks:task.cancel()
        await asyncio.gather(*self.tasks,return_exceptions=True)
        if self.client:await self.client.close()
        if self.lock:self.lock.close()
