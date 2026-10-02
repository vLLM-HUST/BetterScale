from types import SimpleNamespace
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
        c=Coordinator(tmp_path/"directory.db","http://10.244.1.16:55581","http://10.244.2.32:55586",p_per_instance=1,sticky_owners=False)
        c.p_affinity["warm"]=2
        assert (await c.acquire_p("warm"))[0]==2
        assert (await c.acquire_p("another"))[0]==0
        c.p_affinity["busy"]=2
        assert (await asyncio.wait_for(c.acquire_p("busy"),.1))[0]==1
        assert sum(a.slots for a in c.p_admission)==1
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
                capacities=[dict(block_size=2048,max_requests=16,free_blocks=1044)]*(1 if kind=="P" else 4))
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


def test_prefill_admission_uses_available_pages_not_four_request_cap(tmp_path):
    from online_coordinator import Coordinator
    async def run():
        c=Coordinator(tmp_path/"d.db","http://10.244.1.16:55581","http://10.244.2.32:55586")
        c.p_admission=[Admission(260,16) for _ in range(4)]
        held=[await c.acquire_p(str(i),262144) for i in range(8)]
        assert all(a.free==2 and a.slots==14 for a in c.p_admission)
        c.p_affinity["small"]=2
        i,n=await c.acquire_p("small",1);assert (i,n)==(2,1)
        waiter=asyncio.create_task(c.acquire_p("large",262144))
        await asyncio.sleep(0);assert not waiter.done()
        await c.release_p(*held.pop())
        held.append(await asyncio.wait_for(waiter,1))
        await c.release_p(i,n)
        for row in held:await c.release_p(*row)
        assert all(a.free==260 and a.slots==16 for a in c.p_admission)
        short=[await c.acquire_p(str(i),1) for i in range(64)]
        assert all(a.slots==0 for a in c.p_admission)
        for row in short:await c.release_p(*row)
    asyncio.run(run())


def test_owner_admission_lock_does_not_cover_transfer_completion(tmp_path):
    from online_coordinator import Coordinator
    async def run():
        c=Coordinator(tmp_path/"d.db","http://10.244.1.16:55581","http://10.244.2.32:55586")
        release=asyncio.Event();both=asyncio.Event();waiting=[];number=0
        async def prepare(kind,index,command):
            nonlocal number
            if command["kind"]=="adopt":return True
            number+=1;return number
        async def cache(kind,index,command):
            if command["kind"]=="wait":
                waiting.append(command["operation"])
                if len(waiting)==2:both.set()
                await release.wait();return {"done":True}
            return {"key":command["key"]}
        c.prepare=prepare;c.cache=cache;c.sink=SimpleNamespace(put=lambda *args:None)
        a=asyncio.create_task(c.save("D",0,[1,2],"a"))
        b=asyncio.create_task(c.load("D",0,{"key":"other"}))
        await asyncio.wait_for(both.wait(),1)
        assert len(waiting)==2 and not a.done() and not b.done()
        release.set();await asyncio.gather(a,b)
    asyncio.run(run())



def test_sticky_owners_warm_owner_waits_and_survives_controller_recreation(tmp_path):
    from online_coordinator import Coordinator
    async def run():
        path=tmp_path/"rank-local.db"
        urls=("http://10.244.1.16:55581","http://10.244.2.32:55586")
        c=Coordinator(path,*urls,p_per_instance=1,sticky_owners=True)
        held=await c.acquire_p("warm",1)
        waiter=asyncio.create_task(c.acquire_p("warm",1))
        await asyncio.sleep(0)
        assert not waiter.done()
        other=await asyncio.wait_for(c.acquire_p("unrelated",1),1)
        assert other[0]!=held[0]  # no global admission stall
        await c.release_p(*held)
        assert (await asyncio.wait_for(waiter,1))[0]==held[0]
        reopened=Coordinator(path,*urls,p_per_instance=1,sticky_owners=True)
        assert reopened.p_affinity["warm"]==held[0]
        assert (await reopened.acquire_p("warm",1))[0]==held[0]
    asyncio.run(run())



def test_sticky_owners_new_d_uses_capacity_and_old_d_sticks(tmp_path):
    from online_coordinator import Coordinator
    async def run():
        path=tmp_path/"d-owners.db"
        urls=("http://10.244.1.16:55581","http://10.244.2.32:55586")
        c=Coordinator(path,*urls,sticky_owners=True,p_per_instance=1)
        c.admission=[Admission(n,1) for n in (100,200,300,400)]
        first=await c.acquire_d("warm",4096)
        assert first==(3,3)
        second=await c.acquire_d("new",4096)
        assert second==(2,3)
        wait=asyncio.create_task(c.acquire_d("warm",4096))
        await asyncio.sleep(0);assert not wait.done()
        unrelated=await c.acquire_d("third",4096)
        assert unrelated[0]==1
        await c.release_d(*first)
        assert await asyncio.wait_for(wait,1)==first
        # P assignment must not overwrite the earlier D assignment in SQLite.
        await c.acquire_p("warm",1)
        reopened=Coordinator(path,*urls,sticky_owners=True)
        assert reopened.d_affinity["warm"]==3
        assert reopened.p_affinity["warm"]==c.p_affinity["warm"]
        assert (await reopened.acquire_d("warm",4096))[0]==3
    asyncio.run(run())



def test_sticky_owner_cannot_silently_move_for_oversize_request(tmp_path):
    from online_coordinator import Coordinator
    async def run():
        c=Coordinator(tmp_path/"owner-budget.db","http://10.244.1.16:55581","http://10.244.2.32:55586")
        c.p_affinity["p"]=0;c.d_affinity["d"]=0
        c.p_admission[0]=Admission(1,16);c.admission[0]=Admission(1,16)
        with pytest.raises(ValueError,match="sticky P"):
            await c.acquire_p("p",4096)
        with pytest.raises(ValueError,match="sticky D"):
            await c.acquire_d("d",4096)
        assert c.p_affinity["p"]==c.d_affinity["d"]==0
    asyncio.run(run())



def test_existing_state_without_placement_is_not_rerouted_on_upgrade(tmp_path):
    from online_coordinator import Coordinator,IDENTITY
    async def run():
        c=Coordinator(tmp_path/"old.db","http://10.244.1.16:55581","http://10.244.2.32:55586")
        c.directory.create("old",IDENTITY,"P")
        lease=c.directory.claim("old","P",IDENTITY)
        c.directory.publish(lease,"saved-manifest","P")
        try:
            with pytest.raises(RuntimeError,match="lacks sticky placement"):await c.start()
        finally:await c.close()
    asyncio.run(run())


def test_output_ready_backup_does_not_reject_next_request_at_ingress_bound(tmp_path):
    from online_coordinator import Coordinator
    async def run():
        c=Coordinator(tmp_path/"bounded.db","http://10.244.1.16:55581","http://10.244.2.32:55586")
        commits={};started=[]
        async def turn(session,prompt,n,future,generated,on_tokens):
            started.append(session);commits[session]=asyncio.Event()
            generated.set_result(session)
            await commits[session].wait()
            future.set_result(session)
        c.turn=turn
        assert len(await asyncio.gather(*(c.submit(str(i),[1],1,output_ready=True) for i in range(128))))==128
        assert len(c.inflight)==128
        waiting=asyncio.create_task(c.submit("next",[1],1,output_ready=True))
        await asyncio.sleep(0)
        assert not waiting.done() and "next" not in started
        commits["0"].set()
        assert await asyncio.wait_for(waiting,1)=="next"
        assert len(c.inflight)==128
        for e in commits.values():e.set()
        await asyncio.gather(*c.tasks)
        await asyncio.sleep(0)
        assert not c.inflight and c.request_slots._value==128
    asyncio.run(run())


def test_cancelled_ingress_wait_does_not_leak_or_create_state(tmp_path):
    from online_coordinator import Coordinator
    async def run():
        c=Coordinator(tmp_path/"cancel.db","http://10.244.1.16:55581","http://10.244.2.32:55586")
        c.request_slots=asyncio.Semaphore(0)
        waiting=asyncio.create_task(c.submit("wait",[1],1))
        await asyncio.sleep(0);waiting.cancel()
        with pytest.raises(asyncio.CancelledError):await waiting
        assert not c.inflight and not c.tasks and c.request_slots._value==0
        c.failure="failed closed"
        retry=asyncio.create_task(c.submit("retry",[1],1))
        with pytest.raises(RuntimeError,match="failed closed"):await retry
    asyncio.run(run())


def test_routing_does_not_reserve_or_wait_for_device_capacity(tmp_path):
    from online_coordinator import Coordinator
    c=Coordinator(tmp_path/"route.db","http://10.244.1.16:55581","http://10.244.2.32:55586")
    c.admission=[Admission(100,1) for _ in range(4)]
    for a in c.admission:a.slots=0;a.free=0
    owners=[c.route_d(f"s{i}",4096) for i in range(8)]
    assert owners==[0,1,2,3,0,1,2,3]
    assert all(a.slots==0 and a.free==0 for a in c.admission)
    assert c.route_d("s0",4096)==0


def test_prefill_finishes_and_releases_p_before_waiting_for_d(tmp_path):
    from online_coordinator import Coordinator
    async def run():
        c=Coordinator(tmp_path/"queue.db","http://10.244.1.16:55581","http://10.244.2.32:55586")
        c.rank_private=True
        for a in c.admission:a.slots=0
        saved=asyncio.Event()
        class Peer:
            async def rpc(self,instance,op,**args):
                assert op=="generate"
                return dict(full_tokens=args["tokens"]+[7],cached=0)
        c.p=Peer()
        async def save(kind,index,tokens,salt,**kwargs):
            assert kind=="P" and kwargs["peer_group"]==c.d_affinity["queued"]
            saved.set()
            return "manifest",dict(tokens=tokens,salt=salt,key="key"),{}
        c.save=save
        loop=asyncio.get_running_loop()
        future=loop.create_future();generated=loop.create_future()
        task=asyncio.create_task(c.turn("queued",[1,2],2,future,generated,None))
        await asyncio.wait_for(saved.wait(),1)
        await asyncio.sleep(0)
        assert not task.done() and not generated.done()
        assert all(a.slots==16 for a in c.p_admission)
        assert all(a.slots==0 and a.free==a.blocks for a in c.admission)
        # Stop the test at the queued boundary without fabricating a D result.
        task.cancel()
        await task
    asyncio.run(run())


def test_full_128_generation_connections_do_not_starve_state_control(tmp_path,monkeypatch):
    from aiohttp import web
    import online_coordinator as module
    from pd_limits import context_limit
    class Peer:
        def __init__(self,url,*args):self.kind="P" if "1.16" in url else "D"
        async def health(self):
            kind=self.kind
            info=dict(state_wire="raw-v2",capacities=[dict(block_size=2048,
                max_requests=16 if kind=="P" else 32,free_blocks=2048)]*(1 if kind=="P" else 4))
            return dict(ready=True,kind=kind,context_limit=context_limit(),
                actors=[dict(alive=True,quarantined=False,info=info)]*(4 if kind=="P" else 1))
    monkeypatch.setattr(module,"Peer",Peer)
    async def run():
        release=asyncio.Event();full=asyncio.Event();entered=0
        async def hold(request):
            nonlocal entered
            entered+=1
            if entered==128:full.set()
            await release.wait()
            return web.Response(text="done")
        async def control(request):return web.Response(text="control")
        app=web.Application();app.router.add_get("/hold",hold);app.router.add_get("/control",control)
        runner=web.AppRunner(app);await runner.setup()
        site=web.TCPSite(runner,"127.0.0.1",0);await site.start()
        port=site._server.sockets[0].getsockname()[1]
        c=module.Coordinator(tmp_path/"connections.db","http://10.244.1.16:55581","http://10.244.2.32:55586")
        tasks=[]
        try:
            await c.start()
            assert c.request_slots._value==256  # two bounded waves at C32x4
            async def request(path):
                async with c.client.get(f"http://127.0.0.1:{port}/{path}") as response:
                    return await response.text()
            tasks=[asyncio.create_task(request("hold")) for _ in range(128)]
            await asyncio.wait_for(full.wait(),5)
            assert await asyncio.wait_for(request("control"),1)=="control"
            assert not any(t.done() for t in tasks)
        finally:
            release.set()
            await asyncio.gather(*tasks,return_exceptions=True)
            await c.close();await runner.cleanup()
    asyncio.run(run())



def test_store_releases_device_before_manifest_work_but_not_before_quorum(tmp_path):
    from online_coordinator import Coordinator
    async def run():
        c=Coordinator(tmp_path/"device-release.db","http://10.244.1.16:55581","http://10.244.2.32:55586")
        committed=asyncio.Event();metadata=asyncio.Event();released=[]
        async def prepare(kind,index,command):return 7
        async def cache(kind,index,command):
            if command["kind"]=="wait":
                await committed.wait()
                return dict(kind="store",cancelled=False,ranks=[0,1])
            assert command["kind"]=="describe"
            assert released==["device"]
            await metadata.wait()
            return dict(key="key",tokens=[1,2],salt="s")
        class Sink:
            def put(self,*args):pass
        c.prepare=prepare;c.cache=cache;c.sink=Sink()
        async def release():released.append("device")
        task=asyncio.create_task(c.save("D",0,[1,2],"s",on_device_released=release))
        await asyncio.sleep(0)
        assert not released
        committed.set()
        await asyncio.sleep(0)
        assert released==["device"] and not task.done()
        metadata.set();await task
    asyncio.run(run())


@pytest.mark.parametrize("receipt",[
    dict(kind="store",cancelled=False,ranks=[0]),
    dict(kind="store",cancelled=True,ranks=[0,1]),
    dict(kind="load",cancelled=False,ranks=[0,1]),
])
def test_missing_store_quorum_keeps_device_permit(tmp_path,receipt):
    from online_coordinator import Coordinator
    async def run():
        c=Coordinator(tmp_path/"no-release.db","http://10.244.1.16:55581","http://10.244.2.32:55586")
        async def prepare(*args):return 1
        async def cache(*args):return receipt
        async def release():raise AssertionError("must not release")
        c.prepare=prepare;c.cache=cache
        with pytest.raises(RuntimeError,match="quorum"):
            await c.save("D",0,[1,2],"s",on_device_released=release)
    asyncio.run(run())



def test_local_device_quorum_releases_permit_while_replication_is_pending(tmp_path):
    from online_coordinator import Coordinator
    async def run():
        c=Coordinator(tmp_path/"staged.db","http://10.244.1.16:55581","http://10.244.2.32:55586")
        c.d_device_release=True
        staged=asyncio.Event();committed=asyncio.Event();released=[]
        async def prepare(*args):return 7
        async def cache(kind,index,command):
            if command["kind"]=="wait_device":
                await staged.wait()
                return dict(operation=7,kind="store",device_released=True,ranks=[0,1])
            if command["kind"]=="wait":
                assert released==["device"]
                await committed.wait()
                return dict(kind="store",cancelled=False,ranks=[0,1])
            return dict(key="key",tokens=[1,2],salt="s")
        class Sink:
            def put(self,*args):pass
        c.prepare=prepare;c.cache=cache;c.sink=Sink()
        async def release():released.append("device")
        task=asyncio.create_task(c.save("D",0,[1,2],"s",on_device_released=release))
        await asyncio.sleep(0);assert not released
        staged.set();await asyncio.sleep(0)
        assert released==["device"] and not task.done()
        committed.set();await task
        assert released==["device"]
        assert validate("D",dict(instance=0,op="cache",args=dict(owner=0,
            command=dict(kind="wait_device",operation=7))))[1]=="cache"
    asyncio.run(run())



@pytest.mark.parametrize("action",["load","save"])
def test_busy_state_admission_does_not_lock_out_independent_session(tmp_path,action):
    from online_coordinator import Coordinator
    async def run():
        c=Coordinator(tmp_path/"independent.db","http://10.244.1.16:55581","http://10.244.2.32:55586")
        blocked=asyncio.Event();other=asyncio.Event()
        async def cache(kind,index,command):
            op=command["kind"]
            if op=="adopt":return True
            if op in ("load_match","store_match"):
                label=command.get("salt",command.get("key"))
                if label=="A" and not other.is_set():
                    blocked.set();return None
                if label=="B":other.set()
                return dict(already_resident=True) if op=="load_match" else 7
            if op=="wait":return dict(kind="store",cancelled=False,ranks=[0,1])
            if op=="describe":return dict(key=command["key"],tokens=[1,2],salt="s")
            raise AssertionError(op)
        class Sink:
            def put(self,*args):pass
        c.cache=cache;c.sink=Sink()
        async def request(label):
            if action=="load":return await c.load("D",0,dict(key=label))
            return await c.save("D",0,[1,2],label)
        first=asyncio.create_task(request("A"))
        await asyncio.wait_for(blocked.wait(),1)
        second=asyncio.create_task(request("B"))
        try:
            await asyncio.wait_for(asyncio.gather(first,second),.3)
            assert other.is_set()
        finally:
            for task in (first,second):task.cancel()
            await asyncio.gather(first,second,return_exceptions=True)
    asyncio.run(run())


def test_private_store_placement_joins_native_command_without_rpc():
    import ast
    from pathlib import Path
    from types import SimpleNamespace as NS
    tree=ast.parse(Path(__file__).with_name("online_entry.py").read_text())
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="_cache_command")
    namespace={}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),"cache-command","exec"),namespace)
    pending={}
    def store(seat,key):
        pending[7]=NS(command=dict(kind="store",key=key))
        return 7
    cache=NS(store=store,pending=pending)
    seat=NS(cache_salt="s",tokens=(1,2),owner=None,io_owner=None,fence=0,index=0)
    core=NS(state_cache=lambda command:None,
            scheduler=NS(cache_actions=cache,residents=NS(seats=[seat]),processed_step_seq=0),
            vllm_config=NS(additional_config=dict(pd_rank_private=True)))
    # No model_executor at all: placement must not perform a synchronous RPC.
    command=dict(kind="store_match",salt="s",tokens=[1,2],key="k",peer_group=3)
    assert namespace["_cache_command"](core,command)==7
    assert pending[7].command["peer_group"]==3
    for bad in (-1,4,True,"0"):
        with pytest.raises(ValueError,match="peer group"):
            namespace["_cache_command"](core,dict(command,peer_group=bad))


def test_worker_sets_private_peer_before_dispatching_store():
    import ast
    from pathlib import Path
    tree=ast.parse(Path(__file__).with_name("online_entry.py").read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="Worker")
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=="state_cache_actions")
    calls=[]
    class Base:
        def state_cache_actions(self,commands):
            calls.append(("dispatch",commands))
            return "enqueued"
        def pd_rank_peer(self,key,group):
            calls.append(("peer",key,group))
    cls.body=[method];cls.bases=[ast.Name(id="Base",ctx=ast.Load())]
    namespace={"Base":Base}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),"worker-command","exec"),namespace)
    worker=namespace["Worker"]()
    commands=[dict(kind="store",key="k",peer_group=2),dict(kind="load",key="other")]
    assert worker.state_cache_actions(commands)=="enqueued"
    assert calls==[("peer","k",2),("dispatch",commands)]
    with pytest.raises(ValueError,match="only to store"):
        worker.state_cache_actions([dict(kind="load",key="bad",peer_group=2)])


def test_owner_control_window_pipelines_four_rpcs_but_stays_bounded(tmp_path):
    from online_coordinator import Coordinator
    async def run():
        c=Coordinator(tmp_path/"window.db","http://10.244.1.16:55581","http://10.244.2.32:55586")
        active=0;peak=0;arrived=asyncio.Event();release=asyncio.Event()
        async def cache(kind,index,command):
            nonlocal active,peak
            active+=1;peak=max(peak,active)
            if active==4:arrived.set()
            try:
                await release.wait()
                return command["key"]
            finally:active-=1
        c.cache=cache
        tasks=[asyncio.create_task(c.prepare("D",0,dict(kind="load_match",key=i))) for i in range(5)]
        try:
            await asyncio.wait_for(arrived.wait(),.3)
            await asyncio.sleep(.01)
            assert active==peak==4 and not any(t.done() for t in tasks)
            tasks[0].cancel()
            await asyncio.gather(tasks[0],return_exceptions=True)
            await asyncio.sleep(.01)
            assert active==4  # cancelled attempt returns its permit
            release.set()
            assert await asyncio.gather(*tasks[1:])==[1,2,3,4]
            assert peak==4
        finally:
            for task in tasks:task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
    asyncio.run(run())


def test_new_p_route_balances_host_seats_and_pages_without_reserving(tmp_path):
    from online_coordinator import Coordinator
    c=Coordinator(tmp_path/"p-route.db","http://10.244.1.16:55581","http://10.244.2.32:55586")
    loads=[.1,.4,.8,.6]
    c.host_cache=SimpleNamespace(load=lambda kind,i:loads[i])
    c.p_admission=[Admission(100,16) for _ in range(4)]
    c.p_admission[0].slots=0  # Largest free host pool, but compute saturated.
    c.p_admission[1].free=0  # Seats free, but KV saturated.
    before=[(a.free,a.slots) for a in c.p_admission]
    assert c.route_p("new",4096)==3
    assert [(a.free,a.slots) for a in c.p_admission]==before
    loads[3]=1
    assert c.route_p("new",4096)==3  # Returning sessions never migrate.
    assert c.route_p("other",4096)==2
    c.p_admission[0].slots=16
    assert c.route_p("idle",4096)==0  # Host still decides with equal compute.
    with pytest.raises(ValueError,match="P KV budget"):
        c.route_p("oversized",2048*101)
