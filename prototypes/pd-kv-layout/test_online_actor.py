import asyncio
from collections import deque
import multiprocessing as mp
from online_actor import Actor


def test_receive_batch_is_bounded_and_preserves_order():
    class Pipe:
        def __init__(self):self.rows=deque(range(130))
        def recv(self):return self.rows.popleft()
        def poll(self):return bool(self.rows)
    actor=Actor.__new__(Actor);actor.pipe=Pipe()
    assert actor.receive_batch()==list(range(64))
    assert actor.receive_batch()==list(range(64,128))
    assert actor.receive_batch()==[128,129]


def test_burst_demultiplexing_and_eof_delivers_prior_replies():
    async def run():
        actor=Actor.__new__(Actor)
        actor.pipe,writer=mp.Pipe(duplex=False)
        loop=asyncio.get_running_loop()
        first=loop.create_future();second=loop.create_future()
        actor.pending={"a":first,"b":second};seen=[]
        actor.progress={"a":lambda v:seen.append(("a",v)),
                        "b":lambda v:seen.append(("b",v))}
        actor.quarantined=False
        writer.send(("a","tokens",{"token_ids":[1]}))
        writer.send(("b","tokens",{"token_ids":[2,3]}))
        writer.send(("b","ok","second"))
        writer.send(("a","ok","first"))
        writer.close()
        await asyncio.wait_for(actor.receive(),1)
        assert await first=="first" and await second=="second"
        assert [(tag,v["token_ids"]) for tag,v in seen]==[("a",[1]),("b",[2,3])]
        assert all(v["node_arrival_ns"]>0 for _,v in seen)
        assert actor.quarantined and not actor.pending and not actor.progress
        actor.pipe.close()
    asyncio.run(run())


def test_error_in_batch_quarantines_remaining_callers():
    async def run():
        actor=Actor.__new__(Actor);actor.pipe,writer=mp.Pipe(duplex=False)
        loop=asyncio.get_running_loop()
        actor.pending={tag:loop.create_future() for tag in ("a","b")}
        futures=list(actor.pending.values());actor.progress={};actor.quarantined=False
        writer.send(("a","error","failure"));writer.close()
        await asyncio.wait_for(actor.receive(),1)
        assert actor.quarantined and not actor.pending
        assert all(isinstance(f.exception(),RuntimeError) for f in futures)
        actor.pipe.close()
    asyncio.run(run())


def test_actor_rpc_envelope_scales_from_native_capacity():
    from online_actor import rpc_capacity
    import pytest
    assert rpc_capacity([{"max_requests":16}]) == 160
    for width in (16,32,48,64,80):
        count=rpc_capacity([{"max_requests":width}]*4)
        assert count == max(160, 8*width+32)
        assert count >= 4*width+32
    for bad in ([],[{"max_requests":True}],[{"max_requests":81}],
                [{"max_requests":16}]*3):
        with pytest.raises(ValueError):rpc_capacity(bad)


def test_wide_actor_accepts_control_above_old_ceiling_but_is_bounded():
    from types import SimpleNamespace
    from online_actor import rpc_capacity
    import pytest
    async def run():
        actor=Actor.__new__(Actor)
        actor.quarantined=False
        actor.process=SimpleNamespace(is_alive=lambda:True)
        actor.max_calls=rpc_capacity([{"max_requests":80}]*4)
        actor.pending={str(i):None for i in range(320)}
        actor.progress={}
        def reply(message):
            tag,op,args=message
            assert op=="cache"
            actor.pending.pop(tag).set_result("control completed")
        actor.pipe=SimpleNamespace(send=reply)
        assert await actor.call("cache",{})=="control completed"
        actor.pending={str(i):None for i in range(actor.max_calls)}
        with pytest.raises(RuntimeError,match="bound exceeded"):
            await actor.call("cache",{})
    asyncio.run(run())


def test_private_host_budget_is_per_p_group_and_d_stays_scalar(monkeypatch):
    from online_actor import host_budget_gib
    import pytest
    monkeypatch.delenv("BETTERSCALE_PD_STATE_HOST_GIB", raising=False)
    monkeypatch.delenv("BETTERSCALE_PD_P_STATE_HOST_GIB", raising=False)
    assert host_budget_gib("P", 3) == 128
    monkeypatch.setenv("BETTERSCALE_PD_STATE_HOST_GIB", "24")
    assert host_budget_gib("P", 0) == 24
    monkeypatch.setenv("BETTERSCALE_PD_P_STATE_HOST_GIB", "[32,32,12,48]")
    assert [host_budget_gib("P", i) for i in range(4)] == [32,32,12,48]
    assert host_budget_gib("D", 0) == 24
    for bad in ('[32,32,12]', '[32,32,12,0]', '[32,32,12,129]',
                '[32,32,12,true]', '[32,32,12,48.0]', '{}', 'invalid'):
        monkeypatch.setenv("BETTERSCALE_PD_P_STATE_HOST_GIB", bad)
        with pytest.raises(ValueError):host_budget_gib("P", 0)
        assert host_budget_gib("D", 0) == 24
    for role, index in (("D", 1), ("P", 4), ("P", -1), ("P", True), ("X", 0)):
        with pytest.raises(ValueError):host_budget_gib(role, index)


def test_fatal_startup_reply_preserves_error_and_owned_group():
    from types import SimpleNamespace
    import pytest
    async def run():
        actor=Actor.__new__(Actor);actor.group_ready=False
        actor.pipe=SimpleNamespace(recv=lambda:("fatal","error","allocation207001"))
        with pytest.raises(RuntimeError,match="allocation207001"):await actor.ready()
        assert actor.group_ready
    asyncio.run(run())


def test_abort_signals_only_owned_process_until_private_group_exists(monkeypatch):
    import online_actor,signal
    from types import SimpleNamespace
    calls=[];actor=Actor.__new__(Actor);actor.group_ready=False
    actor.process=SimpleNamespace(pid=101,is_alive=lambda:True,
        terminate=lambda:calls.append("terminate"),kill=lambda:calls.append("kill"))
    monkeypatch.setattr(online_actor.os,"getpgid",lambda pid:999)
    monkeypatch.setattr(online_actor.os,"killpg",lambda pid,sig:calls.append((pid,sig)))
    actor.signal_owned(signal.SIGTERM)
    assert calls==["terminate"] and not actor.group_ready
    monkeypatch.setattr(online_actor.os,"getpgid",lambda pid:101)
    actor.signal_owned(signal.SIGTERM)
    assert calls[-1]==(101,signal.SIGTERM) and actor.group_ready
    actor.process.is_alive=lambda:False
    actor.signal_owned(signal.SIGKILL)
    assert calls[-1]==(101,signal.SIGKILL)  # descendants after actor exit


def _owned_group_fixture(pipe):
    import os,subprocess,sys,time
    os.setsid()
    child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])
    pipe.send((os.getpid(),child.pid))
    time.sleep(60)


def test_unready_actor_close_reaps_its_real_owned_process_group():
    import os,time,psutil
    context=mp.get_context("spawn");parent,child=context.Pipe()
    process=context.Process(target=_owned_group_fixture,args=(child,))
    process.start();child.close()
    assert parent.poll(15)
    root,descendant=parent.recv();assert root==process.pid and os.getpgid(root)==root
    actor=Actor.__new__(Actor);actor.process=process;actor.pipe=parent
    actor.info=None;actor.reader=None;actor.quarantined=True;actor.group_ready=True
    try:
        asyncio.run(asyncio.wait_for(actor.close(),25))
        assert not process.is_alive()
        asyncio.run(actor.close())  # repeated app cleanup is a no-op
        deadline=time.monotonic()+5
        while psutil.pid_exists(descendant) and time.monotonic()<deadline:
            if psutil.Process(descendant).status()==psutil.STATUS_ZOMBIE:break
            time.sleep(.01)
        assert not psutil.pid_exists(descendant) or psutil.Process(descendant).status()==psutil.STATUS_ZOMBIE
    finally:
        import signal
        try:os.killpg(root,signal.SIGKILL)
        except ProcessLookupError:pass
        process.join(5);parent.close()


def test_worker_constructor_failure_sends_fatal_before_pipe_close(monkeypatch):
    import online_actor
    from types import SimpleNamespace
    import pytest
    events=[]
    monkeypatch.setattr(online_actor.os,"setsid",lambda:events.append("setsid"))
    async def fail(*args):raise RuntimeError("allocation207001")
    monkeypatch.setattr(online_actor,"run",fail)
    pipe=SimpleNamespace(send=lambda value:events.append(value),close=lambda:events.append("close"))
    with pytest.raises(RuntimeError,match="allocation207001"):
        online_actor.worker("P",2,pipe)
    assert events[0]=="setsid" and events[-1]=="close"
    assert events[1][:2]==("starting","starting")
    assert events[2][:2]==("fatal","error") and "allocation207001" in events[2][2]


def test_native_crash_after_start_handshake_retains_group_cleanup_identity():
    from types import SimpleNamespace
    import pytest
    async def run():
        actor=Actor.__new__(Actor);actor.group_ready=False
        actor.process=SimpleNamespace(pid=123)
        replies=iter([("starting","starting",123)])
        def recv():
            try:return next(replies)
            except StopIteration:raise EOFError("native crash")
        actor.pipe=SimpleNamespace(recv=recv)
        with pytest.raises(EOFError,match="native crash"):await actor.ready()
        assert actor.group_ready
    asyncio.run(run())
