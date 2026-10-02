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
