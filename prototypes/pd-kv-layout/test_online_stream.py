import asyncio
import json
from types import SimpleNamespace as NS
import pytest
from online_stream import Progress
from online_coordinator import Coordinator
from session import Conflict


def test_progress_bound_never_blocks_producer():
    async def run():
        p=Progress(2);p.push(1);p.push(2);p.push(3)
        with pytest.raises(RuntimeError,match="bounded"):await p.read()
        p.detach();p.push(4)
        assert not p.items
    asyncio.run(run())


def test_output_precedes_backup_and_following_turn_waits_for_commit(tmp_path):
    async def run():
        c=Coordinator(tmp_path/"directory.db","http://10.244.1.16:55581","http://10.244.2.32:55586")
        entered=asyncio.Event();release=asyncio.Event();objects={};calls=[]
        async def rpc(instance,op,**args):
            calls.append(list(args["tokens"]))
            ids=[7]*args["n"]
            return dict(full_tokens=args["tokens"]+ids,token_ids=ids,cached=0,
                request_id="fake",start_ns=0,arrivals=[])
        c.p=c.d=NS(rpc=rpc)
        async def save(kind,index,tokens,salt):
            if kind=="D" and not release.is_set():entered.set();await release.wait()
            key=str(len(objects))
            cp=dict(tokens=tokens,salt=salt)
            objects[key]=json.dumps(cp).encode()
            return key,cp,{}
        async def load(*args):return {}
        c.save=save;c.load=load;c.sink=NS(get=objects.__getitem__)
        result=await asyncio.wait_for(c.submit("session",[1,2],3,output_ready=True),1)
        await entered.wait()
        assert result["token_ids"]==[7]*3 and not c.inflight["session"].done()
        following=asyncio.create_task(c.submit("session",result["full_tokens"]+[3],2,output_ready=True))
        await asyncio.sleep(0)
        assert not following.done() and len(calls)==2
        release.set()
        assert len((await asyncio.wait_for(following,1))["token_ids"])==2
        await asyncio.gather(*c.tasks)
        assert c.failure is None
    asyncio.run(run())


def test_overlapping_generation_remains_conflict(tmp_path):
    async def run():
        c=Coordinator(tmp_path/"directory.db","http://10.244.1.16:55581","http://10.244.2.32:55586")
        loop=asyncio.get_running_loop()
        c.inflight["session"]=loop.create_future();c.generated["session"]=loop.create_future()
        with pytest.raises(Conflict):await c.submit("session",[1],1,output_ready=True)
    asyncio.run(run())


def test_exact_frontend_validation_allows_real_budgets():
    from online_frontend import validate
    body=dict(model="m",prompt=[1,2],max_tokens=1024,cache_salt="s",
        stream=True,ignore_eos=True,return_token_ids=True,temperature=0,
        stream_options={"include_usage":True})
    assert validate(body,"m")[2]==1024
    with pytest.raises(ValueError):validate(dict(body,prompt=[True]),"m")
    with pytest.raises(ValueError):validate(dict(body,max_tokens=262144),"m")
