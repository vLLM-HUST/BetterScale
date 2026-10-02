import asyncio
from types import SimpleNamespace
import pytest
import online_coordinator as module
from online_node import validate
from pd_limits import context_limit


@pytest.mark.parametrize("peer", [None, 0, 3])
def test_store_peer_is_explicit_bounded_index(peer):
    c=dict(kind="store_match",key="cp",tokens=[1,2],salt="session",peer_group=peer)
    assert validate("P",dict(instance=2,op="cache",args=dict(owner=0,command=c)))[2]["command"]==c
    for bad in (-1,4,True,"2"):
        c["peer_group"]=bad
        with pytest.raises(ValueError):
            validate("P",dict(instance=2,op="cache",args=dict(owner=0,command=c)))


@pytest.mark.parametrize("wire",["rank-private-v1","rank-private-mtp-prefix-v1"])
def test_private_turns_publish_then_retire_both_sticky_copies_without_object_service(tmp_path,monkeypatch,wire):
    class Peer:
        def __init__(self,url,*args):
            self.kind="P" if "1.16" in url else "D"
        async def health(self):
            kind=self.kind
            info=dict(state_wire=wire,capacities=[
                dict(block_size=2048,max_requests=16,free_blocks=1044)]*(1 if kind=="P" else 4))
            return dict(ready=True,kind=kind,context_limit=context_limit(),
                actors=[dict(alive=True,quarantined=False,info=info)]*(4 if kind=="P" else 1))
        async def rpc(self,instance,op,**args):
            assert op=="generate"
            output=[7]*args["n"]
            return dict(full_tokens=args["tokens"]+output,token_ids=output,cached=0,
                        request_id="fake",start_ns=1,arrivals=[])
    monkeypatch.setattr(module,"Peer",Peer)
    async def run():
        c=module.Coordinator(tmp_path/"private.db","http://10.244.1.16:55581","http://10.244.2.32:55586")
        await c.start()
        def forbidden(*args):raise AssertionError("private mode touched legacy object service")
        c.sink=SimpleNamespace(get=forbidden,put=forbidden)
        copies={};operations=[];number=0
        async def prepare(kind,index,command):
            nonlocal number
            action=command["kind"]
            if action=="store_match":
                assert "peer_group" in command
                cp=dict(key=command["key"],tokens=command["tokens"],salt=command["salt"],
                        block_count=1,byte_length=100,pages=["page:"+command["key"]])
                copies[kind,index,cp["key"]]=cp
                number+=1
                operations.append((action,kind,index,command["peer_group"]))
                return number
            if action=="adopt":
                cp=command["checkpoint"]
                copies[kind,index,cp["key"]]=cp
                return True
            if action=="load_match":
                assert (kind,index,command["key"]) in copies
                return dict(already_resident=True)
            assert action=="drop"
            # The directory already points to a newer published generation.
            with c.directory.transaction() as db:
                row=db.execute("SELECT m.payload FROM sessions s JOIN rank_manifests m ON s.manifest=m.id").fetchone()
            import json
            assert json.loads(row[0])["key"]!=command["key"]
            del copies[kind,index,command["key"]]
            operations.append((action,kind,index,command["key"]))
            number+=1
            return number
        async def cache(kind,index,command):
            if command["kind"]=="wait":return dict(done=True)
            assert command["kind"]=="describe"
            return copies[kind,index,command["key"]]
        c.prepare=prepare;c.cache=cache
        try:
            first=await c.submit("sticky",[1,2],3)
            owners=(c.p_affinity["sticky"],c.d_affinity["sticky"])
            assert len(copies)==2
            second=await c.submit("sticky",first["full_tokens"]+[9],1)
            assert (c.p_affinity["sticky"],c.d_affinity["sticky"])==owners
            assert len(copies)==2  # P-only turn refreshes the existing sticky D replica too
            assert second["full_tokens"][-1]==7
            with c.directory.transaction() as db:
                assert db.execute("SELECT count(*) FROM rank_manifests").fetchone()[0]==1
            stores=[op for op in operations if op[0]=="store_match"]
            assert stores==[("store_match","P",owners[0],owners[1]),
                            ("store_match","D",owners[1],owners[0]),
                            ("store_match","P",owners[0],owners[1])]
        finally:await c.close()
    asyncio.run(run())


def test_memory_rpc_is_readonly_actor_wide_without_arbitrary_arguments():
    assert validate("D",dict(instance=0,op="memory",args=dict(owner=0)))[1]=="memory"
    for args in (dict(owner=True),dict(owner=1),dict(owner=0,reset=True)):
        with pytest.raises(ValueError):
            validate("D",dict(instance=0,op="memory",args=args))
