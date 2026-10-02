import ctypes
import hashlib
import pytest
import torch
from rank_state_pool import RankStatePool
from rank_replica_receiver import RankReplicaReceiver


class Engine:
    def __init__(self):self.registered={}
    def register_memory(self,pointer,size):
        assert pointer not in self.registered
        self.registered[pointer]=size;return 0
    def unregister_memory(self,pointer):
        del self.registered[pointer];return 0


def setup(size=1024):
    pool=RankStatePool(("P",2,1),size,lambda n:torch.empty(n,dtype=torch.uint8))
    engine=Engine()
    return pool,engine,RankReplicaReceiver(pool,engine)


def prepare(receiver,cp,oid,value,op):
    data=bytes([value])*32
    result=receiver.prepare(("D",0,1),cp,oid,32,hashlib.sha256(data).hexdigest(),op)
    return data,result


def test_independent_transactions_do_not_wait_for_owner_commit():
    pool,engine,r=setup()
    r.begin("A",("a",));r.begin("B",("b",))
    a,first=prepare(r,"A","a",3,"a"*32)
    b,second=prepare(r,"B","b",7,"b"*32)
    assert first["status"]==second["status"]=="write"
    assert len(engine.registered)==2
    ctypes.memmove(second["pointer"],b,32)
    assert r.commit("b"*32) and pool.complete("B") and not pool.complete("A")
    with pytest.raises(RuntimeError):r.drop("A")
    ctypes.memmove(first["pointer"],a,32)
    assert r.commit("a"*32) and r.commit("a"*32)
    assert not engine.registered
    with pool.read("a") as reader:assert ctypes.string_at(reader.pointer,32)==a
    r.drop("A");r.drop("B");pool.close()


def test_corruption_is_never_published_or_recycled():
    pool,engine,r=setup();r.begin("A",("a",))
    data,target=prepare(r,"A","a",3,"a"*32)
    ctypes.memmove(target["pointer"],bytes([9])*32,32)
    with pytest.raises(RuntimeError,match="checksum"):r.commit("a"*32)
    assert not pool.complete("A") and engine.registered
    with pytest.raises(RuntimeError):pool.read("a")
    with pytest.raises(RuntimeError):pool.close()
    assert r.failure and len(r.quarantined)==1


def test_shard_identity_and_operation_fences():
    pool,engine,r=setup();r.begin("A",("a",))
    digest=hashlib.sha256(bytes(32)).hexdigest()
    with pytest.raises(ValueError,match="TP shard"):
        r.prepare(("D",0,0),"A","a",32,digest,"a"*32)
    _,target=prepare(r,"A","a",0,"a"*32)
    assert r.prepare(("D",0,1),"A","a",32,digest,"a"*32)==target
    with pytest.raises(ValueError,match="identity changed"):
        r.prepare(("D",0,1),"A","a",31,digest,"a"*32)
    ctypes.memset(target["pointer"],0,32);r.commit("a"*32)
    r.drop("A");pool.close()


def test_shared_page_new_manifest_reuses_immutable_object_without_rehash():
    pool,engine,r=setup();r.begin("A",("a",))
    data,target=prepare(r,"A","a",3,"a"*32)
    ctypes.memmove(target["pointer"],data,32);r.commit("a"*32)
    r.begin("B",("a",))
    def forbidden(*args):raise AssertionError("rehashing sealed historical State")
    r.digest=forbidden
    assert prepare(r,"B","a",3,"b"*32)[1]==dict(status="complete")
    with pytest.raises(ValueError,match="collision"):prepare(r,"B","a",4,"c"*32)
    r.drop("A");assert pool.complete("B")
    r.drop("B");pool.close()


def test_capacity_pressure_is_retryable_without_poisoning_rank():
    pool,engine,r=setup(32)
    r.begin("A",("a",))
    data,a=prepare(r,"A","a",3,"a"*32)
    ctypes.memmove(a["pointer"],data,32);r.commit("a"*32)
    r.begin("B",("b",))
    assert prepare(r,"B","b",4,"b"*32)[1]==dict(status="busy")
    assert r.failure is None and not r.pending
    r.drop("A")
    data,b=prepare(r,"B","b",4,"b"*32)
    ctypes.memmove(b["pointer"],data,32);r.commit("b"*32)
    r.drop("B");pool.close()


def test_commit_receipt_survives_unrelated_traffic_until_checkpoint_drop():
    pool,engine,r=setup()
    r.begin("A",("a",))
    data,target=prepare(r,"A","a",3,"a"*32)
    ctypes.memmove(target["pointer"],data,32);r.commit("a"*32)
    for i in range(260):
        key=str(i);op=f"{i:032x}"
        r.begin(key,(key,))
        data,target=prepare(r,key,key,7,op)
        ctypes.memmove(target["pointer"],data,32);r.commit(op);r.drop(key)
    assert r.commit("a"*32)
    assert len(r.completed)==1
    r.drop("A");assert not r.completed
    pool.close()
