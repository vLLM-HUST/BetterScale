import ctypes
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import pytest
import torch
from rank_state_pool import RankStatePool
from rank_state_transport import RankStateTransport
from test_native_state_frame import fixture,copy_cpu
from betterscale.live.runtime.host_state import HostStateSelection,HostStateDomainSelection


def setup(replicate=lambda *args:None):
    domain,states=fixture()
    select=lambda i:HostStateSelection((HostStateDomainSelection(domain,(i,)),))
    pool=RankStatePool(("D",0,0),1<<20,lambda n:torch.empty(n,dtype=torch.uint8))
    transport=RankStateTransport(pool,"test",lambda args,*_:lambda:copy_cpu(args),replicate,verify=True)
    return states,pool,transport,select


def test_local_seal_releases_device_before_peer_and_audit_preserves_cache():
    proceed=Event();staged=Event()
    def replica(*args):assert proceed.wait(3)
    states,pool,t,select=setup(replica)
    expected={n:s.tensor[[0,1]].clone() for n,s in states}
    transfer=t.transfer("A",states,{"a":select(0),"b":select(1)},store=True,
                        stream=None,on_staged=lambda _:staged.set())
    with ThreadPoolExecutor(1) as worker:
        result=worker.submit(transfer.result)
        assert staged.wait(3) and not result.done()
        for _,s in states:s.tensor.zero_()
        proceed.set();result.result(3)
    with pool.read(t.object_id("a")) as reader:
        before=ctypes.string_at(reader.pointer,reader.size)
        t.transfer("A",states,{"a":select(2)},store=False,stream=None,manifest=("a","b")).result()
        assert ctypes.string_at(reader.pointer,reader.size)==before
    t.transfer("A",states,{"b":select(3)},store=False,stream=None,manifest=("a","b")).result()
    for n,s in states:assert torch.equal(s.tensor[[2,3]],expected[n])
    t.release("A");pool.close()


def test_sparse_manifest_retains_all_pages_and_reuses_sealed_identity():
    states,pool,t,select=setup()
    t.transfer("A",states,{"a":select(0),"b":select(1)},store=True,stream=None).result()
    staged=[]
    t.transfer("B",states,{},store=True,stream=None,manifest=("a","b"),on_staged=staged.append).result()
    assert staged==[0]
    t.release("A")
    assert pool.complete("B") and len(pool.groups["B"])==2
    t.release("B");pool.close()


def test_missing_object_never_signals_ready():
    states,pool,t,select=setup()
    ready=[]
    tr=t.transfer("bad",states,{"a":select(0)},store=True,stream=None,manifest=("a","missing"),on_staged=ready.append)
    with pytest.raises(ValueError,match="missing source"):tr.result()
    assert not ready and not pool.objects
    t.release("bad");pool.close()


def test_partial_enqueue_failure_keeps_source_reader_pinned():
    states,pool,t,select=setup()
    t.transfer("A",states,{"a":select(0)},store=True,stream=None).result()
    def fail(*args):raise RuntimeError("uncertain enqueue")
    t.submit=fail
    tr=t.transfer("A",states,{"a":select(2)},store=False,stream=None)
    for _ in range(2):
        with pytest.raises(RuntimeError,match="uncertain"):tr.result()
    assert len(t.quarantined)==1
    t.release("A")
    with pytest.raises(RuntimeError):pool.close()
    assert pool.objects[t.object_id("a")].readers==1


def test_completed_audit_scratch_is_not_held_into_next_object_allocation():
    import weakref
    states,pool,t,select=setup()
    t.transfer("A",states,{"a":select(0),"b":select(1)},store=True,stream=None).result()
    refs=[weakref.ref(item.buffer) for item in pool.objects.values()]
    allocate=pool.allocate
    def bounded(size):
        assert sum(ref() is not None for ref in refs)==2
        result=allocate(size);refs.append(weakref.ref(result));return result
    pool.allocate=bounded
    t.transfer("A",states,{"a":select(2),"b":select(3)},store=False,stream=None).result()
    assert sum(ref() is not None for ref in refs)==2
    t.release("A");pool.close()
