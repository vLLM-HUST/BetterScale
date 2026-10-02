from types import SimpleNamespace as S
import pytest
import torch
from betterscale.live.runtime.host_state import HostStateKey, HostStateSelection, HostStateDomainSelection
from betterscale.live.runtime.page_transport import ObjectStateTransport

class Sink:
    def __init__(self): self.data={}; self.puts=[]
    def ensure(self,k): return k in self.data
    def put(self,k,v): self.data[k]=v; self.puts.append(k)
    def get(self,k): return self.data[k]

def fixture():
    domain=object()
    tensor=torch.arange(32,dtype=torch.float32).reshape(8,4)
    state=S(tensor=tensor,domain=domain,storage_dtype=tensor.dtype,
            block_shape=(4,),num_blocks=8,physical_blocks_per_logical_block=1,
            leading_physical_blocks=0,logical_block_bytes=16)
    select=lambda i: HostStateSelection((HostStateDomainSelection(domain,(i,)),))
    return [("target.fa.key",state)],tensor,select

def test_incremental_cross_worker_restore_and_bounded_staging():
    states,t,select=fixture(); sink=Sink()
    p=ObjectStateTransport(sink,"model/head0",staging_bytes=32)
    d=ObjectStateTransport(sink,"model/head0",staging_bytes=32,verify=True)
    first=t[:3].clone()
    tr=p.transfer(HostStateKey("A",1),states,{"sealed":select(0),"tail:A":select(1)},store=True,stream=None)
    tr.result(); assert tr.byte_length==32 and p.staging.committed_bytes==0
    tr=p.transfer(HostStateKey("B",1),states,{"sealed":select(0),"tail:B":select(2)},store=True,stream=None)
    tr.result(); assert tr.byte_length==16 and len(sink.puts)==3
    t[4:].zero_()
    tr=d.transfer(HostStateKey("B",1),states,{"sealed":select(4),"tail:B":select(5)},store=False,stream=None)
    tr.result(); assert tr.byte_length==32
    assert torch.equal(t[4],first[0]) and torch.equal(t[5],first[2])
    assert not t[6:].any() and d.staging.committed_bytes==0
    assert p.object_id("sealed") != ObjectStateTransport(sink,"model/head1").object_id("sealed")

@pytest.mark.parametrize("damage",["truncate","geometry"])
def test_wire_damage_rejected_before_destination_write(damage):
    states,t,select=fixture(); sink=Sink(); backend=ObjectStateTransport(sink,"model/head0")
    backend.transfer(HostStateKey("A",1),states,{"a":select(0)},store=True,stream=None).result()
    key=backend.object_id("a")
    sink.data[key]=sink.data[key][:-1] if damage=="truncate" else sink.data[key].replace(b"torch.float32",b"torch.float16")
    old=t.clone()
    with pytest.raises(ValueError):
        backend.transfer(HostStateKey("A",1),states,{"a":select(3)},store=False,stream=None).result()
    assert torch.equal(t,old) and backend.quarantined

def test_failed_publication_retains_staging_not_success():
    states,t,select=fixture(); sink=Sink()
    def fail(*args): raise IOError("replica acknowledgement lost")
    sink.put=fail
    backend=ObjectStateTransport(sink,"model/head0")
    with pytest.raises(IOError):
        backend.transfer(HostStateKey("A",1),states,{"a":select(0)},store=True,stream=None).result()
    assert backend.quarantined and backend.staging.committed_bytes==16


def test_standalone_restore_and_audit_wait_for_their_own_events(monkeypatch):
    states,t,select=fixture();sink=Sink();transport=ObjectStateTransport(sink,"model/head0",verify=True)
    transport.transfer(HostStateKey("A",1),states,{"a":select(0)},store=True,stream=None).result()
    expected=t[0].clone();t[3].zero_();waits=[]
    backend=transport.restore_backend
    original=backend._enqueue_copies
    class Deferred:
        def __init__(self,copy):self.copy=copy
        def synchronize(self):waits.append(True);self.copy()
    def enqueue(lanes,payloads,*,to_host,stream):
        return Deferred(lambda:original(lanes,payloads,to_host=to_host))
    monkeypatch.setattr(backend,"_copy_lanes",enqueue)
    transport.transfer(HostStateKey("A",1),states,{"a":select(3)},store=False,stream=None).result()
    assert torch.equal(t[3],expected) and len(waits)==2
    assert backend.committed_bytes==0 and not backend._inflight


def test_independent_transfers_have_bounded_exclusive_staging_lanes():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier,Event
    states,t,select=fixture();sink=Sink();entered=Barrier(3);release=Event()
    original=sink.put;calls=[]
    def put(key,data):
        calls.append(key)
        if len(calls)<=2:
            entered.wait(timeout=3);assert release.wait(timeout=3)
        original(key,data)
    sink.put=put
    transport=ObjectStateTransport(sink,"model/head0",staging_bytes=16,max_transfers=2)
    transfers=[transport.transfer(HostStateKey(str(i),1),states,
        {str(i):select(i)},store=True,stream=None) for i in range(3)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(x.result) for x in transfers]
        entered.wait(timeout=3)
        assert transport.active_transfers==2 and transport.peak_transfers==2
        assert transport.staging_budget_bytes==32 and not any(f.done() for f in futures)
        release.set()
        for f in futures:f.result(timeout=3)
    assert [x.byte_length for x in transfers]==[16]*3
    assert transport.active_transfers==0 and len(transport.available)==2
    assert all(a.committed_bytes==b.committed_bytes==0 for a,b in transport.available)
    expected=t[:3].clone();t[4:7].zero_()
    with ThreadPoolExecutor(max_workers=3) as pool:
        fs=[pool.submit(transport.transfer(HostStateKey(str(i),1),states,
            {str(i):select(i+4)},store=False,stream=None).result) for i in range(3)]
        for f in fs:f.result(timeout=3)
    assert torch.equal(t[4:7],expected)


def test_staging_failure_wakes_queued_transfers_without_reusing_uncertain_storage():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    states,t,select=fixture();sink=Sink();entered=Event();release=Event()
    def fail(*args):
        entered.set();assert release.wait(timeout=3);raise IOError("lost acknowledgement")
    sink.put=fail
    transport=ObjectStateTransport(sink,"model/head0",staging_bytes=16,max_transfers=1)
    first=transport.transfer(HostStateKey("A",1),states,{"A":select(0)},store=True,stream=None)
    second=transport.transfer(HostStateKey("B",1),states,{"B":select(1)},store=True,stream=None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a=pool.submit(first.result);assert entered.wait(timeout=3)
        b=pool.submit(second.result);release.set()
        with pytest.raises(IOError):a.result(timeout=3)
        with pytest.raises(RuntimeError,match="quarantined"):b.result(timeout=3)
    assert transport.quarantined and not transport.available
    assert transport.staging.committed_bytes==16


def test_one_transfer_result_is_idempotent_under_concurrent_waiters():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from betterscale.live.runtime.page_transport import ObjectTransfer
    entered=Event();release=Event();calls=[]
    def run():
        calls.append(True);entered.set();assert release.wait(timeout=3);return 42
    transfer=ObjectTransfer(run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a=pool.submit(transfer.result);assert entered.wait(timeout=3)
        b=pool.submit(transfer.result);release.set()
        a.result(timeout=3);b.result(timeout=3)
    assert calls==[True] and transfer.byte_length==42 and transfer.finished


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16,
                                   torch.int32, torch.int64, torch.bool, torch.uint8])
def test_wire_codec_preserves_dtype_bytes_and_owns_destination(dtype):
    from betterscale.live.runtime.page_transport import encode_snapshot, decode_snapshot
    from betterscale.live.runtime.host_state import TorchHostStateBackend
    import json, struct
    states, _, select = fixture()
    state = states[0][1]
    state.tensor = torch.arange(32).reshape(8, 4).to(dtype)
    state.storage_dtype = dtype
    state.logical_block_bytes = 4 * state.tensor.element_size()
    backend = TorchHostStateBackend(memory_budget_bytes=1024)
    key = HostStateKey("codec", 1)
    backend.offload(states, key, select(2), stream=None).result()
    original = backend._snapshots[key]
    data = encode_snapshot(original)
    header_size = struct.unpack_from("<I", data)[0]
    assert json.loads(data[4:4+header_size])["schema"] == 1
    assert data[4+header_size:] == state.tensor[2].view(torch.uint8).numpy().tobytes()
    backend.release(key)
    mutable = bytearray(data)
    decode_snapshot(mutable, backend, key, states, select(7))
    mutable[:] = b"\0" * len(mutable)
    assert encode_snapshot(backend._snapshots[key]) == data
