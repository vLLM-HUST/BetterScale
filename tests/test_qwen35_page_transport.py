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
