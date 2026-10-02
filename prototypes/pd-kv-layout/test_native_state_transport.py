import ctypes
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import pytest
import torch
from native_state_frame import FramePlan
from native_state_transport import NativeStateTransport,CheckedReplicas
from native_dram_staging import NativeDramStaging
from native_store_leases import NativeStoreLeases
from test_native_dram_staging import Allocator
from test_native_state_frame import fixture,copy_cpu
from betterscale.live.runtime.host_state import HostStateSelection,HostStateDomainSelection


class Sink:
    def __init__(self):
        self.values={};self.allow=Event()
    def register_buffer(self,*args):return 0
    def unregister_buffer(self,*args):return 0
    def batch_is_exist(self,keys):return [int(k in self.values) for k in keys]
    def put_from(self,key,pointer,size,config):
        assert self.allow.wait(3)
        self.values[key]=ctypes.string_at(pointer,size);return 0
    def get_into(self,key,pointer,size):
        data=self.values[key]
        assert len(data)==size
        ctypes.memmove(pointer,data,size);return size


def setup():
    domain,states=fixture();sink=Sink()
    select=lambda i:HostStateSelection((HostStateDomainSelection(domain,(i,)),))
    size=FramePlan.build([(n,s,(0,)) for n,s in states]).byte_length+32
    leases=NativeStoreLeases(lambda keys:set(keys)&set(sink.values),ttl_seconds=60,background=False)
    arena=NativeDramStaging(CheckedReplicas(sink,None,leases),slot_bytes=size,slots=2,allocator=Allocator())
    def submit(args,to_host,stream):return lambda:copy_cpu(args)
    transport=NativeStateTransport(sink,arena,"test",submit,leases=leases,verify=True)
    return states,sink,arena,transport,select


def test_local_completion_precedes_both_acks_and_source_reuse_is_safe():
    states,sink,arena,transport,select=setup()
    expected={n:s.tensor[[0,1]].clone() for n,s in states}
    staged=Event()
    transfer=transport.transfer("A",states,{"a":select(0),"b":select(1)},
        store=True,stream=None,on_staged=lambda size:staged.set())
    with ThreadPoolExecutor(1) as worker:
        final=worker.submit(transfer.result)
        assert staged.wait(3) and not final.done()
        # Scheduler may now recycle every source lane; snapshots remain exact.
        for _,s in states:s.tensor.zero_()
        sink.allow.set();final.result(3)
    transport.transfer("A",states,{"a":select(2),"b":select(3)},
        store=False,stream=None).result()
    for n,s in states:assert torch.equal(s.tensor[[2,3]],expected[n])
    # A subsequent sealed identity hit performs no device copies.
    skipped=[]
    transport.transfer("B",states,{"a":select(0)},store=True,stream=None,
        on_staged=skipped.append).result()
    assert skipped==[0]
    transport.release("A");transport.release("B")
    assert not transport.leases.references
    transport.leases.close();arena.close()


def test_corrupt_body_is_rejected_before_any_device_write_and_failure_is_sticky():
    states,sink,arena,transport,select=setup();sink.allow.set()
    transport.transfer("A",states,{"a":select(0)},store=True,stream=None).result()
    key=transport.object_id("a")
    raw=bytearray(sink.values[key]);raw[-33]^=1;sink.values[key]=bytes(raw)
    expected={n:s.tensor.clone() for n,s in states}
    transfer=transport.transfer("A",states,{"a":select(2)},store=False,stream=None)
    for _ in range(2):
        with pytest.raises(RuntimeError,match="checksum"):transfer.result()
    for n,s in states:assert torch.equal(s.tensor,expected[n])
    assert len(transport.quarantined)==1
    with pytest.raises(RuntimeError):arena.close()
    arena.executor.shutdown(wait=True)


def test_sparse_device_hits_do_not_shrink_complete_host_manifest():
    states,sink,arena,transport,select=setup();sink.allow.set()
    expected={n:s.tensor[[0,1]].clone() for n,s in states}
    transport.transfer("A",states,{"a":select(0),"b":select(1)},store=True,stream=None).result()
    for _,s in states:s.tensor.zero_()
    for identity,destination in (("a",2),("b",3)):
        transport.transfer("A",states,{identity:select(destination)},store=False,
            stream=None,manifest=("a","b")).result()
        assert len(transport.leases.groups["A"])==2
    for n,s in states:assert torch.equal(s.tensor[[2,3]],expected[n])
    transport.release("A");transport.leases.close();arena.close()


def test_incomplete_export_never_emits_local_ready():
    states,sink,arena,transport,select=setup();sink.allow.set()
    ready=[]
    transfer=transport.transfer("bad",states,{"a":select(0)},store=True,stream=None,
        manifest=("a","missing"),on_staged=ready.append)
    with pytest.raises(ValueError,match="missing source"):transfer.result()
    assert not ready and not arena.active and not sink.values
    transport.release("bad");transport.leases.close();arena.close()
