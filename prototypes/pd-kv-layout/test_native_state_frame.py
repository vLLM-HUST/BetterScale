import ctypes
from types import SimpleNamespace as S
import pytest
import torch
from native_state_frame import FramePlan
from native_dram_staging import NativeDramStaging
from test_native_dram_staging import Allocator, Store
from betterscale.live.runtime.host_state import (
    TorchHostStateBackend,HostStateKey,HostStateSelection,HostStateDomainSelection)
from betterscale.live.runtime.page_transport import decode_snapshot


def fixture():
    domain = object()
    states=[]
    for name,dtype,width in [("a",torch.bool,3),("b",torch.bfloat16,5),("c",torch.float32,7),("d",torch.int64,2)]:
        tensor=torch.arange(4*width).reshape(4,width).to(dtype)
        state=S(tensor=tensor,domain=domain,storage_dtype=dtype,block_shape=(width,),
                num_blocks=4,physical_blocks_per_logical_block=1,leading_physical_blocks=0,
                logical_block_bytes=width*tensor.element_size())
        states.append((name,state))
    return domain,states


def copy_cpu(args):
    for source,destination,size in zip(*[a.tolist() for a in args]):
        ctypes.memmove(destination,source,size)


def test_scattered_mixed_dtype_frame_has_no_payload_assembly_and_legacy_wire_parity():
    domain,states=fixture()
    plan=FramePlan.build([(n,s,(2,0)) for n,s in states])
    store=Store();allocator=Allocator()
    arena=NativeDramStaging(store,slot_bytes=plan.byte_length,slots=1,allocator=allocator)
    lease=arena.acquire(plan.byte_length)
    copy_cpu(plan.copy_descriptors(lease,to_host=True))
    wire=ctypes.string_at(lease.pointer,lease.size)  # test oracle only
    backend=TorchHostStateBackend(memory_budget_bytes=plan.payload_bytes)
    key=HostStateKey("frame",1)
    select=HostStateSelection((HostStateDomainSelection(domain,(1,3)),))
    decode_snapshot(wire,backend,key,states,select)
    for n,s in states:
        assert torch.equal(backend._snapshots[key].payloads[n].tensor,s.tensor[[2,0]])
    expected={n:s.tensor[[2,0]].clone() for n,s in states}
    for _,s in states:s.tensor.zero_()
    destination=FramePlan.build([(n,s,(1,3)) for n,s in states])
    copy_cpu(destination.copy_descriptors(lease,to_host=False))
    for n,s in states:assert torch.equal(s.tensor[[1,3]],expected[n])
    lease.seal(lambda:None);lease.discard();arena.close()


def test_frame_metadata_damage_is_rejected_before_h2d():
    _,states=fixture()
    plan=FramePlan.build([(n,s,(0,)) for n,s in states])
    arena=NativeDramStaging(Store(),slot_bytes=plan.byte_length,slots=1,allocator=Allocator())
    lease=arena.acquire(plan.byte_length)
    plan.copy_descriptors(lease,to_host=True)
    ctypes.memset(lease.pointer+5,0,1)
    with pytest.raises(ValueError,match="metadata"):
        plan.copy_descriptors(lease,to_host=False)
    lease.seal(lambda:None);lease.discard();arena.close()
