"""Actual host-State backend, TP2 Qwen35 lane shapes, no model/Store/codec."""
import argparse
import json
import statistics
import time
from types import SimpleNamespace as NS
from pathlib import Path
import torch
import torch_npu
from betterscale.live.runtime.host_state import (
    TorchHostStateBackend,HostStateKey,HostStateSelection,HostStateDomainSelection)


def qwen35_lanes(device_index, resident_count=8, page_count=128):
    device=f"npu:{device_index}";resident=object();pages=object();states=[]
    def lane(name,shape,dtype,domain,count,span=1):
        tensor=torch.empty((count*span,*shape),dtype=dtype,device=device)
        for i in range(count):
            tensor[i*span:(i+1)*span].fill_(len(states)+1+i)
        states.append((name,NS(tensor=tensor,domain=domain,storage_dtype=dtype,
            block_shape=shape,num_blocks=count,physical_blocks_per_logical_block=span,
            leading_physical_blocks=0,
            logical_block_bytes=span*__import__("math").prod(shape)*tensor.element_size())))
    for i in range(30):
        lane(f"target.gdn{i}.conv",(5,4096),torch.bfloat16,resident,resident_count)
        lane(f"target.gdn{i}.recurrent",(16,128,128),torch.float32,resident,resident_count,3)
    for i in range(10):
        for part in ("key","value"):lane(f"target.fa{i}.{part}",(128,1,256),torch.bfloat16,pages,page_count)
    for name in ("selection","conv_selection","remaining_outputs"):
        lane(name,(),torch.int32,resident,resident_count)
    return states,resident,pages


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--fragmented",action="store_true")
    p.add_argument("--batch",action="store_true")
    args=p.parse_args()
    torch.set_num_threads(8);torch.npu.set_device(args.device)
    if args.batch:
        import vllm_ascend.vllm_ascend_C
        from betterscale.models.qwen35.state_dma import enqueue
    states,resident,pages=qwen35_lanes(args.device)
    def selection(seat):
        ids=tuple(seat+8*k for k in range(16)) if args.fragmented else tuple(range(seat*16,(seat+1)*16))
        return HostStateSelection((HostStateDomainSelection(resident,(seat,)),HostStateDomainSelection(pages,ids)))
    streams=[torch.npu.Stream(device=args.device) for _ in range(2)]
    store=TorchHostStateBackend(memory_budget_bytes=1<<30, enqueue_copies=enqueue if args.batch else None)
    cached=TorchHostStateBackend(memory_budget_bytes=1<<30, enqueue_copies=enqueue if args.batch else None)
    keys=[HostStateKey(f"seed-{i}",1) for i in range(4)]
    seeds=[cached.offload(states,key,selection(i),stream=streams[0]) for i,key in enumerate(keys)]
    for x in seeds:x.result()
    logical_bytes=seeds[0].byte_length;rows=[];sequence=0
    for mode in ("d2h","h2d","duplex"):
        samples=[]
        for repeat in range(4):
            begin=time.perf_counter();submit=0.;wait=0.
            for batch in range(8):
                pending=[]
                for seat in range(4):
                    t=time.perf_counter()
                    if mode in ("d2h","duplex"):
                        sequence+=1;key=HostStateKey(str(sequence),1)
                        handle=store.offload(states,key,selection(seat),stream=streams[0])
                        pending.append((handle,key))
                    if mode in ("h2d","duplex"):
                        handle=cached.restore(states,keys[seat],selection(seat+4),stream=streams[1])
                        pending.append((handle,None))
                    submit+=time.perf_counter()-t
                t=time.perf_counter()
                for handle,key in pending:
                    handle.result()
                    if key:store.release(key)
                wait+=time.perf_counter()-t
            wall=time.perf_counter()-begin
            if repeat:samples.append(dict(wall_seconds=wall,submit_seconds=submit,completion_seconds=wait,
                effective_GB_s_per_direction=32*logical_bytes/wall/1e9))
        rows.append(dict(mode=mode,samples=samples,
            median_effective_GB_s_per_direction=statistics.median(x["effective_GB_s_per_direction"] for x in samples)))
    for i,key in enumerate(keys):
        check=store.offload(states,HostStateKey(f"check-{i}",1),selection(i+4),stream=streams[0]);check.result()
        actual=store._snapshots[check.key].payloads
        expected=cached._snapshots[key].payloads
        assert all(torch.equal(actual[n].tensor,expected[n].tensor) for n in actual),"restore mismatch"
    receipt=dict(scope="actual TorchHostStateBackend,83 target/control lanes,4 independent seats/direction,synthetic exact TP2 shapes,no codec/Store/model",
        batch=args.batch,fragmented=args.fragmented,device=args.device,bytes_per_transfer=logical_bytes,exact=True,results=rows)
    args.output.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)


if __name__=="__main__":main()
