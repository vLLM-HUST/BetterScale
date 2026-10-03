"""Actual83-lane NPU State through native dual-node DRAM/CRC/lease backend."""
import argparse
import json
from pathlib import Path
import time
import torch
import torch_npu
import sys
sys.path.insert(0,"/workspace/pd-kv-layout-results/store-cpu-venv/lib/python3.12/site-packages")
from mooncake.store import MooncakeDistributedStore,ReplicateConfig,ReplicaStatus

from store_shared_hot_probe import NpuStageDriver
from state_dma_probe import qwen35_lanes
from native_state_frame import FramePlan
from native_state_transport import NativeStateTransport,CheckedReplicas
from native_dram_staging import NativeDramStaging
from native_store_leases import NativeStoreLeases
from betterscale.live.runtime.host_state import (
    HostStateKey,HostStateSelection,HostStateDomainSelection,TorchHostStateBackend)

HOSTS={"D":"10.244.2.32","P":"10.244.1.16"}
ENDPOINTS={ip+":55307" for ip in HOSTS.values()}

def run(args):
    if args.output.exists():raise ValueError("Preserve previous evidence")
    torch.set_num_threads(8)
    driver=NpuStageDriver(args.device,0)
    states,residents,pages=qwen35_lanes(args.device,resident_count=2,page_count=32)
    def selection(destination):
        return HostStateSelection((
            HostStateDomainSelection(residents,(int(destination),)),
            HostStateDomainSelection(pages,tuple(range(16 if destination else 0,32 if destination else 16)))))
    plan=FramePlan.build(TorchHostStateBackend._select_lanes(states,selection(False)))
    assert plan.payload_bytes==116572172,plan.payload_bytes
    dummy=query=leases=arena=None
    rows=[];held=[]
    try:
        dummy=MooncakeDistributedStore()
        assert dummy.setup_dummy(64<<20,64<<20,HOSTS[args.role]+":55306")==0
        query=MooncakeDistributedStore()
        assert query.setup(HOSTS[args.role]+":55308","http://10.244.2.32:55302/metadata",
            0,1<<20,"tcp","","10.244.2.32:55301")==0
        def query_complete(keys):
            found=set()
            for key,replicas in query.batch_get_replica_desc(list(keys)).items():
                endpoints={r.get_memory_descriptor().buffer_descriptor.transport_endpoint
                    for r in replicas if r.status==ReplicaStatus.COMPLETE and r.is_memory_replica()}
                if ENDPOINTS.issubset(endpoints):found.add(key)
            return found
        # native_store_node_probe explicitly launches this master TTL.
        leases=NativeStoreLeases(query_complete,ttl_seconds=2)
        config=ReplicateConfig();config.replica_num=2;config.preferred_segments=sorted(ENDPOINTS)
        arena=NativeDramStaging(CheckedReplicas(dummy,config,leases),
            slot_bytes=plan.byte_length+32,slots=args.slots,
            register_device=driver.register,unregister_device=driver.unregister)
        def submit(descriptors,to_host,stream):
            with torch.npu.stream(stream):
                begin=torch.npu.Event(enable_timing=True);begin.record(stream)
                torch.ops._C_ascend.swap_blocks_batch(*descriptors,1 if to_host else 0)
                event=torch.npu.Event(enable_timing=True);event.record(stream)
            def wait():
                event.synchronize();wait.device_seconds=begin.elapsed_time(event)/1000
            return wait
        transport=NativeStateTransport(dummy,arena,"native-full-state-gate-v1",
            submit,leases=leases,verify=True)
        for turn in range(3):
            key=HostStateKey("full-state-"+str(turn),1);held.append(key)
            identity="state-"+str(turn);row=dict(turn=turn)
            if args.write:
                with torch.npu.stream(driver.stream):
                    for index,(_,state) in enumerate(states):
                        span=state.physical_blocks_per_logical_block
                        for block in range(state.num_blocks):
                            state.tensor[block*span:(block+1)*span].fill_(index+1+block+turn*100)
                driver.synchronize()
                begin=time.perf_counter();local=[]
                tr=transport.transfer(key,states,{identity:selection(False)},
                    store=True,stream=driver.stream,on_staged=lambda n:local.append((time.perf_counter(),n)))
                tr.result()
                assert len(local)==1 and local[0][1]==plan.payload_bytes
                row.update(local_ready_seconds=local[0][0]-begin,
                           committed_seconds=time.perf_counter()-begin,store_phases=dict(tr.phase_seconds))
            with torch.npu.stream(driver.stream):
                for _,state in states:state.tensor.zero_()
            driver.synchronize()
            begin=time.perf_counter()
            restore=transport.transfer(key,states,{identity:selection(True)},
                store=False,stream=driver.stream)
            restore.result()
            row["restore_seconds"]=time.perf_counter()-begin
            row["restore_phases"]=dict(restore.phase_seconds)
            # Independent known-value oracle, including untouched source rows.
            for index,(_,state) in enumerate(states):
                host=state.tensor.cpu();span=state.physical_blocks_per_logical_block
                first,count=(1,1) if state.domain is residents else (16,16)
                for block in range(state.num_blocks):
                    value=index+1+(block-first)+turn*100 if first<=block<first+count else 0
                    actual=host[block*span:(block+1)*span]
                    assert torch.equal(actual,torch.full_like(actual,value)),(index,block)
            leases.check(key,complete=True);rows.append(row)
        receipt=dict(role=args.role,write=args.write,device=args.device,exact=True,
            payload_bytes=plan.payload_bytes,frame_bytes=plan.byte_length+32,results=rows,
            scope="synthetic83-lane TP2 geometry; dual-native DRAM, CRC, leases, post-H2D audit; no model or saturation")
        args.output.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
        for key in held:transport.release(key)
    finally:
        driver.synchronize()
        if arena is not None:arena.executor.shutdown(wait=True)
        if leases is not None:leases.close()
        if arena is not None and not arena.active:arena.close()
        if query is not None:query.close()
        if dummy is not None:dummy.close()

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role",choices=HOSTS,required=True);p.add_argument("--write",action="store_true")
    p.add_argument("--slots",type=int,default=2)
    p.add_argument("--device",type=int,default=0);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if not 2<=args.slots<=20:p.error("bounded slots required")
    run(args)
