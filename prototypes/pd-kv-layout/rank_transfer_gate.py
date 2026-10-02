"""Two-host private pinned State gate using Mooncake TransferEngine, not Store.

The gate's ready/commit files are external orchestration, NOT an online wire
protocol. Every exported pointer remains registered/owned through completion.
"""
import argparse
import ctypes
import hashlib
import json
import sys
import time
from pathlib import Path
import torch
import torch_npu
from rank_state_pool import RankStatePool
from rank_replica_receiver import RankReplicaReceiver
from native_state_frame import FramePlan
from state_dma_probe import qwen35_lanes
from store_shared_hot_probe import NpuStageDriver

HOSTS={"P":"10.244.1.16","D":"10.244.2.32"}


def run(args):
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(8)
    driver=NpuStageDriver(0,0)
    states,residents,pages=qwen35_lanes(0,resident_count=2,page_count=32)
    plan=FramePlan.build([(name,state,tuple(range(16)) if state.domain is pages else (0,))
                         for name,state in states])
    pool=RankStatePool((args.role,0,0),512<<20,
                      lambda size:torch.empty(size,dtype=torch.uint8,pin_memory=True))
    sys.path.insert(0,"/workspace/pd-kv-layout-results/store-cpu-venv/lib/python3.12/site-packages")
    from mooncake.engine import TransferEngine
    engine=TransferEngine()
    assert engine.initialize(HOSTS[args.role]+":56301","P2PHANDSHAKE","tcp","")==0
    endpoint=HOSTS[args.role]+":"+str(engine.get_rpc_port())
    objects=[];registered=[];rows=[]
    def copy(desc,to_host):
        with torch.npu.stream(driver.stream):
            begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
            begin.record(driver.stream)
            torch.ops._C_ascend.swap_blocks_batch(*desc,1 if to_host else 0)
            end.record(driver.stream)
        end.synchronize()
        return begin.elapsed_time(end)/1000
    def digest(pointer):
        return hashlib.sha256((ctypes.c_uint8*plan.byte_length).from_address(pointer)).hexdigest()
    receiver=RankReplicaReceiver(pool,engine) if args.role=="P" else None
    peer_source=json.loads(args.peer.read_text()) if args.role=="P" else None
    for turn in range(3):
        key=str(turn);pool.retain(key,(key,))
        if receiver is not None:
            expected=peer_source["objects"][turn]
            assert expected["key"]==key and expected["size"]==plan.byte_length
            target=receiver.prepare(("D",0,0),key,key,plan.byte_length,expected["sha256"],f"{turn:032x}")
            assert target["status"]=="write"
            objects.append(dict(key=key,pointer=target["pointer"],size=target["size"]))
            continue
        writer=pool.reserve(key,plan.byte_length)
        pointer=writer.pointer
        assert engine.register_memory(pointer,plan.byte_length)==0
        registered.append(pointer)
        objects.append(dict(key=key,pointer=pointer,size=plan.byte_length))
        if args.role=="D":
            with torch.npu.stream(driver.stream):
                for i,(_,state) in enumerate(states):state.tensor.fill_(i+1+turn*100)
            driver.synchronize()
            seconds=copy(plan.copy_descriptors(writer,to_host=True),True)
            writer.seal(lambda:None)
            checksum=digest(pointer);objects[-1]["sha256"]=checksum
            rows.append(dict(turn=turn,d2h_seconds=seconds,sha256=checksum))
        else:
            # Hold each writer until the sender's explicit TE completion receipt.
            objects[-1]["writer"]=writer
    (args.output/"ready.json").write_text(json.dumps(dict(role=args.role,endpoint=endpoint,
        objects=[{k:v for k,v in x.items() if k!="writer"} for x in objects])))
    if args.role=="D":
        deadline=time.monotonic()+180
        while not args.peer.exists():
            if time.monotonic()>deadline:raise TimeoutError("peer preparation missing")
            time.sleep(.1)
        peer=json.loads(args.peer.read_text())
        assert peer["role"]=="P" and peer["endpoint"].startswith(HOSTS["P"]+":")
        for row,obj,target in zip(rows,objects,peer["objects"],strict=True):
            assert obj["size"]==target["size"] and obj["key"]==target["key"]
            with pool.read(obj["key"]) as reader:
                begin=time.perf_counter()
                rc=engine.transfer_sync_write(peer["endpoint"],reader.pointer,target["pointer"],reader.size)
                if rc:raise RuntimeError(("TE write failed",rc))
                row["network_seconds"]=time.perf_counter()-begin
                row["network_GB_s"]=reader.size/row["network_seconds"]/1e9
        (args.output/"commit.json").write_text(json.dumps(rows))
    else:
        deadline=time.monotonic()+180
        while not (args.output/"commit.json").exists():
            if time.monotonic()>deadline:raise TimeoutError("sender completion missing; writers remain quarantined")
            time.sleep(.1)
        sender=json.loads((args.output/"commit.json").read_text())
        for turn,(obj,sent) in enumerate(zip(objects,sender,strict=True)):
            assert sent["turn"]==turn and digest(obj["pointer"])==sent["sha256"]
            assert receiver.commit(f"{turn:032x}")
            with torch.npu.stream(driver.stream):
                for _,state in states:state.tensor.zero_()
            driver.synchronize()
            with pool.read(obj["key"]) as reader:
                seconds=copy(plan.copy_descriptors(reader,to_host=False),False)
            for i,(_,state) in enumerate(states):
                host=state.tensor.cpu()
                n=16 if state.domain is pages else state.physical_blocks_per_logical_block
                assert torch.equal(host[:n],torch.full_like(host[:n],i+1+turn*100))
                assert bool((host[n:]==0).all())
            rows.append(dict(turn=turn,h2d_seconds=seconds,exact=True))
    driver.synchronize()
    for pointer in registered:
        assert engine.unregister_memory(pointer)==0
    for turn in range(3):pool.drop(str(turn))
    pool.close()
    receipt=dict(role=args.role,scope="RankReplicaReceiver + private pinned pools + Mooncake TCP TE;83-lane NPU; file-orchestrated control, not online scheduling",
                 payload_bytes=plan.payload_bytes,results=rows)
    (args.output/"result.json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role",choices=HOSTS,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--peer",type=Path)
    run(p.parse_args())
