"""Real per-rank control + native TransferEngine + NPU State gate, no Store."""
import argparse
import json
import sys
import time
from pathlib import Path
import torch
import torch_npu
from rank_state_pool import RankStatePool
from rank_replica_receiver import RankReplicaReceiver
from rank_peer_control import RankControl
from rank_replicator import RankReplicator,HOSTS,CONTROL_BASE
from rank_state_transport import RankStateTransport
from state_dma_probe import qwen35_lanes
from store_shared_hot_probe import NpuStageDriver
from betterscale.live.runtime.host_state import HostStateKey,HostStateSelection,HostStateDomainSelection


def run(args):
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(8);driver=NpuStageDriver(0,0)
    states,residents,pages=qwen35_lanes(0,resident_count=2,page_count=32)
    select=HostStateSelection((HostStateDomainSelection(residents,(0,)),
                               HostStateDomainSelection(pages,tuple(range(16)))))
    def allocate(size):
        torch.npu.set_device(0)
        return torch.empty(size,dtype=torch.uint8,pin_memory=True)
    pool=RankStatePool((args.role,0,0),512<<20,allocate)
    sys.path.insert(0,"/workspace/pd-kv-layout-results/store-cpu-venv/lib/python3.12/site-packages")
    from mooncake.engine import TransferEngine
    engine=TransferEngine()
    assert engine.initialize(HOSTS[args.role]+":56301","P2PHANDSHAKE","tcp","")==0
    receiver=RankReplicaReceiver(pool,engine)
    server=RankControl(receiver,HOSTS[args.role],CONTROL_BASE,set(HOSTS.values())).start()
    replica=RankReplicator(pool,engine)
    def submit(descriptors,to_host,stream):
        with torch.npu.stream(stream):
            begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
            begin.record(stream)
            torch.ops._C_ascend.swap_blocks_batch(*descriptors,1 if to_host else 0)
            end.record(stream)
        def wait():
            end.synchronize();wait.device_seconds=begin.elapsed_time(end)/1000
        return wait
    transport=RankStateTransport(pool,"rank-rpc-gate/tp2/head0",submit,replica.replicate,verify=True,release_checkpoint=receiver.drop)
    (args.output/"ready.json").write_text(json.dumps(dict(role=args.role,control_port=CONTROL_BASE)))
    rows=[];keys=[]
    for turn in range(3):
        key=HostStateKey("rpc-"+str(turn),1);keys.append(key)
        identity="state-"+str(turn)
        if args.role=="D":
            replica.set_peer(key,0)
            with torch.npu.stream(driver.stream):
                for i,(_,state) in enumerate(states):state.tensor.fill_(i+1+turn*100)
            driver.synchronize()
            begin=time.perf_counter();local=[]
            tr=transport.transfer(key,states,{identity:select},store=True,stream=driver.stream,
                                  on_staged=lambda _:local.append(time.perf_counter()-begin))
            tr.result()
            assert len(local)==1
            rows.append(dict(turn=turn,local_ready_seconds=local[0],committed_seconds=time.perf_counter()-begin,
                             phases=dict(tr.phase_seconds)))
        else:
            deadline=time.monotonic()+180
            while True:
                with pool.lock:ready=key in pool.groups and pool.complete(key)
                if ready:break
                if receiver.failure:raise RuntimeError(receiver.failure)
                if time.monotonic()>deadline:raise TimeoutError("private State commit missing")
                time.sleep(.05)
            with torch.npu.stream(driver.stream):
                for _,state in states:state.tensor.zero_()
            driver.synchronize()
            tr=transport.transfer(key,states,{identity:select},store=False,stream=driver.stream)
            tr.result()
            for i,(_,state) in enumerate(states):
                host=state.tensor.cpu()
                n=16 if state.domain is pages else state.physical_blocks_per_logical_block
                assert torch.equal(host[:n],torch.full_like(host[:n],i+1+turn*100))
                assert bool((host[n:]==0).all())
            rows.append(dict(turn=turn,exact=True,phases=dict(tr.phase_seconds)))
    assert not receiver.pending and not replica.registered and not replica.quarantined
    server.close();driver.synchronize()
    for key in keys:transport.release(key)
    assert not receiver.completed
    pool.close()
    receipt=dict(role=args.role,scope="RankStateTransport + private pinned pools + control RPC + native TCP TE;83-lane synthetic State, not model scheduling",
                 results=rows)
    (args.output/"result.json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role",choices=HOSTS,required=True)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
