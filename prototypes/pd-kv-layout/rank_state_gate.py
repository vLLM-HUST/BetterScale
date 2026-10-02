"""Bounded83-lane DMA gate for one rank's final private pinned DRAM cache."""
import argparse
import json
import time
from pathlib import Path
import torch
import torch_npu
from rank_state_pool import RankStatePool
from native_state_frame import FramePlan
from rank_state_transport import RankStateTransport
from betterscale.live.runtime.host_state import HostStateSelection,HostStateDomainSelection
from state_dma_probe import qwen35_lanes
from store_shared_hot_probe import NpuStageDriver


def run(output):
    if output.exists():
        raise ValueError("preserve prior evidence")
    torch.set_num_threads(8)
    driver=NpuStageDriver(0,0)
    states,residents,pages=qwen35_lanes(0,resident_count=2,page_count=32)
    plan=FramePlan.build([(name,state,tuple(range(16)) if state.domain is pages else (0,))
                         for name,state in states])
    pool=RankStatePool(("D",0,0),512<<20,
                      lambda size:torch.empty(size,dtype=torch.uint8,pin_memory=True))
    rows=[]
    select=HostStateSelection((HostStateDomainSelection(residents,(0,)),
                               HostStateDomainSelection(pages,tuple(range(16)))))
    def submit(descriptors,to_host,stream):
        with torch.npu.stream(stream):
            begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
            begin.record(stream)
            torch.ops._C_ascend.swap_blocks_batch(*descriptors,1 if to_host else 0)
            end.record(stream)
        def wait():
            end.synchronize()
            wait.device_seconds=begin.elapsed_time(end)/1000
        return wait
    def replica(key,objects):
        assert pool.complete(key)  # network is independently gated elsewhere
    transport=RankStateTransport(pool,"rank-gate",submit,replica,verify=True)
    for turn in range(3):
        key=str(turn)
        with torch.npu.stream(driver.stream):
            for index,(_,state) in enumerate(states):
                state.tensor.fill_(index+1+turn*100)
        driver.synchronize()
        start=time.perf_counter()
        staged=[]
        transfer=transport.transfer(key,states,{key:select},store=True,stream=driver.stream,
            on_staged=lambda _:staged.append(time.perf_counter()-start))
        transfer.result()
        assert len(staged)==1
        store_seconds=transfer.phase_seconds["d2h_device"]
        staged_seconds=staged[0]
        with torch.npu.stream(driver.stream):
            for _,state in states:state.tensor.zero_()
        driver.synchronize()
        restored=transport.transfer(key,states,{key:select},store=False,stream=driver.stream)
        restored.result()
        load_seconds=restored.phase_seconds["h2d_device"]
        for index,(_,state) in enumerate(states):
            host=state.tensor.cpu()
            n=16 if state.domain is pages else state.physical_blocks_per_logical_block
            assert torch.equal(host[:n],torch.full_like(host[:n],index+1+turn*100))
            assert bool((host[n:]==0).all())
        rows.append(dict(turn=turn,payload_bytes=plan.payload_bytes,exact=True,
                         staged_seconds=staged_seconds,d2h_device_seconds=store_seconds,
                         h2d_device_seconds=load_seconds,
                         d2h_GB_s=plan.payload_bytes/store_seconds/1e9,
                         h2d_GB_s=plan.payload_bytes/load_seconds/1e9))
    for turn in range(3):transport.release(str(turn))
    pool.close()
    receipt=dict(scope="RankStateTransport final private DRAM objects;83 lanes; post-H2D audit; local replica stub, no Store/network/model",
                 results=rows)
    output.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output)
