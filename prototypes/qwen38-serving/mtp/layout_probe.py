"""Paired complete mixed-core graphs, plus independent recurrence and slot oracles."""
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import statistics
import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from mixed_core import MixedCore
from device_metadata import publish_slots_reference as reference_slots
from device_slots import publish_slots as fused_slots


def slots_probe():
    rows = []
    for decode, pre, ver in [(True, [], [0,1,2]), (False,[0,2],[1]),
                              (False,[0,1,2],[]), (False,[1],[2,0])]:
        for step in range(3):
            table = torch.arange(3*12,dtype=torch.int32).reshape(3,12)[:,:10]
            table[1,1] = 0
            seq = torch.tensor([16+step,31+step,3+step],dtype=torch.int32)
            def metadata(device):
                return NS(live=3,width=3,decode=decode,
                    device_slot_source=(table.to(device),seq.to(device),16,pre,ver),
                    cu=torch.tensor([0,17,20,23,23,23,23,23,23,23],dtype=torch.int32,device=device),
                    initial=torch.zeros(9,dtype=torch.bool,device=device),
                    prefill_ids=torch.tensor(pre+[0]*(9-len(pre)),device=device),
                    verify_ids=torch.tensor(ver+[0]*(9-len(ver)),device=device),
                    prefill_conv=torch.full((9,1),-1,dtype=torch.int32,device=device),
                    verify_conv=torch.full((9,1),-1,dtype=torch.int32,device=device),
                    prefill=NS(state=torch.zeros((9,2),dtype=torch.int64,device=device)),
                    verify=NS(slots=torch.full((9,3),-1,dtype=torch.int64,device=device)))
            cpu, npu = metadata('cpu'), metadata('npu')
            reference_slots(cpu); fused_slots(npu)
            for attr in ('initial','prefill_conv','verify_conv'):
                torch.testing.assert_close(getattr(npu,attr).cpu(),getattr(cpu,attr),rtol=0,atol=0)
            torch.testing.assert_close(npu.prefill.state.cpu(),cpu.prefill.state,rtol=0,atol=0)
            torch.testing.assert_close(npu.verify.slots.cpu(),cpu.verify.slots,rtol=0,atol=0)
            rows.append(dict(decode=decode,pre=pre,verify=ver,step=step,passed=True))
    return rows


def elapsed(graph, count=20):
    start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(count): graph.replay()
    end.record();end.synchronize()
    return start.elapsed_time(end)*1000/count


def main():
    assert enable_custom_op()
    torch.npu.set_device(0);init_device_properties_triton();torch.set_num_threads(4)
    torch.manual_seed(728)
    root=Path(os.environ['CAPSULE'])
    slot_rows=slots_probe()
    print('slot oracle PASS',flush=True)
    # An independent CPU recurrence checks intermediate candidate states too.
    import mixed_state_probe
    oracle=root/'cpu-oracle';oracle.mkdir()
    os.environ['CAPSULE']=str(oracle)
    os.environ['MTP_GDN_LAYOUT_FUSION']='1'
    mixed_state_probe.main()
    os.environ['CAPSULE']=str(root)
    rows=[]
    for capacity,lengths,roles in [(512,[3,33,257,97],[True,False,False,False]),
                                  (1536,[3,1024,509],[True,False,False]),
                                  (2048,[3,1536],[True,False]),
                                  (64,[3,17,2,1],[True,False,True,False])]:
        cores=[]
        for flag in ('0','1'):
            os.environ['MTP_GDN_LAYOUT_FUSION']=flag
            cores.append(MixedCore(capacity))
        n=len(lengths);slots=torch.randperm(n*3).reshape(n,3).tolist()
        seed=torch.randn(n*3+1,24,128,128,device='npu')*.01
        convseed=(torch.randn(n*3+1,5,5120,device='npu')*.1).bfloat16()
        weight=(torch.randn(4,5120,device='npu')*.2).bfloat16()
        log=torch.randn(24,device='npu')*.1;bias=torch.randn_like(log)*.1
        x=(torch.randn(capacity,5120,device='npu')*.1).bfloat16()
        a=(torch.randn(capacity,24,device='npu')*.1).bfloat16();b=torch.randn_like(a)*.1
        banks=[seed.clone(),seed.clone()];convs=[convseed.clone(),convseed.clone()]
        graphs=[];outputs=[]
        for core,state,conv in zip(cores,banks,convs):
            core.prepare(lengths,roles,slots,[3]*n,[True]*n)
            core(x,a,b,weight,log,bias,conv,state);torch.npu.synchronize()
            graph=torch.npu.NPUGraph()
            with torch.npu.graph(graph): out=core(x,a,b,weight,log,bias,conv,state)
            graphs.append(graph);outputs.append(out)
        checks=[]
        for wave in range(3):
            x.normal_(0,.1);a.normal_(0,.1);b.normal_(0,.1)
            counts=[1+(wave+i)%3 for i in range(n)]
            initial=[role or wave%2==0 for role in roles]
            for i in range(2):
                cores[i].prepare(lengths,roles,slots,counts,initial)
                banks[i].copy_(seed);convs[i].copy_(convseed);graphs[i].replay()
            torch.npu.synchronize()
            t=sum(lengths)
            torch.testing.assert_close(outputs[0][:,:t],outputs[1][:,:t],rtol=0,atol=0)
            torch.testing.assert_close(banks[0],banks[1],rtol=0,atol=0)
            torch.testing.assert_close(convs[0],convs[1],rtol=0,atol=0)
            checks.append(dict(wave=wave,exact=True))
        times=[[],[]]
        for i in (0,1,1,0):
            for _ in range(3): graphs[i].replay()
            times[i].append(elapsed(graphs[i]))
        row=dict(capacity=capacity,lengths=lengths,roles=roles,checks=checks,
                 microseconds=times,means=[statistics.mean(v) for v in times])
        rows.append(row);print(json.dumps(row),flush=True)
    (root/'receipt.json').write_text(json.dumps(dict(passed=True,slots=slot_rows,cores=rows),indent=2))


if __name__=='__main__':
    with torch.inference_mode(): main()
