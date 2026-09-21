"""Matched complete-core fusion controls; independent recurrence and full-pool parity.

MTP_GDN_PROBE_VARIANTS selects original, parallel, fused1/3/6, or merge.
Graphs share inputs, not state storage; preparation changes device counts and
warm/cold states after capture. Timings are unprofiled, forward/reverse order.
"""
import json
import os
from pathlib import Path
import statistics
import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from mixed_core import MixedCore


def elapsed(graph, count=20):
    start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(count): graph.replay()
    end.record();end.synchronize()
    return start.elapsed_time(end)*1000/count


def select(variant):
    choices = {
        "original": ("original", "1"),
        "parallel": ("parallel", "1"),
        "fused1": ("fused", "1"),
        "fused3": ("fused", "3"),
        "fused6": ("fused", "6"),
        "merge": ("merge", "1"),
    }
    mode, group = choices[variant]
    os.environ["MTP_GDN_WY_MODE"] = mode
    os.environ["MTP_GDN_WY_HEADS"] = group


def main():
    assert enable_custom_op()
    torch.npu.set_device(0);init_device_properties_triton();torch.set_num_threads(4)
    torch.manual_seed(728)
    root=Path(os.environ['CAPSULE'])
    variants=os.environ.get("MTP_GDN_PROBE_VARIANTS", "original,parallel,fused1").split(",")
    assert len(variants) >= 2 and variants[0] == "original"
    for variant in variants: select(variant)
    # An independent CPU recurrence checks intermediate candidate states too.
    import mixed_state_probe
    oracle=root/'cpu-oracle';oracle.mkdir()
    os.environ['CAPSULE']=str(oracle)
    os.environ['MTP_GDN_LAYOUT_FUSION']='1'
    select(variants[-1])
    mixed_state_probe.main()
    os.environ['CAPSULE']=str(root)
    rows=[]
    cases = [
        (16, [3,9], [True,False]),
        (64, [3,17,2,1], [True,False,True,False]),
        (128, [3,65,17], [True,False,False]),
        (256, [3,193,33], [True,False,False]),
        (512, [3,33,257,97], [True,False,False,False]),
        (1024, [3,769,193], [True,False,False]),
        (1536, [3,1024,509], [True,False,False]),
        (2048, [3,1536], [True,False]),
        (2048, [1]*7+[2041], [False]*8),
    ]
    for capacity,lengths,roles in cases:
        cores=[]
        for variant in variants:
            os.environ['MTP_GDN_LAYOUT_FUSION']='1'
            cores.append(MixedCore(capacity))
        n=len(lengths);slots=torch.randperm(n*3).reshape(n,3).tolist()
        seed=torch.randn(n*3+1,24,128,128,device='npu')*.01
        convseed=(torch.randn(n*3+1,5,5120,device='npu')*.1).bfloat16()
        weight=(torch.randn(4,5120,device='npu')*.2).bfloat16()
        log=torch.randn(24,device='npu')*.1;bias=torch.randn_like(log)*.1
        x=(torch.randn(capacity,5120,device='npu')*.1).bfloat16()
        a=(torch.randn(capacity,24,device='npu')*.1).bfloat16();b=torch.randn_like(a)*.1
        banks=[seed.clone() for _ in cores];convs=[convseed.clone() for _ in cores]
        graphs=[];outputs=[]
        for mode,core,state,conv in zip(variants,cores,banks,convs):
            select(mode)
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
            for i in range(len(variants)):
                cores[i].prepare(lengths,roles,slots,counts,initial)
                banks[i].copy_(seed);convs[i].copy_(convseed);graphs[i].replay()
            torch.npu.synchronize()
            t=sum(lengths)
            for i in range(1,len(variants)):
                torch.testing.assert_close(outputs[0][:,:t],outputs[i][:,:t],rtol=0,atol=0)
                torch.testing.assert_close(banks[0],banks[i],rtol=0,atol=0)
                torch.testing.assert_close(convs[0],convs[i],rtol=0,atol=0)
            checks.append(dict(wave=wave,exact=True))
        times=[[] for _ in variants]
        indices=list(range(len(variants)))
        for i in indices + indices[::-1]:
            for _ in range(3): graphs[i].replay()
            times[i].append(elapsed(graphs[i]))
        row=dict(order=variants,capacity=capacity,lengths=lengths,roles=roles,checks=checks,
                 microseconds=times,means=[statistics.mean(v) for v in times])
        rows.append(row);print(json.dumps(row),flush=True)
    (root/'receipt.json').write_text(json.dumps(dict(passed=True,cores=rows,scope="operator only; no service promotion"),indent=2))


if __name__=='__main__':
    with torch.inference_mode(): main()
