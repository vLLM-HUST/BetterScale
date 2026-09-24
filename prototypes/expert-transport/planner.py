"""Measure existing device routing primitives; not a serving protocol change."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import torch
    import torch_npu
    torch.npu.set_device(0);results=[]
    for rows in [3,96,512,4096]:
        for pattern in ['hot8','broad']:
            ids=((torch.arange(rows*8,device='npu').reshape(rows,8)*71+17)%(8 if pattern=='hot8' else 256)).int()
            route=torch.arange(rows*8,device='npu',dtype=torch.int32)
            expected_counts=torch.bincount(ids.cpu().flatten().long(),minlength=256)
            def plan():
                # Unique integer keys are <=8388607 for this4096*8*256 envelope.
                order=torch.argsort((ids.flatten()*(rows*8)+route).float())
                inverse=torch.empty_like(route).scatter_(0,order,route)
                counts=torch_npu.npu_moe_compute_expert_tokens(ids.flatten().index_select(0,order),256)
                return inverse,counts
            for method,width in [('sort-prefix',0),('native-routing',32),('native-routing',2048)]:
                x=torch.zeros(rows,max(width,32),device='npu',dtype=torch.bfloat16)
                # Encode row identity exactly despite BF16's limited integer precision.
                token=torch.arange(rows,device='npu',dtype=torch.int32)
                x[:,0]=(token%128).bfloat16();x[:,1]=(token//128).bfloat16()
                def body():
                    if method=='sort-prefix':return plan()
                    return torch_npu.npu_moe_init_routing_v2(x,ids,active_num=rows*8,expert_num=256,
                        expert_tokens_num_type=1,expert_tokens_num_flag=True,quant_mode=-1,active_expert_range=[0,256],row_idx_type=0)
                for _ in range(3):value=body()
                torch.npu.synchronize()
                def check(value):
                    if method=='sort-prefix':
                        inverse,ends=value
                        expected_order=torch.argsort(ids.cpu().flatten(),stable=True)
                        inverse_cpu=torch.empty(rows*8,dtype=torch.int32);inverse_cpu[expected_order]=torch.arange(rows*8,dtype=torch.int32)
                        assert torch.equal(inverse.cpu(),inverse_cpu)
                        assert torch.equal(ends.cpu().long(),expected_counts.cumsum(0))
                        return 'stable-row-major-inverse'
                    expanded,mapping,counts,_=value
                    assert torch.equal(counts.cpu().long(),expected_counts),counts.cpu()
                    # Accept only a demonstrated interpretation; do not infer from API name.
                    idx=mapping.cpu().long();assert torch.equal(idx.sort().values,torch.arange(rows*8))
                    actual=expanded[:,:2].cpu();xx=x[:,:2].cpu()
                    out_experts=torch.repeat_interleave(torch.arange(256),expected_counts)
                    for layout,tokenidx,source_ids in [('row-major',torch.arange(rows*8)//8,ids.cpu().flatten()),('topk-major',torch.arange(rows*8)%rows,ids.cpu().T.flatten())]:
                        if torch.equal(actual[idx],xx[tokenidx]) and torch.equal(out_experts[idx],source_ids.long()):return layout+'-source-to-destination'
                        if torch.equal(actual,xx[tokenidx[idx]]) and torch.equal(out_experts,source_ids[idx].long()):return layout+'-destination-to-source'
                    raise AssertionError('unrecognized native row mapping')
                mapping=check(value)
                graph=torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    for _ in range(32):value=body()
                graph.replay();torch.npu.synchronize();check(value);times=[]
                for _ in range(5):
                    start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                    start.record();graph.replay();end.record();end.synchronize()
                    times.append(start.elapsed_time(end)*1000/32)
                row=dict(rows=rows,pattern=pattern,method=method,hidden=width,us=times,mapping=mapping)
                results.append(row);print(json.dumps(row),flush=True);del graph
    a.output.write_text(json.dumps(dict(status='PASS',results=results,scope='native planner primitives; no IPC/multi-source integration'),indent=2)+'\n')

if __name__=='__main__':main()
