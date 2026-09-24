"""Qwen35 real-weight native EP2 DFC reference, not equal-resource A+E serving.

Two synchronous source/compute ranks, H2048/M512/E256/K8, original BF16.
Capacity covers every route even if all routes land on one owner. Independent
plain linear/SILU oracle precedes timing and verifies changed graph inputs.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics


def worker(rank, a):
    import torch
    import torch.distributed as dist
    import torch_npu
    torch.set_num_threads(2); torch.npu.set_device(rank)
    torch_npu.npu.config.allow_internal_format=True
    torch.ops.load_library(os.environ['DFC_EXTENSION'])
    dist.init_process_group('hccl',init_method=f'tcp://127.0.0.1:{a.port}',
                            rank=rank,world_size=2,timeout=timedelta(seconds=120))
    bootstrap=torch.ones(1,device='npu');dist.all_reduce(bootstrap);torch.npu.synchronize()
    comm=dist.distributed_c10d._get_default_group()._get_backend(
        torch.device('npu')).get_hccl_comm_name(rank)
    from betterscale.patches.expert_service.checkpoint import Checkpoint,plain_experts
    cp=Checkpoint(a.model)
    catalogs={}
    for layer in a.layers:
        gate,up,down=cp.experts(layer,bounds=(rank*128,(rank+1)*128))
        catalogs[layer]=([torch_npu.npu_format_cast(torch.cat((gate,up),dim=1).transpose(1,2).contiguous(),29)],
                         [torch_npu.npu_format_cast(down.transpose(1,2).contiguous(),29)])
    del gate,up,down
    scale1=[torch.ones((128,1024),dtype=torch.int64,device='npu')]
    scale2=[torch.ones((128,2048),dtype=torch.int64,device='npu')]
    results=[]
    for layer in a.layers:
        oracle_weights=cp.experts(layer)
        for rows in a.rows:
            single_owner=a.one_source or a.sentinel_source
            local_rows=(rows-1 if rank==0 else 1) if a.sentinel_source else (0 if a.one_source and rank==1 else rows)
            for pattern in (('remote128',) if single_owner else ('hot8','broad')):
                torch.manual_seed(20260924+(0 if single_owner else rank))
                x=(torch.randn(local_rows,2048,device='npu')*.1).bfloat16()
                ids=(torch.arange(local_rows*8,device='npu').reshape(local_rows,8)%
                     (8 if pattern=='hot8' else 128 if pattern=='remote128' else 256)).int()
                if pattern=='remote128':ids.add_(128)
                probs=torch.full((local_rows,8),.125,device='npu')
                if a.sentinel_source:
                    # Equal global fixture with one zero-valued token originating
                    # locally at the sole compute owner; zero-size donor inputs
                    # failed before readiness in the separately retained probe.
                    torch.manual_seed(20260924)
                    full_x=(torch.randn(rows,2048,device='npu')*.1).bfloat16();full_x[-1].zero_()
                    full_ids=(torch.arange(rows*8,device='npu').reshape(rows,8)%128+128).int()
                    x=full_x[:-1] if rank==0 else full_x[-1:]
                    ids=full_ids[:-1] if rank==0 else full_ids[-1:]
                output=torch.empty_like(x)
                counts=torch.zeros(128,dtype=torch.int32,device='npu')
                global_rows=rows if single_owner else 2*rows
                capacity=max(512, (global_rows*8+511)//512*512)
                def body():
                    torch.ops._C_ascend.dispatch_ffn_combine(x=x,
                        weight1=catalogs[layer][0],weight2=catalogs[layer][1],
                        expert_idx=ids,scale1=scale1,scale2=scale2,bias1=[],bias2=[],
                        probs=probs,group=comm,max_output_size=capacity,
                        out=output,expert_token_nums=counts)
                def error(expected):
                    e=float(torch.linalg.vector_norm(output.float()-expected.float())/
                            torch.linalg.vector_norm(expected.float()).clamp_min(1e-12))
                    assert torch.isfinite(output).all() and e<.02,(rank,layer,rows,pattern,e)
                    return e
                expected=plain_experts(oracle_weights,x,ids.long(),probs)
                print(json.dumps(dict(stage='warmup',rank=rank,layer=layer,rows=rows,pattern=pattern)),flush=True)
                dist.barrier()
                body();torch.npu.synchronize()
                initial=error(expected)
                graph=torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    for _ in range(a.repeats):body()
                graph.replay();torch.npu.synchronize();error(expected)
                samples=[]
                for _ in range(5):
                    dist.barrier();torch.npu.synchronize()
                    start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                    start.record();graph.replay();end.record();end.synchronize()
                    samples.append(start.elapsed_time(end)*1000/a.repeats)
                x.mul_(.5)
                if single_owner:ids.add_(17).remainder_(128).add_(128)
                else:ids.add_(137).remainder_(256)
                if a.sentinel_source:
                    full_probs=torch.softmax(torch.randn(rows,8,device='npu'),dim=-1)
                    probs.copy_(full_probs[:-1] if rank==0 else full_probs[-1:])
                else:probs.copy_(torch.softmax(torch.randn(local_rows,8,device='npu'),dim=-1))
                expected=plain_experts(oracle_weights,x,ids.long(),probs)
                dist.barrier();graph.replay();torch.npu.synchronize()
                changed=error(expected)
                result=dict(rank=rank,layer=layer,rows_per_source=local_rows,global_rows=global_rows,one_source=a.one_source,sentinel_source=a.sentinel_source,
                    pattern=pattern,capacity=capacity,initial_l2=initial,changed_l2=changed,
                    trials_us=samples,median_us=statistics.median(samples))
                results.append(result)
                (a.output/f'rank{rank}.json').write_text(json.dumps(results,indent=2))
                print(json.dumps(result),flush=True);graph.reset()
        del oracle_weights
    dist.barrier();dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='/workspace/models/Qwen3.5-35B-A3B')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=43870)
    p.add_argument('--rows',type=int,nargs='+',default=[3,96,512,4096])
    p.add_argument('--layers',type=int,nargs='+',default=[0,40])
    p.add_argument('--repeats',type=int,default=16)
    p.add_argument('--one-source',action='store_true',help='Only rank0 supplies tokens; all routes use rank1 experts128..255')
    p.add_argument('--sentinel-source',action='store_true',help='N-1 tokens at rank0 and one zero token at sole owner/rank1; equal global work')
    a=p.parse_args()
    assert not (a.one_source and a.sentinel_source)
    assert not a.sentinel_source or min(a.rows)>=2
    assert a.rows and all(1<=n<=4096 for n in a.rows)
    assert a.layers and set(a.layers)<={0,40}
    assert 1<=a.repeats<=64 and 1024<=a.port<=65535
    # DFC reserves about one third of its HCCL input window for expanded hidden.
    # 4096*8*2048*2 =128MiB; a512MiB window leaves room for peer metadata/output.
    assert int(os.environ['HCCL_BUFFSIZE'])>=512
    os.environ['BETTERSCALE_EXPERT_DRAFT_LAYERS']='1'
    a.output.mkdir(exist_ok=False)
    import torch.multiprocessing as mp
    mp.spawn(worker,args=(a,),nprocs=2,join=True)
    (a.output/'result.json').write_text(json.dumps(dict(status='PASS',
        scope=('N-1+1 sources, sole compute owner/rank1, same global remote128-sentinel fixture; one local token' if a.sentinel_source else
               'one source/rank0, one active expert owner/rank1, exact remote128 FFN work' if a.one_source else
               'native EP2 real-weight reference; two synchronous source/compute ranks, not A+E parity'),
        geometry=dict(hidden=2048,inner=512,experts=256,topk=8),
        provider=os.environ['DFC_EXTENSION']),indent=2))


if __name__=='__main__': main()
