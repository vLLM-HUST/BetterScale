"""Captured BT64 solve controls; exact donor parity and FP64 inverse oracle."""
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace
import torch
import torch_npu
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from betterscale.patches.qwen_gdn.metadata import chunk_rows
from betterscale.patches.qwen_gdn.solve_tril import solve_tril, solve_tril_16x16_kernel
from vllm_ascend.ops.triton.fla.solve_tril import merge_16x16_to_64x64_inverse_kernel
from solve_fusion import solve


def tiled(A, meta):
    _, T, H, _ = A.shape
    Ad=torch.empty((1,T,H,16),device=A.device,dtype=torch.float32)
    out=torch.empty_like(A,dtype=torch.bfloat16)
    ix=meta.indices[64]
    solve_tril_16x16_kernel[(len(ix),H)](A,Ad,meta.cu,ix,T,H,BT=64,
        LARGE_BLOCK_T=64,EXTRACT_SLICE_STRIDE_1=2,num_warps=1,num_stages=4)
    merge_16x16_to_64x64_inverse_kernel[(len(ix),H)](A,Ad,out,meta.cu,ix,T,H,BT=64,
        num_warps=4,num_stages=3)
    return out


def elapsed(graph):
    a,b=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
    a.record()
    for _ in range(30): graph.replay()
    b.record();b.synchronize()
    return a.elapsed_time(b)*1000/30


def main():
    torch.npu.set_device(0); init_device_properties_triton();torch.set_num_threads(4)
    torch.manual_seed(738)
    rows=[]
    for capacity,lengths in [(64,[1,15,17,31]),(512,[33,257,97]),(1536,[1024,509]),(2048,[1536]),(2048,[1]*7+[2041])]:
        cu=[0]
        for n in lengths: cu.append(cu[-1]+n)
        meta=SimpleNamespace(cu=torch.tensor(cu+[cu[-1]]*(10-len(cu)),device='npu',dtype=torch.int64),
            indices={size:torch.tensor(chunk_rows(lengths,size,capacity),device='npu',dtype=torch.int64) for size in (64,1216)})
        A=torch.zeros(1,capacity,24,64,device='npu')
        funcs=[lambda:solve_tril(A,meta.cu,meta.indices[1216],meta.indices[64],torch.bfloat16),lambda:tiled(A,meta),lambda:solve(A,meta)]
        graphs=[];outs=[]
        for fn in funcs:
            fn();torch.npu.synchronize()
            g=torch.npu.NPUGraph()
            with torch.npu.graph(g): out=fn()
            graphs.append(g);outs.append(out)
        errors=[]
        for wave in range(3):
            cpu=torch.zeros_like(A,device='cpu')
            for req,n in enumerate(lengths):
                for start in range(0,n,64):
                    size=min(64,n-start)
                    block=torch.randn(24,size,64)*(.01*(wave+1))
                    block=torch.tril(block,diagonal=-1)
                    cpu[0,cu[req]+start:cu[req]+start+size]=block.transpose(0,1)
            A.copy_(cpu)
            for g in graphs:g.replay()
            torch.npu.synchronize()
            observed=[o.cpu()[0,:sum(lengths)] for o in outs]
            for out in observed[1:]:torch.testing.assert_close(out,observed[0],rtol=0,atol=0)
            maxerr=0.
            for req,n in enumerate(lengths):
                for start in range(0,n,64):
                    size=min(64,n-start);base=cu[req]+start
                    mat=cpu[0,base:base+size,:,:size].transpose(0,1).double()
                    inv=torch.linalg.inv(mat+torch.eye(size))
                    obs=observed[-1][base:base+size,:,:size].transpose(0,1).double()
                    torch.testing.assert_close(obs,inv,rtol=.008,atol=2e-5)
                    maxerr=max(maxerr,float((obs-inv).abs().max()))
            errors.append(maxerr)
        timings=[[],[],[]]
        for i in (0,1,2,2,1,0):
            for _ in range(3):graphs[i].replay()
            timings[i].append(elapsed(graphs[i]))
        row=dict(capacity=capacity,lengths=lengths,exact=True,oracle_error=errors,
            order=['old1216','split64','fused64'],times_us=timings,means_us=list(map(statistics.mean,timings)))
        rows.append(row);print(json.dumps(row),flush=True)
    Path(os.environ['CAPSULE'],'receipt.json').write_text(json.dumps(dict(passed=True,rows=rows),indent=2))


if __name__=='__main__':
    with torch.inference_mode():main()
