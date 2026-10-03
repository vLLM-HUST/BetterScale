"""Owned TP2 GDN prefill composition with independent CPU math and host resume.

Covers conv, normalization/gates, WY/chunk kernel and target State; not the model.
"""
import argparse
import json
from pathlib import Path
import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op
from betterscale.models.qwen35.mixed_core import MixedCore

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
torch.set_num_threads(4);torch.manual_seed(175)
torch.npu.set_device(0);assert torch.npu.device_count()==1;assert enable_custom_op()
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
init_device_properties_triton()
T=256
wc=(torch.randn(4,4096)*.2).bfloat16();w=wc.npu()
logc=torch.full((16,),-2.);biasc=torch.randn(16)*.1
log=logc.npu();bias=biasc.npu()
cc=(torch.randn(12,3,4096)*.2).bfloat16();conv=cc.npu()
sc=torch.randn(12,16,128,128)*.01;state=sc.npu()
x=torch.zeros(T,4096,dtype=torch.bfloat16).npu()
aa=torch.zeros(T,16,dtype=torch.bfloat16).npu();bb=torch.zeros_like(aa)
core=MixedCore(T)
owners=[0,3,6]
def prepare(lengths):
    core.prepare(lengths,[False]*3,[[i,i+1,i+2] for i in owners],[1]*3,[True]*3)
def call():return core(x,aa,bb,w,log,bias,conv,state)
prepare([65,127,3]);call();torch.npu.synchronize()
graph=torch.npu.NPUGraph()
with torch.npu.graph(graph):output=call()
conv.copy_(cc);state.copy_(sc)
rows=[]
for wave,lengths in enumerate(([65,127,3],[1,129,64],[128,3,65],[63,64,65])*2):
    if wave==4:
        # Restore both canonical histories at a retired writer boundary.
        cp_conv=conv[owners[0]].cpu().clone();cp_state=state[owners[0]].cpu().clone()
        torch.testing.assert_close(cp_conv,cc[owners[0]],rtol=0,atol=0)
        torch.testing.assert_close(cp_state,sc[owners[0]],rtol=.02,atol=.003)
        conv[9].copy_(cp_conv);state[9].copy_(cp_state)
        cc[9].copy_(cc[owners[0]]);sc[9].copy_(sc[owners[0]])
        owners[0]=9
    xc=(torch.randn(x.shape)*.2).bfloat16()
    ac=(torch.randn(aa.shape)*.2).bfloat16();bc=(torch.randn(bb.shape)*.2).bfloat16()
    x.copy_(xc);aa.copy_(ac);bb.copy_(bc);prepare(lengths)
    outputs=[];offset=0
    for owner,n in zip(owners,lengths):
        joined=torch.cat([cc[owner],xc[offset:offset+n]])
        cv=sum(joined[i:i+n].float()*wc[i].float() for i in range(4))
        packed=torch.nn.functional.silu(cv).bfloat16()
        cc[owner].copy_(joined[-3:])
        qs,ks,vs=packed.split([1024,1024,2048],-1)
        def norm(t):
            t=t.reshape(n,8,128).float()
            return (t*torch.rsqrt((t*t).sum(-1,keepdim=True)+1e-6)).bfloat16().float().repeat_interleave(2,1)
        qs=norm(qs)*128**-.5;ks=norm(ks);vs=vs.reshape(n,16,128).float()
        gate=-logc.exp()*torch.nn.functional.softplus(ac[offset:offset+n].float()+biasc)
        beta=bc[offset:offset+n].float().sigmoid().bfloat16().float()
        s=sc[owner].clone()
        for t in range(n):
            s*=gate[t].exp()[:,None,None]
            delta=(vs[t]-(s*ks[t,:, :,None]).sum(1))*beta[t,:,None]
            s+=ks[t,:,:,None]*delta[:,None,:]
            outputs.append((s*qs[t,:,:,None]).sum(1))
        sc[owner].copy_(s);offset+=n
    graph.replay();torch.npu.synchronize()
    actual=output[0,:offset].cpu().float();truth=torch.stack(outputs)
    torch.testing.assert_close(actual,truth,rtol=.02,atol=.003)
    torch.testing.assert_close(conv.cpu(),cc,rtol=0,atol=0)
    torch.testing.assert_close(state.cpu(),sc,rtol=.02,atol=.003)
    rows.append(dict(wave=wave,lengths=lengths,output_max=(actual-truth).abs().max().item(),
        state_max=(state.cpu()-sc).abs().max().item()))
    (a.output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
(a.output/'complete.json').write_text(json.dumps(dict(status='passed',rows=rows,
    scope='Owned TP2 prefill composition and canonical conv+GDN host checkpoint/new-slot resume; no model/Store/MTP'),indent=2)+'\n')
print('PASS',rows)
