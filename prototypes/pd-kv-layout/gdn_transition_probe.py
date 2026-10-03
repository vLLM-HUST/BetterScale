"""Independent oracle for extended-history verify -> short-prefill transitions.

Uses actual owned Core and MixedCore, one selected resident, all three candidates.
This is a numerical leaf, not scheduler, MTP-draft or full-model qualification.
"""
import argparse,json
from pathlib import Path
import torch,torch_npu
from vllm_ascend.utils import enable_custom_op
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from betterscale.models.qwen35.service_metadata import Core
from betterscale.models.qwen35.mixed_core import MixedCore
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
torch.set_num_threads(4);torch.manual_seed(817);torch.npu.set_device(0)
assert torch.npu.device_count()==1 and enable_custom_op();init_device_properties_triton()
wc=(torch.randn(4,4096)*.2).bfloat16();w=wc.npu()
lc=torch.full((16,),-2.);bc=torch.randn(16)*.1;l=lc.npu();bias=bc.npu()
cc=(torch.randn(4,5,4096)*.2).bfloat16();conv=cc.npu()
sc=torch.randn(12,16,128,128)*.01;state=sc.npu()
selected=1;conv_selected=1;history=cc[0,:3].clone();truth_state=sc[0].clone();previous_verify=False
banks={}
def prepare(kind,n):
 core,x,aa,bb,graph,out=banks[kind]
 if kind=='pre':
  core.prepare([n],[False],[[selected-1,0,0]],[1],[True]);core.prefill_conv[0,0]=0
 else:
  core.cu.zero_();core.cu[1:]=n;core.verify.cu.copy_(core.cu)
  core.verify_conv.fill_(-1);core.verify_conv[0,0]=0
  core.verify.slots.fill_(-1);core.verify.slots[0].copy_(torch.arange(3))
  core.accepted.fill_(1);core.accepted[0]=conv_selected
  core.verify.accepted.fill_(1);core.verify.accepted[0]=selected
for kind,capacity in [('pre',256),('ver',3)]:
 core=MixedCore(capacity) if kind=='pre' else Core(capacity,'npu')
 x=torch.zeros(capacity,4096,dtype=torch.bfloat16).npu();aa=torch.zeros(capacity,16,dtype=torch.bfloat16).npu();bb=torch.zeros_like(aa)
 banks[kind]=[core,x,aa,bb,None,None];prepare(kind,3)
 def call():return core(x,aa,bb,w,l,bias,conv,state)
 call();torch.npu.synchronize();graph=torch.npu.NPUGraph()
 with torch.npu.graph(graph):out=call()
 banks[kind][4:]=graph,out
conv.copy_(cc);state.copy_(sc)
rows=[]
for wave in range(24):
 kind='pre' if wave%4 in (0,3) else 'ver';n=([65,1,16][(wave//4)%3] if wave%4==0 else 1) if kind=='pre' else 3
 if kind=='pre' and previous_verify:
  h=conv[0,conv_selected-1:conv_selected+2].clone();conv[0,:3].copy_(h)
  cc[0,:3].copy_(cc[0,conv_selected-1:conv_selected+2].clone())
  torch.testing.assert_close(h.cpu(),history,rtol=0,atol=0)
  conv_selected=1
 core,x,aa,bb,graph,out=banks[kind];prepare(kind,n)
 xc=(torch.randn(x.shape)*.2).bfloat16();ac=(torch.randn(aa.shape)*.2).bfloat16();betac=(torch.randn(bb.shape)*.2).bfloat16()
 x.copy_(xc);aa.copy_(ac);bb.copy_(betac)
 joined=torch.cat([history,xc[:n]])
 cv=sum(joined[i:i+n].float()*wc[i].float() for i in range(4))
 packed=torch.nn.functional.silu(cv).bfloat16();q,k,v=packed.split([1024,1024,2048],-1)
 def norm(z):
  z=z.reshape(n,8,128).float();return (z*torch.rsqrt((z*z).sum(-1,keepdim=True)+1e-6)).bfloat16().float().repeat_interleave(2,1)
 q=norm(q)*128**-.5;k=norm(k);v=v.reshape(n,16,128).float()
 gate=-lc.exp()*torch.nn.functional.softplus(ac[:n].float()+bc)
 beta=betac[:n].float().sigmoid().bfloat16().float()
 s=truth_state.clone();expected=[];candidates=[]
 for t in range(n):
  s*=gate[t].exp()[:,None,None];delta=(v[t]-(s*k[t,:,:,None]).sum(1))*beta[t,:,None]
  s+=k[t,:,:,None]*delta[:,None,:];expected.append((s*q[t,:,:,None]).sum(1));candidates.append(s.clone())
 graph.replay();torch.npu.synchronize();got=out[0,:n].cpu().float();ref=torch.stack(expected)
 torch.testing.assert_close(got,ref,rtol=.02,atol=.003)
 if kind=='ver':
  sc[:3].copy_(torch.stack(candidates));cc[0,:5].copy_(joined[1:6])
  accepted=(wave//4)%3+1;selected=accepted;conv_selected=accepted;history=joined[accepted:accepted+3].clone();truth_state=candidates[accepted-1]
 else:
  sc[selected-1].copy_(s);cc[0,:3].copy_(joined[-3:]);history=joined[-3:].clone();truth_state=s
 torch.testing.assert_close(state.cpu(),sc,rtol=.02,atol=.003)
 torch.testing.assert_close(conv.cpu(),cc,rtol=0,atol=0)
 previous_verify=kind=='ver';rows.append(dict(wave=wave,kind=kind,n=n,selected=selected,error=float((got-ref).abs().max())))
 (a.output/'progress.json').write_text(json.dumps(rows,indent=2))
(a.output/'complete.json').write_text(json.dumps(dict(status='passed',rows=rows),indent=2));print('PASS verify/prefill transition',max(r['error'] for r in rows))
