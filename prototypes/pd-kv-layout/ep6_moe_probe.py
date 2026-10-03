"""Bounded EP6 BF16 routing/GMM/SwiGLU/combine numerical probe.

Exact pinned core mapping, native Ascend operators, independent CPU oracle.
Uses all-gather/reduce rather than MC2; not full model or FULL-graph evidence.
"""
import argparse
import ast
from datetime import timedelta
import json
import os
from pathlib import Path
from typing import Literal
import torch
import torch_npu
import torch.distributed as dist

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK'])
assert int(os.environ['WORLD_SIZE'])==6 and torch.npu.device_count()==6
torch.npu.set_device(local);torch.set_num_threads(2)
root=Path(__file__).resolve().parents[2]
source=root/'upstream/vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py'
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='determine_expert_map')
ns=dict(torch=torch,ExpertPlacementStrategy=Literal['linear','round_robin'])
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
count,mapping,_=ns['determine_expert_map'](6,rank,256)
owned=torch.where(mapping>=0)[0];start=int(owned[0]);end=int(owned[-1])+1
assert torch.equal(owned,torch.arange(start,end)) and end-start==count
# All ranks independently create the same small reference expert weights.
g=torch.Generator().manual_seed(6041)
w1=(torch.randn(256,64,128,generator=g)*.08).bfloat16()
w2=(torch.randn(256,64,64,generator=g)*.08).bfloat16()
dw1=w1[start:end].npu();dw2=w2[start:end].npu()
dist.init_process_group('hccl',timeout=timedelta(seconds=120))
errors=[]
try:
 for wave in range(4):
  x=(torch.randn(16,64,generator=torch.Generator().manual_seed(wave*7+rank))*.5).bfloat16().npu()
  gathered=[torch.empty_like(x) for _ in range(6)]
  dist.all_gather(gathered,x);x=torch.cat(gathered)
  # Cover every expert; vary ownership and skew, including entirely empty ranks.
  ids=((torch.arange(96*8).reshape(96,8)+wave*43)%256).int()
  if wave==2:ids%=43
  probs=torch.softmax(torch.arange(8,dtype=torch.float32)/8,0).expand(96,-1).contiguous()
  if wave==3:probs[1::3]=0 # explicit padded/no-contribution rows
  masked=probs*((ids>=start)&(ids<end))
  routed,indices,counts,_=torch_npu.npu_moe_init_routing_v2(x,ids.npu(),active_num=96*8,
      expert_num=256,expert_tokens_num_type=1,expert_tokens_num_flag=True,
      active_expert_range=[start,end],quant_mode=-1)
  assert counts.numel()==count
  expected_counts=torch.bincount(ids.flatten().long(),minlength=256)[start:end]
  torch.testing.assert_close(counts.cpu().long(),expected_counts,rtol=0,atol=0)
  mid=torch_npu.npu_grouped_matmul(x=[routed],weight=[dw1],split_item=2,
      group_list_type=1,group_type=0,group_list=counts.long())[0]
  act=torch_npu.npu_swiglu(mid)
  values=torch_npu.npu_grouped_matmul(x=[act],weight=[dw2],split_item=2,
      group_list_type=1,group_type=0,group_list=counts.long())[0]
  partial=torch_npu.npu_moe_token_unpermute(permuted_tokens=values,
      sorted_indices=indices.abs(),probs=masked.npu())
  dist.all_reduce(partial)
  # Independent per-expert FP32 products with explicit BF16 stage boundaries.
  cx=x.cpu().float();reference=torch.zeros(96,64)
  for expert in range(256):
   rows,slots=torch.where(ids==expert)
   if not rows.numel():continue
   h=(cx[rows]@w1[expert].float()).bfloat16().float();gate,up=h.chunk(2,-1)
   h=(torch.nn.functional.silu(gate)*up).bfloat16().float()
   value=(h@w2[expert].float()).bfloat16().float()
   reference.index_add_(0,rows,value*probs[rows,slots,None])
  got=partial.cpu().float();err=float((got-reference).abs().max());errors.append(err)
  torch.testing.assert_close(got,reference,rtol=.03,atol=.003)
  if wave==3:assert got[1::3].eq(0).all()
  dist.barrier()
 if rank==0:
  a.output.write_text(json.dumps(dict(status='passed',scope='EP6 eager native routing/GMM/SwiGLU/combine and HCCL, not model/FULL',
      expert_counts=[43,43,43,43,42,42],max_errors=errors,waves=4),indent=2)+'\n')
  print('PASS EP6 native MoE numerical composition',errors,flush=True)
finally:
 dist.destroy_process_group()
