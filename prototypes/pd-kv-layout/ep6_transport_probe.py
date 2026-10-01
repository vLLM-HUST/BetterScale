"""Six D ranks, three TP2 attention groups, uneven EP6 HCCL roundtrip.

Uses pinned core expert mapping. This is NOT Ascend fused-MoE/model qualification.
Launch with torchrun and exactly six authorized idle devices exposed.
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
rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
assert int(os.environ['WORLD_SIZE'])==6
assert torch.npu.device_count()==6
torch.npu.set_device(local)
source=Path(__file__).resolve().parents[2]/'upstream/vllm/vllm/model_executor/layers/fused_moe/expert_map_manager.py'
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='determine_expert_map')
ns=dict(torch=torch,ExpertPlacementStrategy=Literal['linear','round_robin'])
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
count,mapping,_=ns['determine_expert_map'](6,rank,256)
owned=torch.where(mapping>=0)[0].int()
counts=[ns['determine_expert_map'](6,r,256)[0] for r in range(6)]
assert counts==[43,43,43,43,42,42]
dist.init_process_group('hccl',timeout=timedelta(seconds=120))
groups=[]
try:
    for first in (0,2,4): groups.append(dist.new_group([first,first+1],backend='hccl'))
    marker=torch.tensor([rank+1.],device='npu')
    dist.all_reduce(marker,group=groups[rank//2])
    assert marker.item()==(rank//2*2+1)+(rank//2*2+2)
    maps=[torch.empty(256,dtype=torch.int32,device='npu') for _ in range(6)]
    dist.all_gather(maps,mapping.npu())
    assert torch.stack([m.cpu()>=0 for m in maps]).sum(0).eq(1).all()
    for wave in range(8):
        sent=(torch.arange(256,dtype=torch.int32)+rank*1000+wave*10000).npu()
        received=torch.empty(6*count,dtype=torch.int32,device='npu')
        dist.all_to_all_single(received,sent,output_split_sizes=[count]*6,input_split_sizes=counts)
        expected=torch.cat([owned+sender*1000+wave*10000 for sender in range(6)])
        torch.testing.assert_close(received.cpu(),expected,rtol=0,atol=0)
        returned=torch.empty_like(sent)
        dist.all_to_all_single(returned,received,output_split_sizes=counts,input_split_sizes=[count]*6)
        torch.testing.assert_close(returned,sent,rtol=0,atol=0)
    dist.barrier()
    if rank==0:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(dict(status='passed',devices=os.environ['ASCEND_RT_VISIBLE_DEVICES'],
            expert_counts=counts,attention_groups=[[0,1],[2,3],[4,5]],waves=8,
            scope='Pinned CPU expert map plus HCCL TP2 reductions and uneven EP6 all-to-all roundtrips; no fused-MoE/model'),indent=2)+'\n')
        print('PASS EP6 mapping and uneven HCCL dispatch/combine roundtrips',flush=True)
finally:
    for group in groups:
        if group!=dist.GroupMember.NON_GROUP_MEMBER:dist.destroy_process_group(group)
    dist.destroy_process_group()
