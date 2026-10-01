"""Fixed-input DP3 reduce-scatter repeatability; no model or KV state."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import torch
import torch_npu
import torch.distributed as dist

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
assert int(os.environ['WORLD_SIZE'])==6 and torch.npu.device_count()==6
torch.set_num_threads(4);torch.npu.set_device(local)
dist.init_process_group('hccl',timeout=timedelta(seconds=120))
groups=[]
try:
    for parity in (0,1):groups.append(dist.new_group(list(range(parity,6,2)),backend='hccl'))
    group=groups[rank%2]
    # CPU generated immutable inputs: distinct contributions per EP rank.
    generator=torch.Generator().manual_seed(71+rank)
    host=(torch.randn((1536,2048),generator=generator)*0.01).bfloat16()
    x=host.npu();y=torch.empty((512,2048),dtype=torch.bfloat16,device='npu')
    def reduce():dist.reduce_scatter_tensor(y,x,group=group)
    reduce();torch.npu.synchronize()
    gathered=[torch.empty_like(x) for _ in range(3)]
    dist.all_gather(gathered,x,group=group);torch.npu.synchronize()
    start=(rank//2)*512
    contributions=[t.cpu()[start:start+512] for t in gathered]
    oracle=sum(t.float() for t in contributions).bfloat16()
    a.output.mkdir(parents=True,exist_ok=True)
    if rank==0:
        (a.output/'protocol.json').write_text(json.dumps(dict(
            input_shape=[1536,2048],output_shape=[512,2048],dtype='bfloat16',
            groups=[[0,2,4],[1,3,5]],repeats_per_mode=32,
            environment={k:os.environ.get(k) for k in ('ASCEND_RT_VISIBLE_DEVICES','HCCL_OP_EXPANSION_MODE','HCCL_DETERMINISTIC','TASK_QUEUE_ENABLE')}),indent=2))
    rows=[]
    for mode in ('eager','full_graph'):
        graph=None
        if mode=='full_graph':
            graph=torch.npu.NPUGraph()
            with torch.npu.graph(graph):reduce()
            torch.npu.synchronize()
        baseline=None
        for repeat in range(32):
            if graph is None:reduce()
            else:graph.replay()
            torch.npu.synchronize()
            result=y.cpu().clone()
            if baseline is None:baseline=result.clone()
            delta=(result.float()-baseline.float()).abs()
            rows.append(dict(mode=mode,repeat=repeat,equal=torch.equal(result,baseline),
                changed=int(torch.count_nonzero(delta)),max_abs=float(delta.max()),
                fp32_oracle_max_abs=float((result.float()-oracle.float()).abs().max())))
        assert torch.equal(x.cpu(),host),'Input mutated'
    a.output.mkdir(parents=True,exist_ok=True)
    torch.save(dict(inputs=contributions,reference=baseline),a.output/f'rank{rank}.pt')
    (a.output/f'rank{rank}.json').write_text(json.dumps(rows,indent=2))
    print('RANK',rank,[(mode,sum(not r['equal'] for r in rows if r['mode']==mode),
                       max(r['max_abs'] for r in rows if r['mode']==mode)) for mode in ('eager','full_graph')],flush=True)
    dist.barrier()
finally:
    for group in groups:
        if group!=dist.GroupMember.NON_GROUP_MEMBER:dist.destroy_process_group(group)
    dist.destroy_process_group()

