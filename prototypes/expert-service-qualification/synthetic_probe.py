"""TP1/TP8 benchmark sampler oracle; independent fixtures, real native argmax."""
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
import torch_npu
rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);assert world in (1,2,8)
torch.npu.set_device(rank)
from vllm.distributed import init_distributed_environment, initialize_model_parallel, get_tp_group
from vllm.config import VllmConfig, set_current_vllm_config
with set_current_vllm_config(VllmConfig()):
    init_distributed_environment(world_size=world,rank=rank,local_rank=rank,backend='hccl')
    initialize_model_parallel(tensor_model_parallel_size=world,backend='hccl')
    import vllm_ascend.ascend_config as ac
    import vllm_ascend.sample.rejection_sampler as ascend
    from betterscale.patches import benchmark_mtp as shim
    root=Path(os.environ['CAPSULE']);root.mkdir(exist_ok=True)
    original=shim.sample
    captured={}
    def observe(*args):
        result=original(*args)
        captured['uniform']=args[4].cpu().numpy()
        return result
    shim.sample=observe
    results=[]
    for reduce in (False,True):
        ac.get_ascend_config=lambda:NS(enable_reduce_sample=reduce,
            rejection_sampler_config=NS(enable_block_verify=False,enable_entropy_verify=False))
        for rates in ([0.,0.],[1.,1.],[.8,.4]):
            for mixed in (False,True):
                rng=np.random.default_rng(17);rows=4096
                counts=np.arange(rows,dtype=np.int32)%3 if mixed else np.full(rows,2,dtype=np.int32)
                cu=np.cumsum(counts,dtype=np.int32);n=int(cu[-1])
                draft=rng.integers(0,32,size=n,dtype=np.int32)
                if mixed:draft[::13]=-1
                full=rng.standard_normal((n,32)).astype(np.float32)
                target=full.argmax(-1);bonus=rng.integers(0,32,size=rows,dtype=np.int32)
                logits=full[:,rank*(32//world):(rank+1)*(32//world)].copy() if reduce else full
                config=NS(speculative_config=NS(method='mtp',num_speculative_tokens=2,
                    rejection_sample_method='synthetic',synthetic_acceptance_rates=rates),
                    parallel_config=NS(tensor_parallel_size=world))
                # Independent leaf cases, no serving model or outstanding work.
                # Production deliberately rejects changing rates in one process.
                if hasattr(ascend,'_betterscale_synthetic_rates'):
                    del ascend._betterscale_synthetic_rates
                shim.install(config)
                # Intentionally inconsistent local RNG states; broadcast must heal it.
                torch.npu.manual_seed(700+rank)
                d,c,l,b=[torch.from_numpy(x).to('npu') for x in (draft,cu,logits,bonus)]
                output=ascend.rejection_sample(d,counts.tolist(),2,c,None,l,b,NS(all_greedy=True,generators={}))
                actual=output.cpu().numpy();uniform=captured['uniform']
                conditional=[rates[0],rates[1]/rates[0] if rates[0] else 0.]
                expected=np.full((rows,3),-1,dtype=np.int32);start=0
                for i,count in enumerate(counts):
                    for pos in range(count):
                        accepted=uniform[start+pos]<conditional[pos] and draft[start+pos]>=0
                        expected[i,pos]=draft[start+pos] if accepted else target[start+pos]
                        if not accepted:break
                    else:expected[i,count]=bonus[i]
                    start+=count
                assert np.array_equal(actual,expected),(rank,reduce,rates,mixed,'oracle')
                gathered=get_tp_group().all_gather(output.unsqueeze(0),dim=0).cpu().numpy()
                assert all(np.array_equal(gathered[0],value) for value in gathered),'TP disagreement'
                observed=float((actual>=0).sum()/rows)
                if not mixed:assert abs(observed-(1+sum(rates)))<.05,(observed,rates)
                results.append(dict(reduce=reduce,rates=rates,mixed=mixed,rows=rows,
                                    exact_oracle=True,tp_agreement=True,observed_length=observed))
                if rank==0:print(results[-1],flush=True)
    (root/f'rank-{rank}.json').write_text(json.dumps({'status':'PASS','scope':'sampler fixture, not serving integration','world_size':world,'cases':results},indent=2)+'\n')
    torch.distributed.barrier();torch.distributed.destroy_process_group()
