"""Native distributed-greedy oracle; can run inside the real Worker lifecycle."""
import json
import os
from pathlib import Path
import torch
import torch_npu
from vllm.distributed import get_tp_group


def check_initialized(directory):
    from vllm_ascend.spec_decode.llm_base_proposer import greedy_sample
    rank=get_tp_group().rank_in_group
    cases=[]
    errors=[]
    cpu_group=get_tp_group().cpu_group
    keepers=[]
    for rows in (1,3,8,16):
        torch.manual_seed(170+rows)
        full = torch.randn(rows,248320,dtype=torch.bfloat16)
        # Equal maxima within/across partitions must choose lowest global ID.
        full[0].fill_(-1);full[0,7]=8;full[0,124167]=8
        if rows>1:
            full[1].fill_(-float('inf'))
            full[2,124163]=float('inf');full[2,124165]=float('inf')
        logits=full[:,rank*124160:(rank+1)*124160].contiguous().to('npu')
        expected=full.argmax(-1)
        for _ in range(3): eager=greedy_sample(logits)
        torch.npu.synchronize()
        assert torch.equal(eager.cpu(),expected),(rank,rows,'eager')
        graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output=greedy_sample(logits)
            # Like a real head, the collective consumes a graph-produced tensor.
            control=get_tp_group().all_gather(logits.clone(),dim=-1).argmax(-1)
        keepers.append((graph,logits,output,control))
        for replay in range(3):
            if replay:
                full.fill_(-2); full[:,(replay%2)*124160+11+replay]=9
                logits.copy_(full[:,rank*124160:(rank+1)*124160].contiguous())
                expected=full.argmax(-1)
            torch.npu.synchronize()
            torch.distributed.barrier(group=cpu_group)
            assert torch.equal(logits.cpu(),full[:,rank*124160:(rank+1)*124160]), ('input',rank,rows,replay)
            graph.replay();torch.npu.synchronize()
            actual, baseline = output.cpu(), control.cpu()
            row={'rank':rank,'rows':rows,'replay':replay,'actual':actual.tolist(),
                 'full_gather':baseline.tolist(),'expected':expected.tolist()}
            print(json.dumps(row),flush=True)
            if not torch.equal(actual,expected) or not torch.equal(baseline,expected): errors.append(row)
            torch.distributed.barrier(group=cpu_group)
        cases.append({'rows':rows,'local_vocab':124160,'eager':True,'graph_replays':3,'exact':not any(e['rows']==rows for e in errors)})
    Path(directory,f'greedy-rank{rank}.json').write_text(json.dumps({'status':'FAIL' if errors else 'PASS','cases':cases,'errors':errors},indent=2)+'\n')
    torch.distributed.barrier(group=cpu_group)
    assert not errors, errors
