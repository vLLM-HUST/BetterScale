"""Incremental pair-vs-unchanged-native-graph oracle, never a timing path."""
import json
import os
from pathlib import Path
import torch
from torch.utils._pytree import tree_flatten, tree_map
from vllm.forward_context import get_forward_context
from vllm.distributed import get_tp_group
from dp_full_shadow import byte_pools


def check(pair, replay):
    r=pair.shadow_runner;ctx=get_forward_context();old_meta=ctx.attn_metadata
    p=r.vllm_config.parallel_config
    rank=p.data_parallel_rank*p.tensor_parallel_size+get_tp_group().rank_in_group
    path=Path(os.environ['DONOR_DP_OUTPUT'])/f'pingpong-shadow-rank{rank}.json'
    torch.npu.synchronize()
    meta=next(iter(old_meta.values()))
    req=getattr(meta,'req_metadata',meta)
    valid=int(req.query_start_loc[-1]);capacity=ctx.batch_descriptor.num_tokens
    pools=byte_pools(r.kv_caches)
    side=[m._mtp_hidden_buffer for m in r.get_model().modules() if hasattr(m,'_mtp_hidden_buffer')]
    state=pools+side;before=[x.clone() for x in state]
    result=replay();torch.npu.synchronize()
    actual=tree_map(lambda x:x.clone() if isinstance(x,torch.Tensor) else x,result)
    after=[x.clone() for x in state]
    bounds=getattr(r,'_cross_step_bounds',None);upper=None
    row=dict(bank=(pair.sequence-1)%2,valid=valid,capacity=capacity,status='RUNNING',reference='original-native-graph')
    try:
        for x,y in zip(state,before):x.copy_(y)
        if bounds is not None:upper=bounds.reference_begin(ctx)
        entry=pair.reference_catalog[ctx.batch_descriptor]
        entry.aclgraph.replay();torch.npu.synchronize()
        tp=get_tp_group();maximum=0.
        left=tree_flatten(actual)[0];right=tree_flatten(entry.output)[0]
        assert len(left)==len(right)
        for x,y in zip(left,right):
            if not isinstance(x,torch.Tensor):continue
            if x.shape[0]==capacity:n=valid
            elif x.shape[0]*tp.world_size==capacity:n=max(0,min(x.shape[0],valid-tp.rank_in_group*x.shape[0]))
            else:raise AssertionError(f'Unreviewed output layout {x.shape}')
            torch.testing.assert_close(x[:n],y[:n],rtol=.01,atol=1e-6)
            if n:maximum=max(maximum,float((x[:n].float()-y[:n].float()).abs().max()))
        for x,y in zip(after[:len(pools)],pools):assert torch.equal(x,y),'KV backing bytes differ'
        for x,y in zip(after[len(pools):],side):torch.testing.assert_close(x[:valid],y[:valid],rtol=.01,atol=1e-6)
        row.update(status='PASS',output_max_diff=maximum,kv_bytes_equal=True,exact_metadata=upper is not None)
    except Exception as e:
        row.update(status='FAIL',error=str(e));raise
    finally:
        pair.checks.append(row);path.write_text(json.dumps(pair.checks,indent=2))
        if upper is not None:bounds.reference_end(ctx,upper)
        ctx.attn_metadata=old_meta
        for x,y in zip(state,after):x.copy_(y)
        for x,y in zip(tree_flatten(result)[0],tree_flatten(actual)[0]):
            if isinstance(x,torch.Tensor):x.copy_(y)
    return result
