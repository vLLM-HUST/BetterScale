"""Bounded FULL-graph observation of cold prefill, without weight dumps.

Capture device copies of layer boundaries, then synchronize/copy to CPU after
replay. This perturbs timing; its results are an instrumented arm, not baseline.
Only the active TP2 owner is observed: up to16 prefills of281 real tokens,
or128 last-token snapshots in the explicit same-owner warm-control arm.
"""
from pathlib import Path
import os


def decoder_layers(named_modules):
    result={}
    for name,module in named_modules:
        parent,_,index=name.rpartition('.')
        if parent.endswith('.layers') and index.isdigit():
            if int(index) in result:raise ValueError('Ambiguous target decoder layer')
            result[int(index)]=module
    return result


def install(worker, output):
    import torch
    from vllm.distributed import get_ep_group
    from vllm.forward_context import get_forward_context
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    rank=get_ep_group().rank_in_group
    native_call=ACLGraphWrapper.__call__
    def python_graph(wrapper,*args,**kwargs):
        context=get_forward_context();previous=context.skip_compiled
        try:
            context.skip_compiled=True
            return native_call(wrapper,*args,**kwargs)
        finally:context.skip_compiled=previous
    ACLGraphWrapper.__call__=python_graph
    if rank not in (2,3):return lambda:None
    runner=worker.model_runner;model=runner.get_model();buffers={};ready=False;count=0
    selected=(0,1,3,7,15,39)
    last_only=bool(os.environ.get('BETTERSCALE_NUMERICAL_LAST_TOKEN_TRACE'))
    layers=decoder_layers(model.named_modules())
    assert set(selected)<=set(layers),list(layers)
    moe_buffers={}
    if os.environ.get("BETTERSCALE_NUMERICAL_MOE_TRACE"):
        from numerical_moe_trace import install as install_moe
        moe_buffers=install_moe(worker,layers[0])
    gdn_buffers={}
    if os.environ.get('BETTERSCALE_NUMERICAL_GDN_TRACE'):
        from numerical_gdn_trace import install as install_gdn
        gdn_buffers=install_gdn(worker,layers[0])
    width=worker.model_config.hf_text_config.hidden_size
    for index in selected:
        pair=tuple(torch.empty((4096,width),dtype=torch.bfloat16,device=runner.device) for _ in range(2))
        layers[index].register_buffer('_probe_hidden',pair[0],persistent=False)
        layers[index].register_buffer('_probe_residual',pair[1],persistent=False)
        buffers[index]=pair
    def hook(index):
        def record(module,args,result):
            if not isinstance(result,tuple) or len(result)!=2:return
            hidden,residual=result
            buffers[index][0][:hidden.shape[0]].copy_(hidden)
            buffers[index][1][:residual.shape[0]].copy_(residual)
        return record
    for index in selected:layers[index].register_forward_hook(hook(index))
    original=ACLGraphWrapper.__call__
    def call(wrapper,*args,**kwargs):
        nonlocal count
        context=get_forward_context();descriptor=context.batch_descriptor
        observe=(ready and wrapper is runner.model and descriptor is not None
                 and (last_only or descriptor.num_tokens==512) and runner.input_batch.num_reqs==1)
        if observe:
            assert count<(128 if last_only else 16),'Trace bound exceeded'
            for tensor in (*moe_buffers.values(),*gdn_buffers.values()):tensor.fill_(float('nan'))
            for pair in buffers.values():
                for tensor in pair:tensor.fill_(float('nan'))
        result=original(wrapper,*args,**kwargs)
        if observe:
            torch.npu.synchronize()
            actual=int(runner.query_start_loc.np[1])
            span=slice(actual-1,actual) if last_only else slice(0,281)
            snapshots={str(i):dict(hidden=h[span].detach().cpu().clone(),
                                  residual=r[span].detach().cpu().clone())
                       for i,(h,r) in buffers.items()}
            assert set(buffers)==set(selected),'Missing captured layer boundaries'
            assert all(torch.isfinite(t).all() for row in snapshots.values() for t in row.values())
            moe={k:v.detach().cpu().clone() for k,v in moe_buffers.items()}
            assert all(torch.isfinite(t).all() for t in moe.values()),'Missing MoE tap'
            gdn_span=slice(0,actual) if os.environ.get('BETTERSCALE_NUMERICAL_RECURRENCE_TRACE') else span
            gdn={k:v[gdn_span].detach().cpu().clone() for k,v in gdn_buffers.items()}
            assert all(torch.isfinite(t).all() for t in gdn.values()),'Missing GDN tap'
            inputs={k:v.detach().cpu().clone() for k,v in kwargs.items()
                    if k in ('input_ids','positions') and isinstance(v,torch.Tensor)}
            root=Path(output);root.mkdir(parents=True,exist_ok=True)
            torch.save(dict(rank=rank,index=count,layers=snapshots,moe=moe,gdn=gdn,inputs=inputs,
                            actual_tokens=actual,last_only=last_only,batch_descriptor=repr(descriptor),skip_compiled=True),root/f'rank{rank}-prefill{count}.pt')
            count+=1
        return result
    ACLGraphWrapper.__call__=call
    def activate():
        nonlocal ready
        assert set(buffers)==set(selected),'Missing observer buffers'
        ready=True
    return activate
