"""First GDN layer boundary taps; no operator replacement."""
def install(worker, decoder):
    import torch
    device=worker.model_runner.device
    attn=decoder.linear_attn
    buffers={}
    def reserve(key,width):
        buffers[key]=torch.empty((4096,width),dtype=torch.bfloat16,device=device)
    for key in ('embedding','input_norm','gdn_core','gdn_gate','gdn_norm','projection'):
        reserve(key,2048)
    def record(key,value):
        if isinstance(value,tuple):value=value[0]
        value=value.reshape(-1,buffers[key].shape[1])
        buffers[key][:value.shape[0]].copy_(value)
    decoder.input_layernorm.register_forward_pre_hook(lambda m,a:record('embedding',a[0]))
    decoder.input_layernorm.register_forward_hook(lambda m,a,r:record('input_norm',r))
    for name in ('in_proj_qkvz','in_proj_ba'):
        module=getattr(attn,name)
        reserve(name,module.weight.shape[0])
        module.register_forward_hook(lambda m,a,r,key=name:record(key,r))
    def before_norm(module,args):
        record('gdn_core',args[0]);record('gdn_gate',args[1])
    attn.norm.register_forward_pre_hook(before_norm)
    attn.norm.register_forward_hook(lambda m,a,r:record('gdn_norm',r))
    attn.out_proj.register_forward_hook(lambda m,a,r:record('projection',r))
    import os
    if os.environ.get('BETTERSCALE_NUMERICAL_RECURRENCE_TRACE'):
        import importlib
        preprocessing=importlib.import_module('betterscale.models.qwen35.preprocess')
        mixed=importlib.import_module('betterscale.models.qwen35.mixed_core')
        active=False;seen=False
        for name,width,dtype in (('q',1024,torch.bfloat16),('k',1024,torch.bfloat16),
                                 ('v',2048,torch.bfloat16),('g',16,torch.float32),
                                 ('beta',16,torch.bfloat16)):
            buffers[name]=torch.empty((4096,width),dtype=dtype,device=device)
        original_preprocess=preprocessing.preprocess
        def preprocess(*args,**kwargs):
            nonlocal seen
            result=original_preprocess(*args,**kwargs)
            if active and not seen:
                for key,value in zip(('q','k','v','g','beta'),result):record(key,value)
                seen=True
            return result
        preprocessing.preprocess=preprocess;mixed.preprocess=preprocess
        for key in ('state_before','state_after'):
            buffers[key]=torch.empty((1,16*128*128),dtype=torch.float32,device=device)
        buffers['state_used']=torch.empty((1,1),dtype=torch.float32,device=device)
        original_core=attn._forward_core
        def core(*args,**kwargs):
            nonlocal active,seen
            active=True;seen=False
            try:
                from vllm.forward_context import get_forward_context
                ctx=get_forward_context()
                if ctx.attn_metadata is None:return original_core(*args,**kwargs)
                meta=ctx.attn_metadata[attn.prefix].owned
                if meta.decode:
                    for key in ('state_before','state_after'):buffers[key].zero_()
                    buffers['state_used'].fill_(-1)
                    return original_core(*args,**kwargs)
                indices=meta.prefill.state[:1,0].long()
                used=meta.prefill.state[:1,1].bool().reshape(1,1)
                bank=attn.kv_cache[1]
                before=torch.index_select(bank,0,indices).reshape(1,-1)
                buffers['state_before'].copy_(torch.where(used,before,torch.zeros_like(before)))
                buffers['state_used'].copy_(used)
                result=original_core(*args,**kwargs)
                buffers['state_after'].copy_(torch.index_select(bank,0,indices).reshape(1,-1))
                return result
            finally:active=False
        attn._forward_core=core
    return buffers
