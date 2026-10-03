"""First-MoE FULL graph taps of active DP1 rows; timing-perturbed observation."""
def install(worker, decoder):
    import torch
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
    runner=next(m for m in decoder.mlp.modules() if isinstance(m,AscendMoERunner))
    device=worker.model_runner.device
    width=worker.model_config.hf_text_config.hidden_size
    experts=runner.moe_config.num_experts
    specifications={'mlp_input':(width,torch.bfloat16),'gathered_input':(width,torch.bfloat16),
        'router':(experts,torch.float32),'local_routed':(width,torch.bfloat16),
        'final_routed':(width,torch.bfloat16),'shared':(width,torch.bfloat16)}
    buffers={k:torch.empty((281,w),dtype=d,device=device) for k,(w,d) in specifications.items()}
    def record(key,tensor,gathered=False):
        expected=1536 if gathered else 512
        if tensor.shape[0]==expected:
            start=512 if gathered else 0
            buffers[key].copy_(tensor[start:start+281])
    decoder.mlp.register_forward_pre_hook(lambda module,args:record('mlp_input',args[0]))
    method=runner._quant_method;original_apply=method.apply
    def apply(*args,**kwargs):
        selected=kwargs.get('layer') is runner.routed_experts
        if selected:
            record('gathered_input',kwargs['x'],True)
            record('router',kwargs['router_logits'],True)
        result=original_apply(*args,**kwargs)
        if selected:record('local_routed',result.routed_out,True)
        return result
    method.apply=apply
    original_no_shared=runner.no_shared_forward_impl
    def no_shared(*args,**kwargs):
        result=original_no_shared(*args,**kwargs)
        record('final_routed',result.routed_out if hasattr(result,'routed_out') else result)
        return result
    runner.no_shared_forward_impl=no_shared
    original_shared=runner._forward_shared_experts
    def shared(*args,**kwargs):
        result=original_shared(*args,**kwargs)
        if result is not None:record('shared',result)
        return result
    runner._forward_shared_experts=shared
    return buffers
