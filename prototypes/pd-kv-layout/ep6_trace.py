"""One first-layer eager MoE observation per rank; explicit external artifact dir."""
import os
from pathlib import Path

def install(worker):
 import torch
 from vllm.distributed import get_ep_group
 from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
 output=os.environ.get('BETTERSCALE_EP6_TRACE')
 if not output:return
 root=Path(output);root.mkdir(parents=True,exist_ok=True)
 layer=next(m for m in worker.model_runner.model.modules() if isinstance(m,AscendMoERunner))
 method=layer._quant_method;original=method.apply;done=False
 def apply(*args,**kwargs):
  nonlocal done
  result=original(*args,**kwargs)
  if not done and not kwargs.get('enable_force_load_balance',False) and kwargs['x'].shape[0]>=64:
   done=True
   data={k:kwargs[k].detach().cpu() for k in ('x','router_logits','expert_map')}
   data.update(w13=kwargs['layer'].w13_weight.detach().cpu(),w2=kwargs['layer'].w2_weight.detach().cpu(),
      routed=result.routed_out.detach().cpu(),top_k=kwargs['top_k'],renormalize=kwargs['renormalize'],
      tp=layer.moe_config.tp_size,dp=layer.moe_config.dp_size,ep=layer.moe_config.ep_size,
      name=layer.layer_name)
   torch.save(data,root/f'rank{get_ep_group().rank_in_group}.pt')
   print('EP6_TRACE_SAVED',get_ep_group().rank_in_group,tuple(data['x'].shape),flush=True)
  return result
 method.apply=apply
