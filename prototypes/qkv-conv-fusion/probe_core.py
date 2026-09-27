"""Composition gate in the actual pinned BetterScale MTP runtime, no model load."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
import torch,torch_npu
torch.npu.set_device(0);torch.set_num_threads(4)
from vllm_ascend.utils import enable_custom_op
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
assert enable_custom_op();init_device_properties_triton()
from betterscale.models.qwen import check_runtime
check_runtime();check_runtime('qwen_mixed_pins.json')
from service_metadata import Core
from pool_adapter import project_conv

torch.manual_seed(1521);torch.npu.config.allow_internal_format=False
rows=[]
with torch.inference_mode():
 for m in (2048,4096):
  print('BEGIN_CORE',m,flush=True)
  meta=Core(m,'npu')
  x=(torch.randn(m,2048,device='npu')*.03).bfloat16()
  w=(torch.randn(6144,2048,device='npu')/2048**.5).bfloat16()
  cw=(torch.randn(4,4096,device='npu')*.2).bfloat16()
  log=torch.randn(16,device='npu')*.1;bias=torch.randn(16,device='npu')*.1
  a_gate=(torch.randn(m,16,device='npu')*.1).bfloat16();b_gate=torch.randn_like(a_gate)*.1
  state_seed=torch.randn(40,16,128,128,device='npu')*.01
  conv_seed=(torch.randn(40,5,4096,device='npu')*.1).bfloat16()
  pools={name:(conv_seed.clone(),state_seed.clone()) for name in ('baseline','fusion')}
  cases=[([m],[False],[[19,20,21]],[1],[False]),
   ([3,m-9,1,2],[True,False,False,True],[[3,4,5],[12,13,14],[23,24,25],[28,29,30]],[2,1,1,1],[True,True,False,True]),
   ([m-4,1],[False,True],[[1,2,3],[32,33,34]],[1,3],[False,True])]
  def baseline():
   projection=x@w.T
   return meta(projection[:,:4096],a_gate,b_gate,cw,log,bias,*pools['baseline'])
  def fusion():
   y,z=project_conv(x,w[:4096],cw,pools['fusion'][0],meta.cu,meta.prefill_conv,meta.initial,meta.verify_conv,meta.accepted)
   return meta.after_conv(y,a_gate,b_gate,log,bias,pools['fusion'][1])
  graphs={};outputs={}
  meta.prepare(*cases[0])
  for name,fn in (('baseline',baseline),('fusion',fusion)):
   fn();torch.npu.synchronize()
   graph=torch.npu.NPUGraph()
   with torch.npu.graph(graph):outputs[name]=fn()
   graphs[name]=graph
  for wave,case in enumerate(cases):
   meta.prepare(*case)
   x.mul_(-.75);cw.mul_(-.75)
   for conv,state in pools.values():conv.copy_(conv_seed);state.copy_(state_seed)
   for g in graphs.values():g.replay()
   torch.npu.synchronize()
   length=sum(case[0]);errs={}
   for label,left,right in (('output',outputs['fusion'][:,:length],outputs['baseline'][:,:length]),
     ('conv',pools['fusion'][0],pools['baseline'][0]),('state',pools['fusion'][1],pools['baseline'][1])):
    torch.testing.assert_close(left,right,rtol=.03,atol=.003)
    errs[label]=(left.float()-right.float()).abs().max().item()
   rows.append(dict(m=m,wave=wave,max_abs=errs));print(rows[-1],flush=True)
(a.out/'complete.json').write_text(json.dumps(dict(status='PASS',rows=rows),indent=2)+'\n')
