"""CPU-only first-layer local MoE oracle and exact source weight identity."""
import argparse,json
from pathlib import Path
import torch
from safetensors import safe_open
p=argparse.ArgumentParser(description=__doc__);p.add_argument('trace',type=Path);p.add_argument('--model',type=Path,default=Path('/data/shared_models/modelscope_cache/Qwen/Qwen3.5-35B-A3B'));a=p.parse_args()
torch.set_num_threads(4)
index=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
results=[]
for rank in range(6):
 d=torch.load(a.trace/f'rank{rank}.pt',map_location='cpu',weights_only=True)
 owned=torch.where(d['expert_map']>=0)[0]
 weights={}
 for name,key in [('w13','gate_up_proj'),('w2','down_proj')]:
  full='model.language_model.layers.0.mlp.experts.'+key
  with safe_open(a.model/index[full],framework='pt',device='cpu') as f:
   original=f.get_slice(full)[int(owned[0]):int(owned[-1])+1].transpose(-1,-2).contiguous()
  weights[name]=dict(shape=list(d[name].shape),source_shape=list(original.shape),equal=torch.equal(original,d[name]),max_error=float((original.float()-d[name].float()).abs().max()) if original.shape==d[name].shape else None)
 rows=torch.tensor(sorted(set([0,1,d['x'].shape[0]//3-1,d['x'].shape[0]//3,2*d['x'].shape[0]//3,d['x'].shape[0]-1])))
 x=d['x'][rows].float();scores=torch.softmax(d['router_logits'][rows].float(),-1)
 probs,ids=torch.topk(scores,d['top_k'],-1)
 if d['renormalize']:probs=probs/probs.sum(-1,keepdim=True)
 probs=probs.bfloat16().float();ref=torch.zeros_like(x)
 for expert in owned.tolist():
  ri,slot=torch.where(ids==expert)
  if not ri.numel():continue
  local=int(d['expert_map'][expert]);h=(x[ri]@d['w13'][local].float()).bfloat16().float();gate,up=h.chunk(2,-1)
  h=(torch.nn.functional.silu(gate)*up).bfloat16().float()
  out=(h@d['w2'][local].float()).bfloat16().float()
  ref.index_add_(0,ri,out*probs[ri,slot,None])
 got=d['routed'][rows].float();diff=(got-ref).abs()
 row=dict(rank=rank,shape=list(d['x'].shape),config=[d['tp'],d['dp'],d['ep']],weights=weights,
  max_error=float(diff.max()),mean_error=float(diff.mean()),max_reference=float(ref.abs().max()),max_actual=float(got.abs().max()))
 results.append(row);print(json.dumps(row),flush=True)
(a.trace/'analysis.json').write_text(json.dumps(results,indent=2))
