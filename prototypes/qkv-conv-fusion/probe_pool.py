"""Serving-boundary leaf: joint QKVZ vs split fused QKV + gate projection.

Includes state bridging, native speculative Conv and projection split costs.
Not an E2E result. One admitted physical NPU, installed wheel, immutable runtime.
"""
import argparse, json, os, random, statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
import torch, torch_npu
from pool_adapter import project_conv

torch.npu.set_device(0);torch.set_num_threads(8)
torch.npu.config.allow_internal_format=False
torch.ops.load_library(str(Path(os.environ['QKV_BASELINE_RUNTIME'])/'vllm_ascend_C.cpython-312-aarch64-linux-gnu.so'))
torch.manual_seed(1511)
rows=[]
with torch.inference_mode():
 for m in (128,512,2048,4096):
  print("BEGIN",m,flush=True)
  n,k=4096,2048
  x=torch.randn(m,k,device='npu',dtype=torch.bfloat16)
  w=(torch.randn(n+2048,k,device='npu')/k**.5).bfloat16()
  cw=(torch.randn(4,n,device='npu')*.2).bfloat16()
  seed=(torch.randn(40,5,n,device='npu')*.1).bfloat16()
  pools={name:seed.clone() for name in ('joint-native','split-fused')}
  cu=torch.empty(18,dtype=torch.int32,device='npu')
  pre=torch.full((17,1),-1,dtype=torch.int32,device='npu')
  ver=torch.full_like(pre,-1);initial=torch.zeros(17,dtype=torch.bool,device='npu')
  accepted=torch.ones(17,dtype=torch.int32,device='npu')
  def prepare(lengths,roles,slots,flags,counts):
   ends=[0]
   for q in lengths:ends.append(ends[-1]+q)
   cu.copy_(torch.tensor(ends+[ends[-1]]*(18-len(ends)),dtype=torch.int32))
   pre.fill_(-1);ver.fill_(-1);initial.zero_();accepted.fill_(1)
   for i,(role,slot,flag,count) in enumerate(zip(roles,slots,flags,counts)):
    (ver if role else pre)[i,0]=slot;initial[i]=flag;accepted[i]=count
  def native():
   projection=x@w.T
   z=projection[:,:n];gate=projection[:,n:]
   y=torch.empty((m,n),dtype=x.dtype,device=x.device)
   for spec,slots in ((False,pre),(True,ver)):
    torch.ops._C_ascend.npu_causal_conv1d_custom(y,z,cw,conv_state=pools['joint-native'],
      bias_opt=None,query_start_loc_opt=cu,cache_indices_opt=slots,
      initial_state_mode_opt=None if spec else initial,
      num_accepted_tokens_opt=accepted if spec else None,
      activation_mode=1,pad_slot_id=-1,run_mode=1 if spec else 0)
   return y,gate,z
  def fused():
   y,z=project_conv(x,w[:n],cw,pools['split-fused'],cu,pre,initial,ver,accepted)
   gate=x@w[n:].T
   return y,gate,z
  funcs={'joint-native':native,'split-fused':fused}
  cases=[([m],[False],[19],[False],[1]),
         ([1,2,m-9,3],[False,False,False,True],[7,30,12,1],[True,False,True,True],[1,1,1,2]),
         ([3,m-7,1],[True,False,True],[25,3,17],[True,False,True],[3,1,1])]
  graphs={}; outputs={}
  prepare(*cases[0])
  for name,fn in funcs.items():
   for _ in range(3):fn()
   g=torch.npu.NPUGraph()
   with torch.npu.graph(g):outputs[name]=fn()
   graphs[name]=g
  for wave,case in enumerate(cases):
   prepare(*case);x.mul_(-.5);cw.mul_(-.75)
   for pool in pools.values():pool.copy_(seed)
   for g in graphs.values():g.replay()
   torch.npu.synchronize()
   live=sum(case[0])
   for j in range(3):
    torch.testing.assert_close(outputs['split-fused'][j][:live],outputs['joint-native'][j][:live],rtol=.03,atol=.03)
   torch.testing.assert_close(pools['split-fused'],pools['joint-native'],rtol=.03,atol=.03)
   untouched=[i for i in range(40) if i not in case[2]]
   for name in funcs:assert torch.equal(pools[name][untouched],seed[untouched])
   for slot,role in zip(case[2],case[1]):
    if not role:
     for pool in pools.values():assert torch.equal(pool[slot,3:],seed[slot,3:])
  # Serving-like mixed layout: independently changing slots, initial flags and
  # speculative accepted counts have already passed against the actual native op.
  samples={name:[] for name in funcs};random.seed(1511)
  for rep in range(9):
   names=list(graphs);random.shuffle(names)
   for name in names:
    for _ in range(5):graphs[name].replay()
    start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(30):graphs[name].replay()
    end.record();end.synchronize()
    samples[name].append(start.elapsed_time(end)*1000/30)
  row={'m':m,'cases':len(cases),'correct':True,'samples_us':samples,
       'median_us':{name:statistics.median(v) for name,v in samples.items()}}
  rows.append(row);print(json.dumps(row),flush=True)
  (a.out/'receipt.json').write_text(json.dumps({'rows':rows},indent=2)+'\n')
(a.out/'complete.json').write_text(json.dumps({'status':'PASS','rows':rows},indent=2)+'\n')
