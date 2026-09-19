"""Isolate gate/up GEMM shape dispatch and split cost, including concatenation."""
import json
import os
from pathlib import Path
import statistics
import torch
import torch_npu

root=Path(os.environ['CAPSULE'])
torch.npu.set_device(0)
receipt=dict(status='RUNNING',scope='One910B2, BF16 ND; one fixed random gate/up weight. Unprofiled captured timings, changed inputs, FP32 reference. Splits include cat; packed layout includes one-time extra weight storage, not a production memory claim.',cases=[])
def save(): (root/'receipt.json').write_text(json.dumps(receipt,indent=2))
try:
 with torch.inference_mode():
  torch.manual_seed(1909)
  w=(torch.randn((17408,5120),device='npu',dtype=torch.float32)/5120**.5).to(torch.bfloat16)
  wt=w.T.contiguous()
  extended = os.environ.get('PROBE_NZ') == '1'
  if extended:
   torch.npu.config.allow_internal_format = True
   wnz=torch_npu.npu_format_cast(w,29)
   halves=[v.contiguous() for v in w.chunk(2,dim=0)]
  for n in ((1024,1536) if extended else (512,768,1024,1152,1280,1408,1536,1792,2048)):
   x=torch.randn((n,5120),device='npu',dtype=torch.bfloat16)
   functions={'native':lambda:torch.nn.functional.linear(x,w)}
   if extended:
    functions.update(weight_nz=lambda:torch.nn.functional.linear(x,wnz),
                     split_output=lambda:torch.cat([torch.nn.functional.linear(x,v) for v in halves],dim=-1))
   if n==1536 and not extended:
    functions.update(split1024_512=lambda:torch.cat([torch.nn.functional.linear(x[:1024],w),torch.nn.functional.linear(x[1024:],w)],dim=0),
                     split768=lambda:torch.cat([torch.nn.functional.linear(t,w) for t in x.split(768)],dim=0),
                     split512=lambda:torch.cat([torch.nn.functional.linear(t,w) for t in x.split(512)],dim=0),
                     packed_kn=lambda:torch.mm(x,wt))
   graphs={};case=dict(tokens=n,checks=[],timings=[]);receipt['cases'].append(case);save()
   for arm,fn in functions.items():
    for _ in range(3):fn()
    torch.npu.synchronize();graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):output=fn()
    graphs[arm]=(graph,output)
   for generation in range(3):
    x.copy_(torch.randn_like(x));reference=x.float()@w.float().T
    for arm,(graph,output) in graphs.items():
     graph.replay();torch.npu.synchronize();error=(output.float()-reference).abs()
     valid=bool(torch.isfinite(output).all()) and bool(torch.allclose(output.float(),reference,atol=.03,rtol=.03))
     case['checks'].append(dict(arm=arm,generation=generation,passed=valid,max_abs=float(error.max()),rmse=float((error**2).mean().sqrt())))
     assert valid,case['checks'][-1]
   for arm in list(graphs)+list(reversed(graphs)):
    graph,_=graphs[arm]
    for _ in range(5):graph.replay()
    torch.npu.synchronize();samples=[]
    for _ in range(30):
     a,b=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
     a.record();graph.replay();b.record();b.synchronize();samples.append(a.elapsed_time(b)*1000)
    case['timings'].append(dict(arm=arm,median_us=statistics.median(samples),samples_us=samples))
   print(n,[(r['arm'],round(r['median_us'],2)) for r in case['timings']],flush=True);save()
   if os.environ.get('PROBE_PROFILE') == '1':
    from profiling import ProfileWindow
    prof=ProfileWindow(str(root/'profiles'/str(n)),0,2*len(graphs))
    for _ in range(2):
     for arm,(graph,_) in graphs.items():
      prof.mark(arm);graph.replay();torch.npu.synchronize();prof.step()
    prof.close()
   del graphs
  receipt['status']='PASS'
except BaseException as exc:
 receipt.update(status='FAIL',error=f'{type(exc).__name__}: {exc}');raise
finally:save()
