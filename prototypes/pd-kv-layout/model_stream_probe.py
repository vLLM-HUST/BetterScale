"""One-NPU real-page geometry / Store oracle before loading all eight model ranks."""
import argparse,json
from types import SimpleNamespace as NS
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 from dram_store_fixture import dram_store
 with dram_store(a.output/'store',55401) as stores:
  import torch,torch_npu
  from model_stream_export import export_dense
  torch.npu.set_device(0);runner=NS(device=torch.device('npu:0'));ids=torch.tensor([3,1,4,2]);target={};expected={}
  for i in range(10):
   planes={}
   for j,kind in enumerate(('key','value')):
    host=(torch.arange(5*2048,dtype=torch.int32)%(71+i)+j).bfloat16()[:,None,None].expand(-1,1,256).contiguous()
    planes[kind]=NS(tensor=host.view(-1,128,1,256).to(runner.device))
    expected[f'layer{i}/{kind}/head0']=host.view(5,2048,1,256)[ids].flatten(0,1)
   target[f'layer{i}']=NS(**planes)
  rows=[]
  for start in (0,4099):
   header=dict(dense_start=start,cursor=6145,blocks=ids.tolist(),block_size=2048,
               stream_store=dict(prefix=f'pd/stream-oracle/1/start{start}',ports=[55421,55422]))
   torch.npu.synchronize()
   result=export_dense(runner,NS(target=target),0,header)
   for chunk in result['chunks']:
    for name,key in chunk['keys'].items():
     gold=expected[name][chunk['start']:chunk['stop']].view(torch.uint8).numpy().tobytes()
     assert bytes(stores[1].get(key))==gold,(name,chunk['start'])
   rows.append({k:v for k,v in result.items() if k not in ('chunks','streams')})
  (a.output/'complete.json').write_text(json.dumps(dict(status='passed',scope='NPU0 synthetic pages, permuted logical IDs and partial tail; no model or compute overlap',runs=rows),indent=2))
if __name__=='__main__':main()
