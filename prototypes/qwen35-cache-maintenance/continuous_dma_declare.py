"""CPU-only: declare.py MODEL_CONFIG OUTPUT_JSON; no State activation."""
import json, sys, torch
from betterscale.live import LiveRuntime, live_runtime
from betterscale.live.llm.qwen35.state import Geometry, Capacity, GDNState, AttentionState
from betterscale.models.qwen35.state_binding import BaselineStateRoot
config=json.load(open(sys.argv[1]))
g=Geometry.from_config(config,tensor_parallel_size=2)
consumers={}
for i,kind in enumerate(g.layer_types):
    c=torch.nn.Module(); c.state_binding_abi=(GDNState if kind=='linear_attention' else AttentionState).binding_abi
    consumers[f'model.layers.{i}.attention']=c
c=torch.nn.Module(); c.state_binding_abi=AttentionState.binding_abi
consumers['draft']=c
with live_runtime(LiveRuntime()):
    root=BaselineStateRoot(g,Capacity(16,20,token_pages=16,prefill_tokens=4096),consumers=consumers,draft_names={'draft'})
packets=[]
for name,s in root.named_states():
    if s is root.continuation.resident_epoch: continue
    count=16 if s.domain is root.pages else 1
    packets.append(dict(name=name,bytes=s.logical_block_bytes*count,dtype=str(s.storage_dtype),shape=[count*s.physical_blocks_per_logical_block,*s.block_shape]))
assert len(packets)==90 and sum(p['bytes'] for p in packets)==118673460
open(sys.argv[2],'w').write(json.dumps(packets,indent=2)+'\n')
print('declared',len(packets),sum(p['bytes'] for p in packets))
