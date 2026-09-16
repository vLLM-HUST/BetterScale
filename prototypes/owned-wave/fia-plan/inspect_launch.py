import ctypes, os, json
import torch, torch_npu
from pathlib import Path
lib=ctypes.CDLL(os.environ['LD_PRELOAD'])
lib.select_query.argtypes=[ctypes.c_uint64]
torch.npu.set_device(0)
q=torch.randn(4,16,128,device='npu',dtype=torch.bfloat16)
k=torch.randn(32,128,256,device='npu',dtype=torch.bfloat16);v=torch.randn_like(k)
table=torch.arange(32,device='npu',dtype=torch.int32).reshape(4,8)
mask=torch.ones(2048,2048,device='npu',dtype=torch.bool).triu_(1)
lib.select_query(q.data_ptr())
y=torch_npu.npu_fused_infer_attention_score(q,k,v,block_table=table,atten_mask=mask,
 actual_seq_lengths=[1,2,3,4],actual_seq_lengths_kv=[17,129,257,513],num_heads=16,
 num_key_value_heads=2,scale=128**-0.5,block_size=128,input_layout='TND',sparse_mode=3)[0]
torch.npu.synchronize()
Path('pointers.json').write_text(json.dumps({n:t.data_ptr() for n,t in [('q',q),('k',k),('v',v),('table',table),('mask',mask),('out',y)]}))
assert Path('fia-launch.bin').exists()
