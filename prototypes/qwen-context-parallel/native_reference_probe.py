"""Native-only FIA vs CPU on short rows beside a long request; no preload/CP."""
import argparse
from itertools import accumulate
import json
import os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--repeats', type=int, default=16)
p.add_argument('--dynamic', action='store_true')
p.add_argument('--planner', type=Path, help='Explicit control with metadata-only host planning; no CP kernel')
a = p.parse_args()
if os.environ.get('LD_PRELOAD') and not a.planner:
    raise RuntimeError('This control must not preload the BetterScale host adapter')
import torch
import torch_npu

torch.npu.set_device(0)
assert torch.npu.device_count() == 1
assert 1 <= a.repeats <= 64
torch.set_num_threads(8)
torch.manual_seed(173)
receipt = {'scope': 'native FIA only; CPU oracle for 15 short rows; no CP/library preload',
           'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
           'queue': os.environ.get('TASK_QUEUE_ENABLE'), 'dynamic': a.dynamic, 'planner': str(a.planner) if a.planner else None, 'cases': []}
for padding in (False, True):
    lengths = [1024] * 15 + [262144] + ([0] if padding else [])
    queries = [3] * 16 + ([13] if padding else [])
    qends = list(accumulate(queries))
    pages = [(n+127)//128 for n in lengths]
    columns = max(pages)
    table = torch.zeros(len(lengths), columns, dtype=torch.int32)
    offset = 0
    for row, count in enumerate(pages):
        table[row, :count] = torch.arange(offset, offset+count)
        offset += count
    table = table.npu()
    q = torch.randn(sum(queries), 8, 256, dtype=torch.bfloat16, device='npu')
    k = torch.randn(offset, 128, 256, dtype=torch.bfloat16, device='npu')
    v = torch.randn_like(k)
    mask = torch.ones(2048, 2048, device='npu', dtype=torch.bool).triu_(1)
    if a.planner:
        import ctypes
        lib=ctypes.CDLL(str(a.planner))
        u,i=ctypes.c_uint64,ctypes.c_int64
        lib.plan_native_queries.argtypes=[ctypes.POINTER(u),*[ctypes.c_int]*6,ctypes.c_double,ctypes.POINTER(i),ctypes.POINTER(i),ctypes.c_void_p]
        lib.plan_release.argtypes=[ctypes.c_int]
        scratch=torch.zeros(128*1024**2,dtype=torch.uint8,device='npu')
        plan_output=torch.empty_like(q)
        ptrs=(u*7)(*[x.data_ptr() for x in (q,k,v,mask,table,plan_output,scratch)])
        for _ in range(2):
            handle=lib.plan_native_queries(ptrs,len(queries),sum(queries),8,1,offset,columns,
                256**-.5,(i*len(queries))(*lengths),(i*len(queries))(*qends),torch.npu.current_stream().npu_stream)
            assert handle>=0,handle
            assert lib.plan_release(handle)==0
        torch.npu.synchronize()
        del scratch,plan_output,ptrs
    # CPU references are independent of the native kernel and host adapter.
    cpu_k = k[:sum(pages[:15])].cpu().float().reshape(15, 1024, 256)
    cpu_v = v[:sum(pages[:15])].cpu().float().reshape(15, 1024, 256)
    allowed = torch.arange(1024)[None, :] <= 1024-3+torch.arange(3)[:, None]
    rows = []
    for step in range(a.repeats):
        q.mul_(-1)
        edge=(3,511,512,513)[(step//2)%4]
        active=[max(qn,min(n,edge)) if n else 0 for qn,n in zip(queries,lengths)] if a.dynamic and step%2==0 else lengths
        n=active[0]
        allowed=torch.arange(n)[None,:] <= n-3+torch.arange(3)[:,None]
        out = torch_npu.npu_fused_infer_attention_score(
            q, k, v, block_table=table, atten_mask=mask, input_layout='TND', block_size=128,
            actual_seq_lengths=qends, actual_seq_lengths_kv=active,
            num_heads=8, num_key_value_heads=1, scale=256**-.5,
            sparse_mode=3, next_tokens=0)[0]
        torch.npu.synchronize()
        short_q = q[:45].cpu().float().reshape(15, 3, 8, 256).transpose(1, 2)
        scores = torch.matmul(short_q, cpu_k[:, None, :n].transpose(-1, -2))*256**-.5
        truth = torch.matmul(scores.masked_fill(~allowed, -torch.inf).softmax(-1),
                             cpu_v[:, None, :n]).transpose(1, 2)
        actual = out[:45].cpu().float().reshape_as(truth)
        errors = (actual-truth).abs().reshape(15, -1).amax(-1)
        close = torch.isclose(actual, truth, rtol=.02, atol=.003).reshape(15, -1).all(-1)
        rows.append({'step': step, 'short_kv': n, 'max_error': errors.max().item(),
                     'bad_requests': (~close).nonzero().flatten().tolist(),
                     'padding_max': out[48:].float().abs().max().item() if padding else None})
    receipt['cases'].append({'padding': padding, 'rows': rows})
    del q, k, v, table, mask, out
    torch.npu.empty_cache()
receipt['passed_short_rows'] = all(not row['bad_requests'] for case in receipt['cases'] for row in case['rows'])
a.output.write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps({'passed_short_rows': receipt['passed_short_rows'],
                  'failures': [dict(padding=c['padding'], **r) for c in receipt['cases']
                               for r in c['rows'] if r['bad_requests']]}), flush=True)
raise SystemExit(0 if receipt['passed_short_rows'] else 1)
