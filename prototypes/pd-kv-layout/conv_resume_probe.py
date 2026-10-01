"""Pinned native TP2 target prefill convolution, CPU history restore, CPU oracle.

Tests 3-token canonical history only; speculative extended history is not covered.
"""
import argparse
import json
from pathlib import Path
import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); a.output.mkdir(exist_ok=False,parents=True)
torch.set_num_threads(4); torch.manual_seed(174)
torch.npu.set_device(0); assert torch.npu.device_count()==1
assert enable_custom_op()
weight=(torch.randn(4,4096)*.2).bfloat16()
w=weight.npu()
bank_cpu=(torch.randn(8,3,4096)*.2).bfloat16()
bank=bank_cpu.npu(); x=torch.zeros(256,4096,dtype=torch.bfloat16).npu()
cu=torch.zeros(5,dtype=torch.int32).npu()
slots=torch.tensor([[0],[1],[2],[-1]],dtype=torch.int32).npu()
flags=torch.ones(4,dtype=torch.bool).npu()
def prepare(lengths):
    ends=[0]
    for n in lengths: ends.append(ends[-1]+n)
    cu.copy_(torch.tensor(ends+[ends[-1]],dtype=torch.int32))
def call():
    out=torch.empty_like(x)
    torch.ops._C_ascend.npu_causal_conv1d_custom(out,x,w,conv_state=bank,
        bias_opt=None,query_start_loc_opt=cu,cache_indices_opt=slots,
        initial_state_mode_opt=flags,num_accepted_tokens_opt=None,
        activation_mode=1,pad_slot_id=-1,run_mode=0)
    return out
prepare([3,65,128]);call();torch.npu.synchronize()
graph=torch.npu.NPUGraph()
with torch.npu.graph(graph): output=call()
bank.copy_(bank_cpu)
rows=[]; owners=[0,1,2]
for wave,lengths in enumerate(([3,65,128],[1,127,3],[65,1,129],[128,3,65])*2):
    if wave==4:
        checkpoint=bank[owners[0]].cpu().clone()
        torch.testing.assert_close(checkpoint,bank_cpu[owners[0]],rtol=0,atol=0)
        bank[6].copy_(checkpoint);bank_cpu[6].copy_(checkpoint)
        owners[0]=6;slots[0,0]=6
    xc=(torch.randn(x.shape)*.2).bfloat16();x.copy_(xc);prepare(lengths)
    truth=[];offset=0
    for owner,n in zip(owners,lengths):
        joined=torch.cat([bank_cpu[owner],xc[offset:offset+n]])
        value=sum(joined[i:i+n].float()*weight[i].float() for i in range(4))
        truth.append(torch.nn.functional.silu(value).bfloat16())
        bank_cpu[owner].copy_(joined[-3:]);offset+=n
    graph.replay();torch.npu.synchronize()
    actual=output[:offset].cpu();expected=torch.cat(truth)
    torch.testing.assert_close(actual.float(),expected.float(),rtol=.02,atol=.003)
    torch.testing.assert_close(bank.cpu(),bank_cpu,rtol=0,atol=0)
    rows.append(dict(wave=wave,lengths=lengths,max_error=(actual.float()-expected.float()).abs().max().item()))
    (a.output/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
(a.output/'complete.json').write_text(json.dumps(dict(status='passed',rows=rows,
    scope='TP2 prefill conv graph, canonical 3-token CPU history/new-slot resume; not speculative history'),indent=2)+'\n')
print('PASS',max(r['max_error'] for r in rows))
