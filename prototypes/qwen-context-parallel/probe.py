"""Q8/KV1/D256 leaf gate, device-authoritative endpoints and padding; external NPU lease required."""
import argparse
import ctypes
import hashlib
from itertools import accumulate
import json
import os
from pathlib import Path
import sys

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--kernel',type=Path,required=True)
p.add_argument('--native',type=Path,required=True)
p.add_argument('--device-lengths',action='store_true')
a=p.parse_args();a.output.mkdir(exist_ok=False)
sys.path.insert(0,str(a.source));from plan import schedule,encode
import torch
import torch_npu

torch.npu.set_device(0);torch.set_num_threads(8);assert torch.npu.device_count()==1
torch.manual_seed(173)
lib=ctypes.CDLL(str(a.native));kernel=ctypes.CDLL(str(a.kernel))
u,i=ctypes.c_uint64,ctypes.c_int64
lib.plan_native_queries.argtypes=[ctypes.POINTER(u),*[ctypes.c_int]*6,ctypes.c_double,ctypes.POINTER(i),ctypes.POINTER(i),ctypes.c_void_p]
lib.plan_metadata.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t]
lib.plan_release.argtypes=[ctypes.c_int]
kernel.lane_launch.argtypes=[ctypes.c_void_p,ctypes.POINTER(u)];kernel.lane_launch.restype=None
cases=[('edges',[1,2,3,3],[127,512,513,1025]),
       ('c16-short',[3]*16,[1024]*16),
       ('nonuniform',[1,2,3,1,2,3],[1024,2048,4096,8192,32768,65536]),
       ('moderate',[3]*16,[32768]*14+[262144]*2),
       ('extreme',[3]*16,[1024]*15+[262144]),
       ('long16',[3]*16,[262144]*16)]
(a.output/'protocol.json').write_text(json.dumps(dict(device=int(os.environ['ASCEND_RT_VISIBLE_DEVICES']),cases=cases,
    kernel_sha256=hashlib.sha256(a.kernel.read_bytes()).hexdigest(),native_sha256=hashlib.sha256(a.native.read_bytes()).hexdigest(),
    scope='Two captured banks; device-only length changes and zero-KV padding' if a.device_lengths else 'Exact host lengths; two captured banks',torch=torch.__version__,torch_npu=torch_npu.__version__),indent=2)+'\n')
receipts=[]
for name,qs,upper in cases:
    live_qs,live_upper=list(qs),list(upper)
    if a.device_lengths:
        qs=qs+[13];upper=upper+[0]
    qends=list(accumulate(qs));pages=[(n+127)//128 for n in upper];columns=max(pages)
    table=torch.zeros(len(qs),columns,dtype=torch.int32);offset=0
    for row,count in enumerate(pages):table[row,:count]=torch.arange(offset,offset+count);offset+=count
    table=table.npu();q=torch.randn(sum(qs),8,256,device='npu',dtype=torch.bfloat16)
    k=torch.randn(offset,128,256,device='npu',dtype=torch.bfloat16);v=torch.randn_like(k)
    original=(k.clone(),v.clone(),table.clone())
    mask=torch.ones(2048,2048,device='npu',dtype=torch.bool).triu_(1)
    ql=torch.tensor(qends,dtype=torch.int64,device='npu')
    def native(lengths):
        return torch_npu.npu_fused_infer_attention_score(q,k,v,block_table=table,atten_mask=mask,input_layout='TND',block_size=128,
            actual_seq_lengths=qends,actual_seq_lengths_kv=lengths,num_heads=8,num_key_value_heads=1,scale=256**-.5,sparse_mode=3,next_tokens=0)[0]
    banks=[]
    for bank in range(2):
        workspace=torch.full((128*1024**2+8192,),165,dtype=torch.uint8,device='npu')
        output=torch.full((q.numel()+4096,),-123,dtype=q.dtype,device='npu');out=output[2048:-2048].view_as(q)
        kl=torch.tensor(upper,dtype=torch.int64,device='npu');meta=torch.empty(4096,dtype=torch.uint8,device='npu')
        ptrs=(u*7)(*[x.data_ptr() for x in (q,k,v,mask,table,out,workspace[4096:-4096])])
        handle=lib.plan_native_queries(ptrs,len(qs),sum(qs),8,1,offset,columns,256**-.5,(i*len(qs))(*upper),(i*len(qs))(*qends),torch.npu.current_stream().npu_stream)
        assert handle>=0,handle
        raw=ctypes.create_string_buffer(2528);assert lib.plan_metadata(handle,raw,2528)==2528
        assert lib.plan_release(handle)==0
        (a.output/f'{name}-bank{bank}.bin').write_bytes(raw.raw)
        plan=schedule(live_upper,live_qs);meta.copy_(torch.tensor(list(encode(raw.raw,plan)),dtype=torch.uint8))
        ptrs=(u*10)(*[x.data_ptr() for x in (q,k,v,mask,table,out,ql,kl,workspace[4096:-4096],meta)])
        def call(ptrs=ptrs):kernel.lane_launch(torch.npu.current_stream().npu_stream,ptrs)
        call();torch.npu.synchronize()
        torch.testing.assert_close(out,native(upper),rtol=.02,atol=.003)
        graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph):call()
        banks.append(dict(workspace=workspace,output=output,out=out,kl=kl,meta=meta,raw=raw.raw,ptrs=ptrs,graph=graph))
    errors=[];states=[]
    for step in range(8):
        lengths=(live_upper if step%2 else [max(qn,min(n,(3,511,512,513)[step//2])) for qn,n in zip(live_qs,live_upper)])
        if a.device_lengths:lengths=lengths+[0]
        q.mul_(-1)
        ref=native(lengths)
        for b in banks:
            plan=schedule(live_upper,live_qs) if a.device_lengths else schedule(lengths,qs);states.append(plan['split_nodes'])
            b['kl'].copy_(torch.tensor(lengths,dtype=torch.int64))
            b['meta'].copy_(torch.tensor(list(encode(b['raw'],plan)),dtype=torch.uint8))
            b['out'].fill_(float('nan'));b['workspace'][4096:-4096].zero_()
            b['graph'].replay();torch.npu.synchronize()
            torch.testing.assert_close(b['out'],ref,rtol=.02,atol=.003)
            errors.append((b['out'].float()-ref.float()).abs().max().item())
            assert torch.all(b['workspace'][:4096]==165).item() and torch.all(b['workspace'][-4096:]==165).item()
            assert torch.all(b['output'][:2048]==-123).item() and torch.all(b['output'][-2048:]==-123).item()
        if a.device_lengths:assert torch.all(b['out'][sum(live_qs):]==0).item()
        if name=='edges':
            start=0;page=0
            for qn,kvn,npages in zip(live_qs,lengths,pages):
                qq=q[start:start+qn].cpu().float().transpose(0,1)
                kk=k[page:page+npages].cpu().float().reshape(-1,256)[:kvn]
                vv=v[page:page+npages].cpu().float().reshape(-1,256)[:kvn]
                scores=(qq@kk.T)*256**-.5
                allowed=torch.arange(kvn)[None,:]<=kvn-qn+torch.arange(qn)[:,None]
                expected=(scores.masked_fill(~allowed,-torch.inf).softmax(-1)@vv).transpose(0,1)
                torch.testing.assert_close(banks[0]['out'][start:start+qn].cpu().float(),expected,rtol=.02,atol=.003)
                start+=qn;page+=npages
    for actual,expected in zip((k,v,table),original):torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    receipts.append(dict(case=name,max_error=max(errors),split_states=states,graphs=2,replays=16,guards=True,input_immutable=True))
    (a.output/'progress.json').write_text(json.dumps(receipts,indent=2)+'\n');print(name,'PASS',max(errors),flush=True)
    del banks,b,graph,out,output,workspace,kl,meta,ptrs,call,ref,q,k,v,original,table,mask,ql
    torch.npu.empty_cache()
(a.output/'complete.json').write_text(json.dumps(dict(status='passed',receipts=receipts),indent=2)+'\n')
