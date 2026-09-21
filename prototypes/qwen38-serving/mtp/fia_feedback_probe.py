"""Bounded N+2 ingress/replay/egress with device-authored FIA KV lengths.

Synthetic acceptance, not a model/service scheduler. Host plans an upper bound;
no acceptance or actual KV length is read before enqueueing the following wave.
"""
import ctypes
import json
import os
from collections import deque
from pathlib import Path

import torch
import torch_npu


def main():
    torch.npu.set_device(0)
    torch.manual_seed(173)
    lib = ctypes.CDLL(os.environ['BETTERSCALE_FIA_LIBRARY'])
    u64, i64 = ctypes.c_uint64, ctypes.c_int64
    lib.plan_native_queries.argtypes = [ctypes.POINTER(u64), *[ctypes.c_int]*6,
        ctypes.c_double, ctypes.POINTER(i64), ctypes.POINTER(i64), ctypes.c_void_p]
    lib.plan_bind.argtypes = [ctypes.c_int, ctypes.POINTER(u64)]
    lib.plan_metadata.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    lib.plan_bind_metadata.argtypes = [ctypes.c_int, u64]
    lib.plan_launch.argtypes = [ctypes.c_int, ctypes.c_void_p]

    def check(rc):
        if rc < 0:
            raise RuntimeError(f'FIA status {rc}')
        return rc

    results = []
    draft_padding = os.environ.get('FIA_DRAFT_PADDING') == '1'
    cases = ([('draft', [1]*8+[capacity-8], [4096]*8+[0], [1]*8+[0])
              for capacity in (24,2048)] if draft_padding else [
        ('verify', [3]*8, [4096]*8, [1]*8),
        ('mixed', [3,33,97,257,122], [4096,33,97,257,0], [1,0,0,0,0])])
    for kind, qs, upper, active in cases:
        rows, width, columns = len(qs), sum(qs), 32
        offsets = torch.tensor(qs).cumsum(0).tolist()
        valid = 8 if draft_padding else (width if kind == 'verify' else 390)
        pages = rows*columns
        k = torch.randn(pages,128,512,device='npu',dtype=torch.bfloat16)
        v = torch.randn_like(k)
        mask = torch.ones(2048,2048,device='npu',dtype=torch.bool).triu_(1)
        table = torch.arange(pages,device='npu',dtype=torch.int32).reshape(rows,columns)
        seed = [3070 if flag else n for n,flag in zip(upper,active)]
        lengths = torch.tensor(seed,device='npu',dtype=torch.int64)
        counter = torch.zeros(1,device='npu',dtype=torch.int64)
        axis = torch.arange(rows,device='npu',dtype=torch.int64)
        live = torch.tensor(active,device='npu',dtype=torch.int64)
        ingress, egress = torch.npu.Stream(), torch.npu.Stream()
        compute = torch.npu.current_stream()
        banks = []
        for bank in range(2):
            q = torch.randn(width,12,256,device='npu',dtype=torch.bfloat16)
            out = torch.empty_like(q)
            ql = torch.tensor(offsets,device='npu',dtype=torch.int64)
            kl = torch.tensor(upper,device='npu',dtype=torch.int64)
            gm = torch.empty(4096,device='npu',dtype=torch.uint8)
            scratch = torch.empty(128*1024**2,device='npu',dtype=torch.uint8)
            ptrs = (u64*7)(*[x.data_ptr() for x in (q,k,v,mask,table,out,scratch)])
            plan = check(lib.plan_native_queries(ptrs,rows,width,12,2,pages,columns,
                256**-.5,(i64*rows)(*upper),(i64*rows)(*offsets),compute.npu_stream))
            h = torch.zeros(4096,dtype=torch.uint8,pin_memory=True)
            check(lib.plan_metadata(plan,h.data_ptr(),h.numel()))
            gm.copy_(h,non_blocking=True)
            ptrs = (u64*9)(*[x.data_ptr() for x in (q,mask,ql,kl,table,out,scratch,k,v)])
            check(lib.plan_bind(plan,ptrs));check(lib.plan_bind_metadata(plan,gm.data_ptr()))
            host_tag = torch.zeros(1,dtype=torch.int64,pin_memory=True)
            tag = torch.zeros(1,dtype=torch.int64,device='npu')
            receipt = torch.empty(rows+2,dtype=torch.int64,device='npu')
            host_receipt = torch.empty_like(receipt,device='cpu',pin_memory=True)
            host_out = torch.empty_like(out,device='cpu',pin_memory=True)
            graph = torch.npu.NPUGraph()
            torch.npu.synchronize()
            with torch.npu.graph(graph):
                lengths.add_(((counter+axis)%3+1)*live)
                kl.copy_(lengths)
                receipt[:1].copy_(tag)
                receipt[1:2].copy_(counter)
                receipt[2:].copy_(kl)
                check(lib.plan_launch(plan,torch.npu.current_stream().npu_stream))
                counter.add_(1)
            banks.append(dict(q=q,out=out,ql=ql,kl=kl,gm=gm,scratch=scratch,plan=plan,
                graph=graph,host_tag=host_tag,tag=tag,receipt=receipt,
                host_receipt=host_receipt,host_out=host_out,used=False,
                uploaded=torch.npu.Event(),consumed=torch.npu.Event(),copied=torch.npu.Event()))
        lengths.copy_(torch.tensor(seed,device='npu',dtype=torch.int64));counter.zero_()
        torch.npu.synchronize()
        pending, observed = deque(), []
        # Two outstanding waves. Receipts only govern host buffer reuse/checks,
        # never author the next wave's accepted count or sequence length.
        def collect():
            sequence, b = pending.popleft()
            b['copied'].synchronize()
            observed.append((sequence,b['host_receipt'].clone(),b['host_out'].clone()))
        for sequence in range(24):
            if len(pending) == 2:
                collect()
            b = banks[sequence%2]
            if b['used']:
                b['uploaded'].synchronize()  # pinned source reuse, not target completion
            b['host_tag'][0] = sequence
            with torch.npu.stream(ingress):
                if b['used']:
                    ingress.wait_event(b['consumed'])
                b['tag'].copy_(b['host_tag'],non_blocking=True)
                b['uploaded'].record(ingress)
            compute.wait_event(b['uploaded'])
            if b['used']:
                compute.wait_event(b['copied'])  # output bank still owned by D2H
            b['graph'].replay()
            b['consumed'].record(compute)
            with torch.npu.stream(egress):
                egress.wait_event(b['consumed'])
                b['host_receipt'].copy_(b['receipt'],non_blocking=True)
                b['host_out'].copy_(b['out'],non_blocking=True)
                b['copied'].record(egress)
            b['used'] = True
            pending.append((sequence,b))
        while pending:
            collect()
        expected = seed.copy()
        for sequence, receipt, output in observed:
            expected = [n+((sequence+i)%3+1)*flag for i,(n,flag) in enumerate(zip(expected,active))]
            assert receipt.tolist() == [sequence,sequence,*expected], (sequence,receipt,expected)
            ref_table, ref_offsets, ref_lengths = table, offsets, expected
            if draft_padding:
                # Compare compressed zero-KV tail against donor's original
                # one-query-per-token-capacity representation, not itself.
                ref_table = torch.zeros(width,columns,device='npu',dtype=torch.int32)
                ref_table[:8].copy_(table[:8])
                ref_offsets = list(range(1,width+1))
                ref_lengths = expected[:8]+[0]*(width-8)
            ref = torch_npu.npu_fused_infer_attention_score(banks[sequence%2]['q'],k,v,
                atten_mask=mask,block_table=ref_table,actual_seq_lengths=ref_offsets,
                actual_seq_lengths_kv=ref_lengths,num_heads=12,num_key_value_heads=2,
                scale=256**-.5,block_size=128,input_layout='TND',sparse_mode=3,next_tokens=0)[0]
            ref = ref[:valid].cpu()
            torch.testing.assert_close(output[:valid],ref,rtol=.01,atol=.002)
            results.append(dict(kind=kind,capacity=width,sequence=sequence,lengths=expected,
                max_error=(output[:valid]-ref).abs().max().item()))
        torch.npu.synchronize()
        for b in banks:
            check(lib.plan_release(b['plan']))
    Path(os.environ['CAPSULE'],'complete.json').write_text(json.dumps(dict(status='PASS',
        scope='Synthetic device feedback + real FIA, two outstanding waves, not model/service qualification',
        checks=results),indent=2))
    print('PASS',len(results),'waves',flush=True)


if __name__ == '__main__':
    main()
