"""Exact-shape DSpark runtime-capture experiment, not general graph admission.

One observed pure-verification shape per request count is captured (default1–4;
explicit bounded producer composition can request up to16).
Other shapes fall back; the bank cache cannot grow without bound.
Draft metadata has a PRIVATE stable bank, never the target global RoPE bank.
"""
import copy
import dataclasses
import json
import os
from enum import Enum
from pathlib import Path
import torch
from vllm.forward_context import get_forward_context
from vllm_ascend.attention.context_parallel.dsa_cp import RopeDataProxy


def signature(value, memo=None):
    # kwargs and context share metadata nodes. Visit that DAG once, while
    # retaining the original structural signature/admission contract.
    if memo is False:return _signature(value,False)  # same-graph traversal control
    if memo is None:memo={}
    if id(value) not in memo:memo[id(value)]=_signature(value,memo)
    return memo[id(value)]


def _signature(value,memo):
    if isinstance(value, torch.Tensor):
        return ('tensor',str(value.device),str(value.dtype),tuple(value.shape),
                tuple(value.flatten().tolist()) if value.device.type=='cpu' else None)
    if isinstance(value, RopeDataProxy):return ('rope',value.idx,signature(value._data,memo))
    if dataclasses.is_dataclass(value):
        return (type(value).__name__,tuple((f.name,signature(getattr(value,f.name),memo)) for f in dataclasses.fields(value)))
    if isinstance(value,dict):return tuple((k,signature(v,memo)) for k,v in value.items())
    if isinstance(value,(tuple,list)):return (type(value).__name__,tuple(signature(v,memo) for v in value))
    if isinstance(value,Enum):return (type(value).__name__,value.name)
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError(f'Unreviewed graph metadata type: {type(value)}')


def persistent_layout(value):
    if isinstance(value,torch.Tensor):
        return (value.data_ptr(),tuple(value.shape),tuple(value.stride()),str(value.dtype))
    if isinstance(value,(tuple,list)):return tuple(map(persistent_layout,value))
    if value is None:return None
    raise TypeError(f'Unreviewed persistent draft input: {type(value)}')


def key_changes(old,new,path=()):
    if old==new:return []
    if isinstance(old,tuple) and isinstance(new,tuple) and len(old)==len(new):
        result=[]
        for i,(a,b) in enumerate(zip(old,new)):
            result.extend(key_changes(a,b,path+(i,)))
            if len(result)>=32:break
        return result[:32]
    return [dict(path=path,old=repr(old)[:500],new=repr(new)[:500])]


def bank(value,memo=None):
    if memo is None:memo={}
    if id(value) not in memo:memo[id(value)]=_bank(value,memo)
    return memo[id(value)]


def _bank(value,memo):
    if isinstance(value,torch.Tensor):return value.clone()
    if isinstance(value,RopeDataProxy):
        result=copy.copy(value);result._data=bank(value._data,memo);return result
    if dataclasses.is_dataclass(value):
        result=copy.copy(value)
        for f in dataclasses.fields(value):setattr(result,f.name,bank(getattr(value,f.name),memo))
        return result
    if isinstance(value,dict):return {k:bank(v,memo) for k,v in value.items()}
    if isinstance(value,list):return [bank(v,memo) for v in value]
    if isinstance(value,tuple):return tuple(bank(v,memo) for v in value)
    return value


def refresh(dst,src,seen=None,bindings=None):
    if seen is None:seen=set()
    if bindings is None:bindings={}
    # A destination aliased at capture must not silently take the last of two
    # distinct runtime sources. Signature equality alone cannot detect that.
    if isinstance(src,torch.Tensor):
        layout=(str(src.device),src.data_ptr(),tuple(src.shape),tuple(src.stride()),src.dtype)
        previous=bindings.setdefault(id(dst),layout)
        assert previous==layout, 'Draft metadata source alias split'
    pair=(id(dst),id(src))
    if seen is not False:
        if pair in seen:return
        seen.add(pair)
    if isinstance(src,torch.Tensor):dst.copy_(src,non_blocking=True)
    elif isinstance(src,RopeDataProxy):refresh(dst._data,src._data,seen,bindings)
    elif dataclasses.is_dataclass(src):
        for f in dataclasses.fields(src):refresh(getattr(dst,f.name),getattr(src,f.name),seen,bindings)
    elif isinstance(src,dict):
        for k,v in src.items():refresh(dst[k],v,seen,bindings)
    elif isinstance(src,(list,tuple)):
        for d,s in zip(dst,src):refresh(d,s,seen,bindings)


class ExactDraftGraph:
    # Bounded same-engine study switch: both controls share graph banks/layout;
    # false reproduces the old repeated signature/refresh walk and copy count.
    metadata_dag = True
    def __init__(self,worker,original=None,request_count=4,context_capacity=None,reference=None,pool=None):
        self.worker=worker;self.drafter=worker.model_runner.drafter
        assert type(self.drafter).__name__=='AscendDSparkProposer'
        assert not self.drafter.use_cuda_graph
        self.original=original or self.drafter._runnable
        self.request_count=request_count
        self.context_capacity=context_capacity or 6*request_count
        self.reference=reference or self.original
        self.pool=pool
        self.enabled=True;self.key=None;self.graph=None;self.replays=0;self.fallbacks=0;self.checks=0;self.capture_checked=False;self.signed_zero_words=0
        self.path=Path(os.environ['FULL_MIXED_OUTPUT'])/f'draft-graph-rank{worker.rank}-requests{request_count}.json'

    def receipt(self):
        self.path.write_text(json.dumps(dict(captured=self.graph is not None,replays=self.replays,fallbacks=self.fallbacks,checks=self.checks,capture_checked=self.capture_checked,signed_zero_words=self.signed_zero_words,reference=getattr(self,'reference_kind','original'),key=repr(self.key)),indent=2))

    def __call__(self,**kwargs):
        if not self.enabled:return self.original(**kwargs)
        ctx=get_forward_context();d=self.drafter
        query_only = getattr(self, 'query_only', False)
        eligible=(kwargs['batch_size']==self.request_count and (query_only or d._dflash_num_context==self.context_capacity)
                  and not kwargs.get('is_prefill',False) and not ctx.capturing)
        if not eligible:
            self.fallbacks+=1;return self.original(**kwargs)
        current=(kwargs,ctx.attn_metadata)
        names = ('input_ids','positions','_dspark_seed_buffer','_dspark_draft_buffer')
        if not query_only:
            names += ('_dflash_hidden_states','_context_positions_buffer','_context_slot_mapping_buffers')
        state_inputs=tuple(persistent_layout(getattr(d,name)) for name in names)
        key=(None if query_only else d._dflash_num_context,signature(current,None if self.metadata_dag else False),state_inputs)
        if self.key is not None and key!=self.key:
            if getattr(self,'strict_signature',False):
                self.path.with_suffix('.signature.json').write_text(json.dumps(key_changes(self.key,key),indent=2))
                raise AssertionError('FULL draft bank signature changed; see signature receipt')
            self.fallbacks+=1;return self.original(**kwargs)
        if self.key is None:
            self.key=key;self.buffers=bank(current)
            def capture():
                old=ctx.attn_metadata;ctx.attn_metadata=self.buffers[1]
                graph=torch.npu.NPUGraph()
                torch.npu.synchronize()
                was_capturing=ctx.capturing;ctx.capturing=True
                try:
                    with torch.npu.graph(graph,pool=self.pool):self.output=self.original(**self.buffers[0])
                finally:
                    ctx.attn_metadata=old;ctx.capturing=was_capturing
                self.graph=graph
                # Runtime capture is initialization, not a committed serving
                # invocation. Execute explicitly before exposing output/KV state.
                graph.replay()
                return self.output
            if os.environ.get('DRAFT_GRAPH_SHADOW')=='1':
                self.checked_call(capture,kwargs)
                self.capture_checked=True
            else:
                capture()
            self.receipt();return self.output
        refresh(self.buffers,current,None if self.metadata_dag else False)
        if os.environ.get('DRAFT_GRAPH_SHADOW')=='1' and self.checks<8:
            self.checked_call(lambda:(self.graph.replay(),self.output)[1],kwargs)
            self.checks+=1
        else:
            self.graph.replay()
        self.replays+=1
        if self.replays==1 or self.checks:self.receipt()
        return self.output

    def checked_call(self,run,kwargs):
        # The byte-pool helper has no runner patch-installation side effects.
        from dp_full_shadow import byte_pools
        pools=byte_pools(self.worker.model_runner.kv_caches)
        torch.npu.synchronize()
        before=[x.clone() for x in pools]
        observed=run().clone()
        after=[x.clone() for x in pools]
        torch.npu.synchronize()
        for live,saved in zip(pools,before):live.copy_(saved)
        expected=self.reference(**kwargs)
        torch.npu.synchronize()
        try:
            if not torch.equal(observed,expected) and getattr(self,'allow_addressed_signed_zero',False):
                from draft_oracle import diagnose_output
                detail=diagnose_output(self,kwargs,observed,expected.clone(),pools,before,after)
                self.path.with_suffix('.output-failure.json').write_text(json.dumps(detail,indent=2))
                raise AssertionError('Draft output differs; diagnostic re-executions saved')
            torch.testing.assert_close(observed,expected,rtol=0,atol=0)
            zero_rows={}
            if getattr(self,'allow_addressed_signed_zero',False):
                from draft_oracle import addressed_rows, equivalent
                zero_rows=addressed_rows(self.drafter,self.request_count,self.context_capacity)
            for index,(live,saved) in enumerate(zip(pools,after)):
                if torch.equal(live,saved):continue
                if zero_rows:
                    zero_count=equivalent(live,saved,zero_rows.get(live.untyped_storage().data_ptr(),set()))
                    if zero_count is not None:
                        self.signed_zero_words+=zero_count
                        continue
                # Bound diagnostic allocation instead of materializing a mask
                # for an entire multi-GiB pool. Preserve the first differing slab.
                for start in range(0,live.numel(),1024*1024):
                    lhs=live[start:start+1024*1024]
                    rhs=saved[start:start+1024*1024]
                    if torch.equal(lhs,rhs):continue
                    a,b=lhs.cpu(),rhs.cpu()
                    offset=int(torch.nonzero(a!=b)[0])
                    begin=max(0,(offset//2)*2-16)
                    detail=dict(pool=index,byte_offset=start+offset,
                                requests=self.request_count,context_capacity=self.context_capacity,
                                eager=a[begin:begin+64].tolist(),graph=b[begin:begin+64].tolist())
                    for dst,src in zip(pools,before):dst.copy_(src)
                    self.original(**kwargs)
                    torch.npu.synchronize()
                    detail['padded_eager_matches_graph']=all(torch.equal(dst,src) for dst,src in zip(pools,after))
                    self.path.with_suffix('.failure.json').write_text(json.dumps(detail,indent=2))
                    break
                raise AssertionError('Draft graph/eager KV pool bytes differ')
        finally:
            for live,saved in zip(pools,after):live.copy_(saved)
            self.output.copy_(observed)


class DraftGraphSet:
    """One exact shape per native request count; never an unbounded shape cache."""
    def __init__(self,worker,max_requests=4,pool=None):
        self.worker=worker;self.drafter=worker.model_runner.drafter
        self.original=self.drafter._runnable
        assert self.drafter.num_speculative_tokens==5, 'Only K5 is qualified here'
        assert 1<=max_requests<=16
        assert worker.model_runner.vllm_config.scheduler_config.max_num_seqs==max_requests
        self.max_requests=max_requests
        self.pool=pool if pool is not None else torch.npu.graph_pool_handle()
        self.enabled=True;self.entries={};self.fallbacks=0

    def __call__(self,**kwargs):
        if not self.enabled:return self.original(**kwargs)
        count=kwargs['batch_size']
        if (not 1<=count<=self.max_requests or self.drafter._dflash_num_context!=6*count
                or kwargs.get('is_prefill',False) or get_forward_context().capturing):
            self.fallbacks+=1;return self.original(**kwargs)
        if count not in self.entries:
            self.entries[count]=ExactDraftGraph(self.worker,self.original,count,pool=self.pool)
        return self.entries[count](**kwargs)

    def receipt(self):
        entries=[]
        for count,g in sorted(self.entries.items()):
            g.receipt()
            entries.append(dict(requests=count,captured=g.graph is not None,replays=g.replays,
                                fallbacks=g.fallbacks,checks=g.checks,capture_checked=g.capture_checked))
        path=Path(os.environ['FULL_MIXED_OUTPUT'])/f'draft-graph-rank{self.worker.rank}.json'
        path.write_text(json.dumps(dict(enabled=self.enabled,entries=entries,
            replays=sum(g.replays for g in self.entries.values()),
            fallbacks=self.fallbacks+sum(g.fallbacks for g in self.entries.values())),indent=2))


def install(worker, enabled=True):
    torch.npu.synchronize()
    if hasattr(worker,"_exact_draft_graph"):
        worker._exact_draft_graph.enabled=enabled
        worker._exact_draft_graph.receipt()
        return dict(rank=worker.rank,draft_graph=enabled)
    if not enabled:return dict(rank=worker.rank,draft_graph=False)
    experiment=DraftGraphSet(worker)
    worker.model_runner.drafter._runnable=experiment
    worker._exact_draft_graph=experiment
    return dict(rank=worker.rank,policy='bounded-K5-one-exact-shape-per-request-count')
