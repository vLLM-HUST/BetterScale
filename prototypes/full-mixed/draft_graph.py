"""Exact-shape DSpark runtime-capture experiment, not general graph admission.

Only one observed pure-verification shape is captured. New shapes fall back.
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


def signature(value):
    if isinstance(value, torch.Tensor):
        return ('tensor',str(value.device),str(value.dtype),tuple(value.shape),
                tuple(value.flatten().tolist()) if value.device.type=='cpu' else None)
    if isinstance(value, RopeDataProxy):return ('rope',value.idx,signature(value._data))
    if dataclasses.is_dataclass(value):
        return (type(value).__name__,tuple((f.name,signature(getattr(value,f.name))) for f in dataclasses.fields(value)))
    if isinstance(value,dict):return tuple((k,signature(v)) for k,v in value.items())
    if isinstance(value,(tuple,list)):return (type(value).__name__,tuple(map(signature,value)))
    if isinstance(value,Enum):return (type(value).__name__,value.name)
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError(f'Unreviewed graph metadata type: {type(value)}')


def persistent_layout(value):
    if isinstance(value,torch.Tensor):
        return (value.data_ptr(),tuple(value.shape),tuple(value.stride()),str(value.dtype))
    if isinstance(value,(tuple,list)):return tuple(map(persistent_layout,value))
    if value is None:return None
    raise TypeError(f'Unreviewed persistent draft input: {type(value)}')


def bank(value):
    if isinstance(value,torch.Tensor):return value.clone()
    if isinstance(value,RopeDataProxy):
        result=copy.copy(value);result._data=bank(value._data);return result
    if dataclasses.is_dataclass(value):
        result=copy.copy(value)
        for f in dataclasses.fields(value):setattr(result,f.name,bank(getattr(value,f.name)))
        return result
    if isinstance(value,dict):return {k:bank(v) for k,v in value.items()}
    if isinstance(value,list):return [bank(v) for v in value]
    if isinstance(value,tuple):return tuple(bank(v) for v in value)
    return value


def refresh(dst,src):
    if isinstance(src,torch.Tensor):dst.copy_(src,non_blocking=True)
    elif isinstance(src,RopeDataProxy):refresh(dst._data,src._data)
    elif dataclasses.is_dataclass(src):
        for f in dataclasses.fields(src):refresh(getattr(dst,f.name),getattr(src,f.name))
    elif isinstance(src,dict):
        for k,v in src.items():refresh(dst[k],v)
    elif isinstance(src,(list,tuple)):
        for d,s in zip(dst,src):refresh(d,s)


class ExactDraftGraph:
    def __init__(self,worker):
        self.worker=worker;self.drafter=worker.model_runner.drafter
        assert type(self.drafter).__name__=='AscendDSparkProposer'
        assert not self.drafter.use_cuda_graph
        self.original=self.drafter._runnable
        self.enabled=True;self.key=None;self.graph=None;self.replays=0;self.fallbacks=0;self.checks=0
        self.path=Path(os.environ['FULL_MIXED_OUTPUT'])/f'draft-graph-rank{worker.rank}.json'

    def receipt(self):
        self.path.write_text(json.dumps(dict(captured=self.graph is not None,replays=self.replays,fallbacks=self.fallbacks,checks=self.checks,key=repr(self.key)),indent=2))

    def __call__(self,**kwargs):
        if not self.enabled:return self.original(**kwargs)
        ctx=get_forward_context();d=self.drafter
        eligible=(kwargs['batch_size']==4 and d._dflash_num_context<=24 and not ctx.capturing)
        if not eligible:
            self.fallbacks+=1;return self.original(**kwargs)
        current=(kwargs,ctx.attn_metadata)
        state_inputs=tuple(persistent_layout(getattr(d,name)) for name in (
            'input_ids','positions','_dflash_hidden_states','_context_positions_buffer',
            '_context_slot_mapping_buffers','_dspark_seed_buffer','_dspark_draft_buffer'))
        key=(d._dflash_num_context,signature(current),state_inputs)
        if self.key is not None and key!=self.key:
            self.fallbacks+=1;return self.original(**kwargs)
        if self.key is None:
            self.key=key;self.buffers=bank(current)
            old=ctx.attn_metadata;ctx.attn_metadata=self.buffers[1]
            graph=torch.npu.NPUGraph()
            torch.npu.synchronize()
            was_capturing=ctx.capturing;ctx.capturing=True
            try:
                with torch.npu.graph(graph):self.output=self.original(**self.buffers[0])
            finally:
                ctx.attn_metadata=old;ctx.capturing=was_capturing
            self.graph=graph;self.receipt();return self.output
        refresh(self.buffers,current)
        if os.environ.get('DRAFT_GRAPH_SHADOW')=='1' and self.checks<8:
            from extension import _unique_byte_pools
            from torch.utils._pytree import tree_flatten
            leaves,_=tree_flatten(self.worker.model_runner.kv_caches)
            pools=_unique_byte_pools([x for x in leaves if isinstance(x,torch.Tensor)])
            torch.npu.synchronize()
            before=[x.clone() for x in pools]
            self.graph.replay()
            observed=self.output.clone()
            after=[x.clone() for x in pools]
            torch.npu.synchronize()
            for live,saved in zip(pools,before):live.copy_(saved)
            expected=self.original(**kwargs)
            torch.npu.synchronize()
            try:
                torch.testing.assert_close(observed,expected,rtol=0,atol=0)
                for live,saved in zip(pools,after):
                    assert torch.equal(live,saved), 'Draft graph/eager KV pool bytes differ'
                self.checks+=1
            finally:
                for live,saved in zip(pools,after):live.copy_(saved)
                self.output.copy_(observed)
        else:
            self.graph.replay()
        self.replays+=1
        if self.replays==1 or self.checks: self.receipt()
        return self.output


def install(worker, enabled=True):
    torch.npu.synchronize()
    if hasattr(worker,"_exact_draft_graph"):
        worker._exact_draft_graph.enabled=enabled
        worker._exact_draft_graph.receipt()
        return dict(rank=worker.rank,draft_graph=enabled)
    if not enabled:return dict(rank=worker.rank,draft_graph=False)
    experiment=ExactDraftGraph(worker)
    worker.model_runner.drafter._runnable=experiment
    worker._exact_draft_graph=experiment
    return dict(rank=worker.rank,policy='one-exact-shape-runtime-capture')
