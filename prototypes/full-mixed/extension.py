"""Opt-in exploration only: do not expose this worker extension in production."""
from vllm.v1.attention.backend import AttentionCGSupport
from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPMetadataBuilder

# Experimental admission to reveal real capture failures. This is deliberately
# confined to a worker_extension_cls selected by the dummy probe.
_original_support = AscendDSACPMetadataBuilder.__dict__['get_cudagraph_support']
AscendDSACPMetadataBuilder.get_cudagraph_support = classmethod(
    lambda cls, vllm_config, kv_cache_spec: AttentionCGSupport.ALWAYS)

class FullMixedProbeWorker:
    def enable_n2(self):
        from full_draft import install as install_draft
        from cross_step import install as install_bounds
        return dict(draft=install_draft(self), bounds=install_bounds(self, all_modes=True))

    def draft_bank_status(self):
        state = getattr(self, "_exact_draft_graph", None)
        return dict(rank=self.rank,banks={} if state is None else
                    {str(n):g.graph is not None for n,g in state.entries.items()})

    def set_cross_step_bounds(self, enabled=True):
        from cross_step import install
        return install(self, enabled)

    def set_cpu_qli(self, enabled=False, verify=False):
        from qli_cpu import configure
        return configure(self, enabled, verify)

    def enable_exact_draft_graph(self, enabled=True):
        from draft_graph import install
        return install(self, enabled)

    def set_ordered_replay(self, enabled=False):
        from ordered_replay import configure
        return configure(self, enabled)

    def start_decode_observation(self, profile=False, label="decode"):
        from diagnostics import start_decode_observation
        return start_decode_observation(self, profile, label)

    def stop_decode_observation(self):
        from diagnostics import stop_decode_observation
        return stop_decode_observation(self)

    def graph_receipt(self):
        if hasattr(self,"_exact_draft_graph"):self._exact_draft_graph.receipt()
        import torch
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
        if hasattr(self.model_runner, "_cross_step_bounds"):
            self.model_runner._cross_step_bounds.receipt()
        model=self.model_runner.model
        wrappers=[]
        if isinstance(model,ACLGraphWrapper):
            wrappers.append(dict(mode=str(model.runtime_mode),entries=[dict(descriptor=str(k),tokens=k.num_tokens,requests=k.num_reqs,captured=v.aclgraph is not None,replays=model.__dict__.get('_probe_replays',{}).get(str(k),0)) for k,v in model.concrete_aclgraph_entries.items()]))
        return dict(rank=self.rank,wrappers=wrappers,allocated=torch.npu.memory_allocated(),reserved=torch.npu.memory_reserved(),peak=torch.npu.max_memory_allocated())

# Dummy loading bypasses the specialized checkpoint loader that reshapes wo_a.
# Reproduce that loader's layout conversion, not a changed attention algorithm.
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
_original_load_model = NPUModelRunner.load_model

def _load_dummy_layout(self, *args, **kwargs):
    result = _original_load_model(self, *args, **kwargs)
    if self.vllm_config.load_config.load_format == 'dummy':
        models=[self.get_model(),getattr(getattr(self,'drafter',None),'model',None)]
        for model in models:
            if model is None:continue
            for name, module in model.named_modules():
                if name.endswith('wo_a') and module.weight.ndim == 2:
                    module.weight.data = module.weight.data.view(module.n_local_groups, module.o_lora_rank, -1).transpose(2, 1).contiguous()
    return result

NPUModelRunner.load_model = _load_dummy_layout

# Prefill used transient RoPE tensors; captured consumers need the same
# persistent buffers already used by decode. Scope this to target build,
# leaving the explicit drafting builder's private buffers untouched.
from contextvars import ContextVar
from vllm_ascend.attention.context_parallel import dsa_cp
_target_build = ContextVar('full_mixed_target_build', default=False)
_original_build = AscendDSACPMetadataBuilder.build
_original_rope = dsa_cp.get_cos_and_sin_dsa

def _stable_rope(*args, **kwargs):
    if _target_build.get():
        kwargs['use_cache'] = True
    return _original_rope(*args, **kwargs)

def _build_target(self, *args, **kwargs):
    token = _target_build.set(True)
    try:
        return _original_build(self, *args, **kwargs)
    finally:
        _target_build.reset(token)

dsa_cp.get_cos_and_sin_dsa = _stable_rope
AscendDSACPMetadataBuilder.build = _build_target

# Probe-only shadow oracle: replay and eager start from exactly the same KV
# state and metadata. Restore replay's state before returning to the engine.
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm.forward_context import get_forward_context
from vllm.config import CUDAGraphMode
from torch.utils._pytree import tree_flatten, tree_map
import torch
import json
import os
from pathlib import Path
_original_graph_call = ACLGraphWrapper.__call__


def _metadata_addresses(value, path='metadata'):
    import dataclasses
    result = {}
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu':
            result[path] = dict(ptr=value.data_ptr(), shape=list(value.shape), dtype=str(value.dtype))
    elif dataclasses.is_dataclass(value):
        for f in dataclasses.fields(value): result.update(_metadata_addresses(getattr(value,f.name),path+'.'+f.name))
    elif isinstance(value, dict):
        for k,v in value.items(): result.update(_metadata_addresses(v,path+'.'+str(k)))
    elif isinstance(value, (tuple,list)):
        for i,v in enumerate(value): result.update(_metadata_addresses(v,path+'.'+str(i)))
    elif isinstance(value, dsa_cp.RopeDataProxy):
        result.update(_metadata_addresses(value._data,path+'.rope'))
    return result

def _unique_byte_pools(views):
    # The donor gives heterogeneous cache groups typed views over shared pools.
    # Snapshot each physical storage once; never interpret someone else's FP32
    # state pages through a BF16 cache view. Byte equality is the strict oracle.
    pools={}
    for view in views:
        storage=view.untyped_storage()
        key=(str(view.device),storage.data_ptr(),storage.nbytes())
        if key not in pools:
            raw=torch.as_strided(view,(storage.nbytes()//view.element_size(),),(1,),storage_offset=0)
            pools[key]=raw.view(torch.uint8)
    return list(pools.values())

def _compare_bounded(x, y, *, equal_nan=False, measure_abs=False):
    # Shadow must not materialize multiple whole-cache FP32 copies. This is an
    # oracle-memory bound, not a change to graph or cache behavior.
    assert x.shape == y.shape and x.dtype == y.dtype
    import math
    row_elements = math.prod(x.shape[1:])
    rows = max(1, (16 * 1024 * 1024) // max(1, row_elements))
    max_diff, max_abs, unequal = 0.0, 0.0, 0
    for start in range(0, x.shape[0], rows):
        a, b = x[start:start+rows], y[start:start+rows]
        if x.dtype == torch.uint8:
            if not torch.equal(a,b):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            continue
        torch.testing.assert_close(a, b, rtol=.01, atol=1e-6, equal_nan=equal_nan)
        max_diff = max(max_diff, float(torch.nan_to_num((a.float()-b.float()).abs()).max()))
        unequal += int((a != b).sum())
        if measure_abs: max_abs = max(max_abs, float(a.float().abs().max()))
    return dict(max_diff=max_diff, unequal=unequal, max_abs=max_abs if measure_abs else None)

def _checked_graph_call(self, *args, **kwargs):
    runner = self.__dict__.get('_probe_runner')
    ctx = get_forward_context()
    entry = self.concrete_aclgraph_entries.get(ctx.batch_descriptor)
    if ctx.cudagraph_runtime_mode == CUDAGraphMode.FULL and entry is not None and entry.aclgraph is not None:
        counts=self.__dict__.setdefault('_probe_replays',{})
        key=str(ctx.batch_descriptor); counts[key]=counts.get(key,0)+1
    if ctx.cudagraph_runtime_mode == CUDAGraphMode.FULL and (entry is None or entry.aclgraph is None):
        result = _original_graph_call(self, *args, **kwargs)
        self.__dict__.setdefault('_probe_capture_meta', {})[ctx.batch_descriptor] = _metadata_addresses(ctx.attn_metadata)
        return result
    if runner is not None and getattr(runner, '_probe_validate', False):
        old = self.__dict__.get('_probe_capture_meta', {}).get(ctx.batch_descriptor, {})
        new = _metadata_addresses(ctx.attn_metadata)
        differences = {k:dict(capture=old.get(k),current=v) for k,v in new.items() if old.get(k)!=v}
        Path(os.environ['FULL_MIXED_OUTPUT'], f'pointer-diff-rank{runner._probe_rank}.json').write_text(json.dumps(differences,indent=2))
    if (runner is None or not getattr(runner, '_probe_validate', False)
            or ctx.cudagraph_runtime_mode != CUDAGraphMode.FULL
            or entry is None or entry.aclgraph is None
            or len(runner._probe_shadows) >= 96):
        return _original_graph_call(self, *args, **kwargs)
    torch.npu.synchronize()  # snapshot state only after all producer streams finish
    meta = next(iter(ctx.attn_metadata.values()))
    start_qsl = meta.req_metadata.query_start_loc.cpu().tolist()
    state_layout=[]
    for name,module in runner.compilation_config.static_forward_context.items():
        leaves,_=tree_flatten(getattr(module,'kv_cache',[]))
        for t in leaves:
            if isinstance(t,torch.Tensor):
                state_layout.append(dict(name=name,ptr=t.data_ptr(),shape=list(t.shape),stride=list(t.stride()),format=torch.npu.get_npu_format(t) if hasattr(torch.npu,'get_npu_format') else None))
    Path(os.environ['FULL_MIXED_OUTPUT'],f'state-layout-rank{runner._probe_rank}.json').write_text(json.dumps(state_layout))
    start_meta = dict(descriptor=str(ctx.batch_descriptor),query_offsets=start_qsl,
                     prefills=meta.num_prefills,decodes=meta.num_decodes,slot_mapping=meta.req_metadata.slot_mapping.cpu().tolist() if meta.req_metadata.slot_mapping is not None else None)
    Path(os.environ['FULL_MIXED_OUTPUT'],f'shadow-start-rank{runner._probe_rank}.json').write_text(json.dumps(start_meta))
    tensors, _ = tree_flatten(runner.kv_caches)
    views = [x for x in tensors if isinstance(x, torch.Tensor)]
    caches = _unique_byte_pools(views)
    start_meta['cache_names']=[[v['name'] for v in state_layout if v['ptr']==x.data_ptr()] for x in views]
    start_meta['pool_bytes']=[x.numel() for x in caches]
    start_meta['swa_slots']={name:dict(block_size=m.req_metadata.block_size,slots=m.req_metadata.slot_mapping.cpu().tolist()) for name,m in ctx.attn_metadata.items() if 'swa_cache' in name and m.req_metadata.slot_mapping is not None}
    # DSpark also consumes the persistent pre-HC residual, not just returned
    # hidden states. Keep its valid prefix in the same shadow contract.
    side = [m._mtp_hidden_buffer for m in runner.get_model().modules()
            if hasattr(m, '_mtp_hidden_buffer')]
    side_before = [x.clone() for x in side]
    before = [x.clone() for x in caches]
    if os.environ.get('FULL_MIXED_ORACLE') == 'eager':
        saved_mode=ctx.cudagraph_runtime_mode
        try:
            ctx.cudagraph_runtime_mode=CUDAGraphMode.NONE
            replay=self.runnable(*args, **kwargs)
        finally:ctx.cudagraph_runtime_mode=saved_mode
    else:
        replay = _original_graph_call(self, *args, **kwargs)
    torch.npu.synchronize()  # attribute a graph failure before starting eager
    clone = lambda x: x.clone() if isinstance(x, torch.Tensor) else x
    actual = tree_map(clone, replay)
    after = [x.clone() for x in caches]
    side_after = [x.clone() for x in side]
    for target, source in zip(side, side_before): target.copy_(source)
    for target, source in zip(caches, before): target.copy_(source)
    old_mode = ctx.cudagraph_runtime_mode
    cross_step = getattr(runner, "_cross_step_bounds", None)
    reference_upper = None
    try:
        if cross_step is not None:
            reference_upper = cross_step.reference_begin(ctx)
        if os.environ.get('FULL_MIXED_ORACLE') == 'native_graph':
            # Compare our replay policy with the unchanged donor graph, including
            # all backing writes. This is NOT a graph/eager equivalence claim.
            expected = _native_graph_call(self, *args, **kwargs)
        else:
            ctx.cudagraph_runtime_mode = CUDAGraphMode.NONE
            expected = self.runnable(*args, **kwargs)
        torch.npu.synchronize()
        checks = []
        metadata = next(iter(ctx.attn_metadata.values()))
        valid_global = int(metadata.req_metadata.query_start_loc[-1])
        capacity = ctx.batch_descriptor.num_tokens
        from vllm.distributed import get_tp_group
        tp = get_tp_group()
        for label, left, right in [('output', actual, expected), ('kv', after, caches), ('mtp', side_after, side)]:
            ls, _ = tree_flatten(left); rs, _ = tree_flatten(right)
            assert len(ls) == len(rs)
            for i, (x, y) in enumerate(zip(ls, rs)):
                if not isinstance(x, torch.Tensor): continue
                if label == 'mtp':
                    x,y = x[:valid_global],y[:valid_global]
                    assert torch.isfinite(x).all() and torch.isfinite(y).all()
                if label == 'output':
                    if x.shape[0] == capacity:
                        valid_rows = valid_global
                    elif x.shape[0] * tp.world_size == capacity:
                        valid_rows = max(0,min(x.shape[0],valid_global-tp.rank_in_group*x.shape[0]))
                    else:
                        raise AssertionError(f'unknown output token layout {x.shape}, capacity={capacity}')
                    x,y = x[:valid_rows],y[:valid_rows]
                    assert torch.isfinite(x).all() and torch.isfinite(y).all(), 'nonfinite valid dummy output'
                try:
                    measurement = _compare_bounded(x,y,equal_nan=label=='kv',measure_abs=label=='output')
                except AssertionError as error:
                    samples=[]
                    for row in range(min(x.shape[0],128)):
                        left,right=x[row].reshape(-1),y[row].reshape(-1)
                        bad=(~torch.isclose(left,right,rtol=.01,atol=1e-6,equal_nan=label=='kv')).nonzero().flatten()[:8]
                        if bad.numel():
                            samples.append(dict(row=row,offsets=bad.cpu().tolist(),replay=left[bad].float().cpu().tolist(),eager=right[bad].float().cpu().tolist()))
                        if len(samples)>=4:break
                    nonfinite=[]
                    for row in range(min(x.shape[0],128)):
                        left,right=x[row].reshape(-1),y[row].reshape(-1)
                        bad=(torch.isfinite(left)!=torch.isfinite(right)).nonzero().flatten()[:8]
                        if bad.numel():
                            nonfinite.append(dict(row=row,offsets=bad.cpu().tolist(),replay=left[bad].float().cpu().tolist(),eager=right[bad].float().cpu().tolist()))
                            break
                    failure=dict(error=str(error),nonfinite=nonfinite,start=start_meta,end_query_offsets=meta.req_metadata.query_start_loc.cpu().tolist(),label=label,index=i,shape=list(x.shape),dtype=str(x.dtype),samples=samples)
                    failure_path=Path(os.environ['FULL_MIXED_OUTPUT'],f'shadow-failure-rank{runner._probe_rank}.json')
                    if not failure_path.exists():failure_path.write_text(json.dumps(failure,indent=2))
                    raise
                checks.append(dict(label=label,index=i,shape=list(x.shape),dtype=str(x.dtype),**measurement))
                Path(os.environ['FULL_MIXED_OUTPUT'],f'checking-rank{runner._probe_rank}.json').write_text(json.dumps(dict(valid_global=valid_global,capacity=capacity,checks=checks),indent=2))
        runner._probe_shadows.append(dict(oracle=os.environ.get('FULL_MIXED_ORACLE','graph'),descriptor=str(ctx.batch_descriptor),valid_tokens=valid_global,num_prefills=metadata.num_prefills,num_decodes=metadata.num_decodes,checks=checks,status='PASS'))
        Path(os.environ['FULL_MIXED_OUTPUT'],f'shadow-rank{runner._probe_rank}.json').write_text(json.dumps(runner._probe_shadows,indent=2))
    finally:
        ctx.cudagraph_runtime_mode = old_mode
        if cross_step is not None:
            cross_step.reference_end(ctx, reference_upper)
        for target, source in zip(caches, after): target.copy_(source)
        for target, source in zip(side, side_after): target.copy_(source)
    return replay

ACLGraphWrapper.__call__ = _checked_graph_call

def enable_shadow(self):
    runner = self.model_runner
    runner._probe_validate = True
    runner._probe_rank = self.rank
    runner._probe_shadows = []
    runner.model._probe_runner = runner
    return dict(rank=self.rank,enabled=True)

FullMixedProbeWorker.enable_shadow = enable_shadow

_original_pad = NPUModelRunner._pad_query_start_loc_for_fia

def _pad_dsa_capacity(self, query_start_loc, num_tokens_padded, num_reqs_padded,
                      num_reqs, cudagraph_runtime_mode=None, batch_desc_num_reqs=None):
    if cudagraph_runtime_mode != CUDAGraphMode.FULL:
        return _original_pad(self, query_start_loc, num_tokens_padded, num_reqs_padded,
                             num_reqs, cudagraph_runtime_mode, batch_desc_num_reqs)
    capacity = batch_desc_num_reqs
    assert capacity is not None and num_reqs <= capacity
    # DSACP queries carry explicit lengths. Inactive seats have zero length;
    # don't insert an extra FIA dummy request or leave stale captured tail rows.
    query_start_loc.np[num_reqs + 1:capacity + 1] = query_start_loc.np[num_reqs]
    query_start_loc.copy_to_gpu()
    return capacity

NPUModelRunner._pad_query_start_loc_for_fia = _pad_dsa_capacity

_fixed_build = AscendDSACPMetadataBuilder.build

def _build_fixed_capacity(self, *args, **kwargs):
    result = _fixed_build(self, *args, **kwargs)
    if self.vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL:
        result.num_actual_tokens = result.num_input_tokens
        req = result.req_metadata
        req.num_reqs_actual = req.query_start_loc.shape[0] - 1
        if self.compressor_ratio > 1:
            req.num_compressed_tokens = min(result.num_input_tokens,
                result.num_input_tokens // self.compressor_ratio + req.num_reqs_actual)
    return result

AscendDSACPMetadataBuilder.build = _build_fixed_capacity

# Native MRV1 uses max(K+1, TP) when sequence parallelism is enabled. max is
# not the joint alignment when neither divides the other (K5/TP8 needs24).
# This hook changes bucket alignment only, never the actual speculative length.
from vllm.config import CompilationConfig
_original_adjust_sizes = CompilationConfig.adjust_cudagraph_sizes_for_spec_decode

def _adjust_joint_alignment(self, uniform_decode_query_len, tensor_parallel_size):
    import math
    alignment = uniform_decode_query_len
    if self.pass_config.enable_sp:
        alignment = math.lcm(alignment, tensor_parallel_size)
    return _original_adjust_sizes(self, alignment, tensor_parallel_size)

CompilationConfig.adjust_cudagraph_sizes_for_spec_decode = _adjust_joint_alignment

# Unmodified target graph/metadata path for matched donor controls. Retain only
# dummy loading, receipts and the independently needed K5/TP8 LCM sizing repair.
if os.environ.get('FULL_MIXED_PATCH', '1') == '0':
    AscendDSACPMetadataBuilder.get_cudagraph_support = _original_support
    AscendDSACPMetadataBuilder.build = _original_build
    dsa_cp.get_cos_and_sin_dsa = _original_rope
    NPUModelRunner._pad_query_start_loc_for_fia = _original_pad

# Keep the shadow oracle above this experiment: it checks the actual selected
# replay call rather than silently reverting to the original synchronized path.
_native_graph_call = _original_graph_call

def _ordered_graph_call(self, *args, **kwargs):
    from ordered_replay import call
    return call(_native_graph_call, self, *args, **kwargs)

_original_graph_call = _ordered_graph_call
