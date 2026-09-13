"""Opt-in A2/TP1 DSA target FULL graph; native DP/EP and draft are untouched.

The native 'decode' kernels already consume ragged causal query ranges. Use
that single program for all target rows, rather than capturing Python's changing
[decode | prefill] split. This is experimental until native-state shadow passes.
"""
from contextvars import ContextVar
from copy import copy
from dataclasses import fields

from vllm.config import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport
from vllm_ascend.attention import dsa_v1 as dsa
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

native_reference = ContextVar('dp_full_native_reference', default=False)
unified_build = ContextVar('dp_full_unified_build', default=False)
original_build = dsa.AscendDSAMetadataBuilder.build
original_split = dsa.split_decodes_and_prefills
original_pad = NPUModelRunner._pad_query_start_loc_for_fia


def split_queries(common, *args, **kwargs):
    if unified_build.get():
        return common.num_reqs, 0, common.num_input_tokens, 0
    return original_split(common, *args, **kwargs)


def build(self, common_prefix_len, common_attn_metadata, fast_build=False, **kwargs):
    if native_reference.get():
        if native_reference.get() == 'padded':
            common_attn_metadata = copy(common_attn_metadata)
            common_attn_metadata.num_actual_tokens = common_attn_metadata.num_input_tokens
        return original_build(self, common_prefix_len, common_attn_metadata, fast_build, **kwargs)
    assert self.vllm_config.parallel_config.tensor_parallel_size == 1
    common = copy(common_attn_metadata)
    common.num_actual_tokens = common.num_input_tokens
    # The shared dictionary is populated once per wave, before any layer's
    # forward; no per-layer metadata work is introduced by this hook.
    token = unified_build.set(True)
    try:
        result = original_build(self, common_prefix_len, common, fast_build, **kwargs)
    finally:
        unified_build.reset(token)
    # Clear inactive state maps using native actual-request handling FIRST.
    # Then bind captured scalar/shape bounds to the zero-length padded seats.
    result.decode.num_reqs_actual = common.num_reqs
    decode_limit = self.decode_threshold * self.vllm_config.scheduler_config.max_num_seqs
    if common.num_input_tokens > decode_limit:
        # Preserve native large-prefill prolog arithmetic (BF16 norm then
        # quant), rather than silently replacing it with decode's fused norm.
        # All metadata pointers still come from the stable ragged-query bank.
        req = result.decode
        values = {f.name: getattr(req, f.name) for f in fields(dsa.AscendDSAPrefillMetadata)
                  if hasattr(req, f.name)}
        values.update(query_lens=result.query_lens, context_lens=req.seq_lens,
                      max_query_len=req.max_seqlen_q)
        result.prefill = dsa.AscendDSAPrefillMetadata(**values)
        result.decode = None
        result.num_prefills, result.num_decodes, result.num_decode_tokens = common.num_reqs, 0, 0
    return result


def pad_queries(self, query_start_loc, num_tokens_padded, num_reqs_padded,
                num_reqs, cudagraph_runtime_mode=None, batch_desc_num_reqs=None):
    if cudagraph_runtime_mode != CUDAGraphMode.FULL:
        return original_pad(self, query_start_loc, num_tokens_padded, num_reqs_padded,
                            num_reqs, cudagraph_runtime_mode, batch_desc_num_reqs)
    capacity = batch_desc_num_reqs
    assert capacity is not None and num_reqs <= capacity
    query_start_loc.np[num_reqs + 1:capacity + 1] = query_start_loc.np[num_reqs]
    query_start_loc.copy_to_gpu()
    return capacity


dsa.split_decodes_and_prefills = split_queries
dsa.AscendDSAMetadataBuilder.build = build
dsa.AscendDSAMetadataBuilder.get_cudagraph_support = classmethod(
    lambda cls, config, spec: AttentionCGSupport.ALWAYS)
NPUModelRunner._pad_query_start_loc_for_fia = pad_queries

# Keep the native builder entry arguments for an optional independent oracle.
# Draft has its separate build_for_drafting entry and is not normalized here.
original_metadata = NPUModelRunner._build_attention_metadata


def remember_metadata(self, *args, **kwargs):
    self._dp_full_metadata_call = (args, kwargs)
    return original_metadata(self, *args, **kwargs)


NPUModelRunner._build_attention_metadata = remember_metadata

from donor_dp_worker import DonorDPWorker


class DPFullWorker(DonorDPWorker):
    def enable_dp_shadow(self):
        from dp_full_shadow import install
        return install(self)

    def donor_receipt(self):
        result = super().donor_receipt()
        model = self.model_runner.model
        result['target_graphs'] = [dict(descriptor=str(key), captured=value.aclgraph is not None,
                                       replays=getattr(model, '_dp_full_replays', {}).get(str(key), 0))
                                  for key, value in model.concrete_aclgraph_entries.items()]
        return result


from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm.forward_context import get_forward_context
original_call = ACLGraphWrapper.__call__


def graph_call(self, *args, **kwargs):
    ctx = get_forward_context()
    entry = self.concrete_aclgraph_entries.get(ctx.batch_descriptor)
    replay = ctx.cudagraph_runtime_mode == CUDAGraphMode.FULL and entry is not None and entry.aclgraph is not None
    if replay:
        counts = self.__dict__.setdefault('_dp_full_replays', {})
        key = str(ctx.batch_descriptor)
        counts[key] = counts.get(key, 0) + 1
        runner = self.__dict__.get('_dp_full_shadow_runner')
        if runner is not None and (len(runner._dp_full_checks) < 40
                                   or (ctx.batch_descriptor.num_tokens > 12 and counts[key] <= 4)):
            from dp_full_shadow import check
            return check(self, runner, original_call, args, kwargs)
    return original_call(self, *args, **kwargs)


ACLGraphWrapper.__call__ = graph_call
