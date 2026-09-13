"""Target FULL admission and persistent metadata; no alignment or replay hooks.

Migrated from the qualified target patch; no shadow, fixtures or N+2 scheduler.
"""
from contextvars import ContextVar
from vllm.config import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport
from vllm_ascend.attention.context_parallel import dsa_cp
from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPMetadataBuilder
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

_original_build=AscendDSACPMetadataBuilder.build
_original_rope=dsa_cp.get_cos_and_sin_dsa
_original_pad=NPUModelRunner._pad_query_start_loc_for_fia
_target_build=ContextVar('strengthen_target_build',default=False)
_installed=False

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

_fixed_build=_build_target


def install():
    """Install before runner construction/capture; importing alone changes nothing."""
    global _installed
    if _installed:
        return
    AscendDSACPMetadataBuilder.get_cudagraph_support=classmethod(
        lambda cls,config,spec: AttentionCGSupport.ALWAYS)
    dsa_cp.get_cos_and_sin_dsa=_stable_rope
    AscendDSACPMetadataBuilder.build=_build_fixed_capacity
    NPUModelRunner._pad_query_start_loc_for_fia=_pad_dsa_capacity
    _installed=True
