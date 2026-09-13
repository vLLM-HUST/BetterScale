"""Target FULL admission and persistent metadata; no alignment or replay hooks.

Migrated from the qualified target patch; no shadow, fixtures or N+2 scheduler.
"""

from contextvars import ContextVar
from vllm.config import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport
from vllm_ascend.attention.context_parallel import dsa_cp
from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPMetadataBuilder
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

_original_build = AscendDSACPMetadataBuilder.build
_original_rope = dsa_cp.get_cos_and_sin_dsa
_original_pad = NPUModelRunner._pad_query_start_loc_for_fia
_target_build = ContextVar("strengthen_target_build", default=False)
_installed = False


def _stable_rope(*args, **kwargs):
    if _target_build.get():
        kwargs["use_cache"] = True
    return _original_rope(*args, **kwargs)


def _build_target(self, *args, **kwargs):
    token = _target_build.set(True)
    try:
        return _original_build(self, *args, **kwargs)
    finally:
        _target_build.reset(token)


def _pad_dsa_capacity(
    self,
    query_start_loc,
    num_tokens_padded,
    num_reqs_padded,
    num_reqs,
    cudagraph_runtime_mode=None,
    batch_desc_num_reqs=None,
):
    if cudagraph_runtime_mode != CUDAGraphMode.FULL:
        return _original_pad(
            self,
            query_start_loc,
            num_tokens_padded,
            num_reqs_padded,
            num_reqs,
            cudagraph_runtime_mode,
            batch_desc_num_reqs,
        )
    capacity = batch_desc_num_reqs
    assert capacity is not None and num_reqs <= capacity
    # DSACP queries carry explicit lengths. Inactive seats have zero length;
    # don't insert an extra FIA dummy request or leave stale captured tail rows.
    query_start_loc.np[num_reqs + 1 : capacity + 1] = query_start_loc.np[num_reqs]
    query_start_loc.copy_to_gpu()
    return capacity


def _build_fixed_capacity(self, *args, **kwargs):
    result = _fixed_build(self, *args, **kwargs)
    if self.vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL:
        result.num_actual_tokens = result.num_input_tokens
        req = result.req_metadata
        req.num_reqs_actual = req.query_start_loc.shape[0] - 1
        if self.compressor_ratio > 1:
            req.num_compressed_tokens = min(
                result.num_input_tokens,
                result.num_input_tokens // self.compressor_ratio + req.num_reqs_actual,
            )
    return result


_fixed_build = _build_target


def install(*, native_dsa=False):
    """Install before runner construction/capture; importing alone changes nothing."""
    if native_dsa:
        from . import _dp

        _dp.install()
        return
    global _installed
    if _installed:
        return
    # 放行 target 的 FULL prefill / mixed：原生 UNIFORM_BATCH 只承诺均匀 query
    # 批次（例如 K5 的 [6, 6, 6]），ALWAYS 也允许 [6, 17, 128] 这样的非均匀批次。
    # 这是能力声明，不是直接执行 graph；下面的 RoPE/容量修复负责兑现该声明。
    #
    # 初始化时（必须在 capture 前安装）：
    #   NPUModelRunner._check_and_update_cudagraph_mode()
    #     → 询问所有 attention builder，取最低支持等级
    #     → CompilationConfig.resolve_cudagraph_mode_and_sizes()
    #       用户要求 FULL 且最低等级为 ALWAYS，才不会因 mixed 支持不足降级
    #     → CudagraphDispatcher.initialize_cudagraph_keys()
    #       按 capture sizes 注册支持非均匀批次的 FULL descriptors。
    #   这不会强制覆盖用户模式，也不能绕过其他 backend 的更低支持等级。
    #
    # 一波请求实际怎样进入 FULL prefill / mixed：
    #   原生 scheduler 组好这一波（本补丁不改调度）
    #     → runner._determine_batch_execution_and_padding() 计算 token 数与 uniform_decode
    #     → dispatcher.dispatch() 按桶补齐并匹配 BatchDescriptor
    #       命中允许的 FULL key，返回 (FULL, descriptor)
    #     → 原生 forward context 携带该模式/descriptor
    #     → ACLGraphWrapper 按 descriptor 找到捕获的 target 图并 replay。
    #   因此不是另加一个“prefill 专用调用入口”，而是让非均匀波次也能匹配原生 FULL 图。
    #   没有可匹配的桶或被 force_eager 等条件排除时，仍走原生回退；并非任何请求都强制入图。
    AscendDSACPMetadataBuilder.get_cudagraph_support = classmethod(
        lambda cls, config, spec: AttentionCGSupport.ALWAYS
    )
    dsa_cp.get_cos_and_sin_dsa = _stable_rope
    AscendDSACPMetadataBuilder.build = _build_fixed_capacity
    NPUModelRunner._pad_query_start_loc_for_fia = _pad_dsa_capacity
    _installed = True
