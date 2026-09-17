"""Exact [decode=1, prefill=512] FULL shadow pilot, not a serving policy.

One token total owns one signature here. Do not generalize this key scheme to
different prefill partitions sharing a total; native FIA also keys by total.
"""

import copy
import dataclasses

import torch
from observe_worker import Worker as BaseWorker

SIGNATURE = (1, 512)
TOKENS = sum(SIGNATURE)


def install():
    from vllm.config import CUDAGraphMode
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.ops.gdn_attn_builder import (
        AscendGDNAttentionMetadataBuilder as Builder,
    )
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    original_build = Builder.build

    def build(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
        m = common_attn_metadata
        result = original_build(self, common_prefix_len, m, *args, **kwargs)
        lengths = tuple(m.query_start_loc_cpu.diff().tolist())
        if lengths != SIGNATURE:
            return result
        assert (result.num_decodes, result.num_prefills) == (1, 1)
        # Unused by the pinned Ascend conv path; do not retain alternate Triton
        # metadata whose host values need not have graph-stable identity.
        result.nums_dict = result.batch_ptr = result.token_chunk_offset_ptr = None
        if not hasattr(self, "_mixed_buffers"):
            self._mixed_buffers, self._mixed_scalars = {}, {}

        def stable(value, path):
            if isinstance(value, torch.Tensor):
                if path not in self._mixed_buffers:
                    self._mixed_buffers[path] = value.clone()
                dest = self._mixed_buffers[path]
                assert (dest.shape, dest.dtype) == (value.shape, value.dtype), path
                dest.copy_(value)
                return dest
            if dataclasses.is_dataclass(value):
                out = copy.copy(value)
                for key, child in vars(value).items():
                    setattr(out, key, stable(child, path + "." + key))
                return out
            if isinstance(value, (list, tuple)):
                return type(value)(
                    stable(v, f"{path}.{i}") for i, v in enumerate(value)
                )
            if isinstance(value, dict):
                return {k: stable(v, f"{path}.{k}") for k, v in value.items()}
            assert self._mixed_scalars.setdefault(path, value) == value, path
            return value

        return stable(result, "gdn")

    original_capture = Builder.build_for_cudagraph_capture

    def capture(self, metadata):
        if metadata.num_actual_tokens == TOKENS:
            return self.build(0, metadata)
        return original_capture(self, metadata)

    original_descriptor = CudagraphDispatcher._create_padded_batch_descriptor

    def descriptor(self, *args, **kwargs):
        result = original_descriptor(self, *args, **kwargs)
        if result.num_tokens == TOKENS:
            result = dataclasses.replace(result, num_reqs=len(SIGNATURE))
        return result

    original_determine = NPUModelRunner._determine_batch_execution_and_padding

    def determine(
        self,
        num_tokens,
        num_reqs,
        num_scheduled_tokens_np,
        max_num_scheduled_tokens,
        use_cascade_attn,
        **kwargs,
    ):
        if getattr(self, "_mixed_dummy", False):
            # Dummy's query_lens aliases this array; all later metadata consumes
            # the same exact partition, instead of native equal-size prefills.
            assert num_tokens == TOKENS and num_reqs == len(SIGNATURE)
            num_scheduled_tokens_np[:] = SIGNATURE
            max_num_scheduled_tokens = max(SIGNATURE)
        exact = tuple(num_scheduled_tokens_np.tolist()) == SIGNATURE
        decode = num_tokens == num_reqs and max_num_scheduled_tokens == 1
        if not exact and not decode:
            kwargs["force_eager"] = True
        return original_determine(
            self,
            num_tokens,
            num_reqs,
            num_scheduled_tokens_np,
            max_num_scheduled_tokens,
            use_cascade_attn,
            **kwargs,
        )

    original_attention = NPUModelRunner._build_attention_metadata

    def attention(self, *args, **kwargs):
        if kwargs.get("num_tokens") == TOKENS:
            self.attn_state = AscendAttentionState.ChunkedPrefill
        return original_attention(self, *args, **kwargs)

    original_dummy = NPUModelRunner._dummy_run

    def dummy(self, num_tokens, *args, **kwargs):
        old = self.scheduler_config.max_num_seqs
        self._mixed_dummy = (
            num_tokens == TOKENS
            and kwargs.get("cudagraph_runtime_mode") == CUDAGraphMode.FULL
        )
        try:
            if self._mixed_dummy:
                self.scheduler_config.max_num_seqs = len(SIGNATURE)
            return original_dummy(self, num_tokens, *args, **kwargs)
        finally:
            self.scheduler_config.max_num_seqs = old
            self._mixed_dummy = False

    Builder.build = build
    Builder.build_for_cudagraph_capture = capture
    Builder._cudagraph_support = AttentionCGSupport.ALWAYS
    CudagraphDispatcher._create_padded_batch_descriptor = descriptor
    NPUModelRunner._determine_batch_execution_and_padding = determine
    NPUModelRunner._build_attention_metadata = attention
    NPUModelRunner._dummy_run = dummy

    from shadow_worker import install_shadow

    install_shadow()
    shadow_forward = NPUModelRunner._model_forward

    def forward(self, *args, **kwargs):
        from vllm.forward_context import get_forward_context

        ctx = get_forward_context()
        if (
            getattr(self, "_mixed_shadow_budget", 0)
            and ctx.cudagraph_runtime_mode == CUDAGraphMode.FULL
            and ctx.batch_descriptor.num_tokens == TOKENS
        ):
            self._mixed_shadow_budget -= 1
            self._shadow_remaining = 1
        return shadow_forward(self, *args, **kwargs)

    NPUModelRunner._model_forward = forward


class Worker(BaseWorker):
    def __init__(self, vllm_config, *args, **kwargs):
        assert vllm_config.speculative_config is None
        assert not vllm_config.cache_config.enable_prefix_caching
        assert vllm_config.parallel_config.tensor_parallel_size == 2
        install()
        super().__init__(vllm_config, *args, **kwargs)

    def arm_mixed_shadow(self, steps=2):
        r = self.model_runner
        r._shadow_rank, r._shadow_results = self.rank, []
        r._mixed_shadow_budget = steps
        return dict(rank=self.rank, armed=steps)

    def mixed_shadow_result(self):
        r = self.model_runner
        return dict(
            rank=self.rank,
            remaining=r._mixed_shadow_budget,
            passed=r._mixed_shadow_budget == 0
            and all(x["passed"] for x in r._shadow_results),
            steps=r._shadow_results,
        )
