"""Single-request GDN prefill graphs; native decode and mixed-batch fallback."""

import copy
import dataclasses

PREFILLS = (16, 32, 64, 128, 256, 512, 1024, 1536, 2048)


def prefill_bucket(tokens):
    if 1 < tokens <= PREFILLS[-1]:
        return next(n for n in PREFILLS if n >= tokens)
    return None


def install():
    import torch
    from vllm.config import CUDAGraphMode
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.ops.gdn_attn_builder import (
        AscendGDNAttentionMetadataBuilder as Builder,
    )
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from . import padding

    if getattr(Builder, "_betterscale_qwen_prefill", False):
        return
    padding.install()
    original_build = Builder.build

    def build(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
        m = common_attn_metadata
        if m.num_reqs > 1 and m.num_actual_tokens > 1:
            starts = m.query_start_loc_cpu
            if int(starts[1]) == m.num_actual_tokens and bool(
                (starts[1:] == starts[1]).all()
            ):
                # FIA may add empty virtual requests. Ascend's unpadded helper
                # retains its block table; GDN instead needs only the real row.
                m = m.unpadded(m.num_actual_tokens, 1)
                m.block_table_tensor = m.block_table_tensor[:1]
        metadata = original_build(self, common_prefix_len, m, *args, **kwargs)
        bucket = prefill_bucket(metadata.num_prefill_tokens)
        if not (
            metadata.num_prefills == 1
            and metadata.num_decodes == 0
            and metadata.num_spec_decodes == 0
            and bucket is not None
        ):
            return metadata
        metadata = padding.normalize(self, metadata, bucket)
        if not hasattr(self, "_betterscale_prefill_contracts"):
            self._betterscale_prefill_contracts = {}
        buffers, scalars = self._betterscale_prefill_contracts.setdefault(
            bucket, ({}, {})
        )

        def stable(value, path):
            if isinstance(value, torch.Tensor):
                if path not in buffers:
                    buffers[path] = value.clone()
                dest = buffers[path]
                if dest.shape != value.shape or dest.dtype != value.dtype:
                    raise RuntimeError(
                        f"Qwen FULL metadata tensor contract changed: {path}"
                    )
                dest.copy_(value)
                return dest
            if dataclasses.is_dataclass(value):
                result = copy.copy(value)
                for name, child in vars(value).items():
                    setattr(result, name, stable(child, path + "." + name))
                return result
            if isinstance(value, dict):
                return {k: stable(v, path + "." + str(k)) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return type(value)(
                    stable(v, path + "." + str(i)) for i, v in enumerate(value)
                )
            if path in scalars and scalars[path] != value:
                raise RuntimeError(
                    f"Qwen FULL metadata scalar contract changed: {path}"
                )
            scalars[path] = value
            return value

        return stable(metadata, "gdn")

    original_capture = Builder.build_for_cudagraph_capture

    def capture(self, metadata):
        if metadata.num_actual_tokens in PREFILLS:
            return self.build(0, metadata)
        return original_capture(self, metadata)

    original_attention = NPUModelRunner._build_attention_metadata

    def attention(self, *args, **kwargs):
        if (
            prefill_bucket(kwargs.get("num_tokens", 0)) is not None
            and kwargs.get("num_reqs") == 1
        ):
            self.attn_state = AscendAttentionState.ChunkedPrefill
        return original_attention(self, *args, **kwargs)

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
        pure_decode = num_tokens == num_reqs and bool(
            (num_scheduled_tokens_np == 1).all()
        )
        bucket = prefill_bucket(num_tokens)
        eligible = not pure_decode and num_reqs == 1 and bucket is not None
        if not pure_decode and not eligible:
            kwargs["force_eager"] = True
        return original_determine(
            self,
            bucket if eligible else num_tokens,
            num_reqs,
            num_scheduled_tokens_np,
            max_num_scheduled_tokens,
            use_cascade_attn,
            **kwargs,
        )

    original_descriptor = CudagraphDispatcher._create_padded_batch_descriptor

    def descriptor(self, *args, **kwargs):
        result = original_descriptor(self, *args, **kwargs)
        if result.num_tokens in PREFILLS:
            result = dataclasses.replace(result, num_reqs=1)
        return result

    original_dummy = NPUModelRunner._dummy_run

    def dummy(self, num_tokens, *args, **kwargs):
        single = (
            num_tokens in PREFILLS
            and kwargs.get("cudagraph_runtime_mode") == CUDAGraphMode.FULL
        )
        old_seats = self.scheduler_config.max_num_seqs
        try:
            if single:
                self.scheduler_config.max_num_seqs = 1
            return original_dummy(self, num_tokens, *args, **kwargs)
        finally:
            self.scheduler_config.max_num_seqs = old_seats

    Builder.build = build
    Builder.build_for_cudagraph_capture = capture
    Builder._cudagraph_support = AttentionCGSupport.ALWAYS
    CudagraphDispatcher._create_padded_batch_descriptor = descriptor
    NPUModelRunner._build_attention_metadata = attention
    NPUModelRunner._determine_batch_execution_and_padding = determine
    NPUModelRunner._dummy_run = dummy
    Builder._betterscale_qwen_prefill = True
