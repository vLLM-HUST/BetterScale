"""Bounded Qwen GDN FULL prefill experiment; not the packaged DSV4 Worker.

Transfers BetterScale target_full's stable-metadata/native-graph method.
Only one exact PREFILL-query seat plus donor decode is admitted.
"""

import copy
import dataclasses
import torch
from observe_worker import Worker as NPUWorker
import os
from pathlib import Path

PREFILL = int(os.environ.get("FIXED_PREFILL_TOKENS", "512"))


def install():
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm_ascend.ops.gdn_attn_builder import (
        AscendGDNAttentionMetadataBuilder as Builder,
    )
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    if getattr(Builder, "_qwen_full_pilot", False):
        return
    original = Builder.build

    def build(self, *args, **kwargs):
        metadata = original(self, *args, **kwargs)
        if metadata.num_prefills == 0:
            return metadata
        assert metadata.num_prefills == 1 and metadata.num_prefill_tokens == PREFILL
        assert metadata.num_decodes == 0 and metadata.num_spec_decodes == 0
        if not hasattr(self, "_prefill_buffers"):
            self._prefill_buffers = {}
            self._prefill_scalars = {}

        def stable(value, path):
            if isinstance(value, torch.Tensor):
                if path not in self._prefill_buffers:
                    self._prefill_buffers[path] = value.clone()
                dest = self._prefill_buffers[path]
                assert dest.shape == value.shape and dest.dtype == value.dtype, path
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
            if path in self._prefill_scalars:
                assert self._prefill_scalars[path] == value, (
                    path,
                    self._prefill_scalars[path],
                    value,
                )
            else:
                self._prefill_scalars[path] = value
            return value

        return stable(metadata, "gdn")

    old_capture = Builder.build_for_cudagraph_capture

    def capture(self, metadata):
        if metadata.num_actual_tokens == PREFILL:
            return self.build(0, metadata)
        return old_capture(self, metadata)

    old_attention = NPUModelRunner._build_attention_metadata

    def attention(self, *args, **kwargs):
        # Fixed large-bucket capture and real initial prefill share paged FIA,
        # avoiding capture with dummy DecodeOnly / runtime PrefillNoCache.
        if kwargs.get("num_tokens") == PREFILL:
            self.attn_state = AscendAttentionState.ChunkedPrefill
        return old_attention(self, *args, **kwargs)

    old_determine = NPUModelRunner._determine_batch_execution_and_padding

    def determine(self, *args, **kwargs):
        tokens = kwargs.get("num_tokens", args[0] if args else None)
        if tokens == PREFILL and not getattr(self, "_prefill_full", True):
            kwargs["force_eager"] = True
        return old_determine(self, *args, **kwargs)

    NPUModelRunner._determine_batch_execution_and_padding = determine
    Builder.build = build
    Builder.build_for_cudagraph_capture = capture
    Builder._cudagraph_support = AttentionCGSupport.ALWAYS
    NPUModelRunner._build_attention_metadata = attention
    Builder._qwen_full_pilot = True


class Worker(NPUWorker):
    def __init__(self, vllm_config, *args, **kwargs):
        assert vllm_config.scheduler_config.max_num_seqs == 1
        assert vllm_config.speculative_config is None
        assert vllm_config.parallel_config.tensor_parallel_size == 2
        install()
        super().__init__(vllm_config, *args, **kwargs)

    def set_prefill_full(self, mode):
        if mode not in ("full", "none"):
            raise ValueError(mode)
        self.model_runner._prefill_full = mode == "full"
        return dict(rank=self.rank, mode=mode)

    def start_prefill_profile(self, label):
        from profiling import ProfileWindow

        if self.window is not None and not self.window.closed:
            raise RuntimeError("overlapping profile windows")
        self.window = ProfileWindow(
            Path(os.environ["SERVING_PROFILE"]) / label, self.rank, steps=4
        )
        return dict(rank=self.rank, label=label)
