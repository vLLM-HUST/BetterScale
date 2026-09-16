"""Bounded Qwen GDN FULL prefill experiment; not the packaged DSV4 Worker.

Transfers BetterScale target_full's stable-metadata/native-graph method.
Exact single-prefill buckets use FULL; mixed/other lengths use compiled NONE.
Native decode graphs and eight-seat scheduling remain available.
"""

import copy
import dataclasses
import torch
from observe_worker import Worker as NPUWorker
import os
from pathlib import Path

PADDED = os.environ.get("PADDED_PREFILL") == "1"
SPEC = int(os.environ.get("FULL_MTP", "0"))
PREFILLS = (
    (64, 128, 256, 512, 1024, 1536, 2048)
    if SPEC
    else (
        (16, 32, 64, 128, 256, 512, 1024, 1536, 2048)
        if PADDED
        else (512, 1024, 1536, 2048)
    )
)


if SPEC:
    PREFILLS = tuple(((n + SPEC) // (SPEC + 1)) * (SPEC + 1) for n in PREFILLS)


def prefill_bucket(tokens):
    if PADDED and 1 < tokens <= max(PREFILLS):
        return next(n for n in PREFILLS if n >= tokens)
    return tokens if tokens in PREFILLS else None


def install():
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm_ascend.ops.gdn_attn_builder import (
        AscendGDNAttentionMetadataBuilder as Builder,
    )
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    if getattr(Builder, "_qwen_full_pilot", False):
        return
    if PADDED:
        import padded_gdn

        padded_gdn.install()
    original = Builder.build

    def build(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
        m = common_attn_metadata
        if PADDED and m.num_reqs > 1 and m.num_actual_tokens > 1:
            # Native FIA padding may append an empty GDN virtual request.
            # Keep the real slot/endpoints before stable single-prefill binding;
            # otherwise the extra row silently bypasses the graph contract.
            starts = m.query_start_loc_cpu
            if int(starts[1]) == m.num_actual_tokens and bool(
                (starts[1:] == starts[1]).all()
            ):
                m = m.unpadded(m.num_actual_tokens, 1)
                # Ascend's override deliberately keeps the FULL block table for
                # FIA; GDN state indices instead require the real request rows.
                m.block_table_tensor = m.block_table_tensor[:1]
        metadata = original(self, common_prefix_len, m, *args, **kwargs)
        if metadata.num_prefills == 0:
            return metadata
        if not (
            metadata.num_prefills == 1
            and prefill_bucket(metadata.num_prefill_tokens) is not None
            and metadata.num_decodes == 0
            and metadata.num_spec_decodes == 0
        ):
            return metadata
        if PADDED:
            metadata = padded_gdn.normalize(
                self, metadata, prefill_bucket(metadata.num_prefill_tokens)
            )
        if not hasattr(self, "_prefill_contracts"):
            self._prefill_contracts = {}
        buffers, scalars = self._prefill_contracts.setdefault(
            metadata.num_prefill_tokens, ({}, {})
        )

        def stable(value, path):
            if isinstance(value, torch.Tensor):
                if path not in buffers:
                    buffers[path] = value.clone()
                dest = buffers[path]
                assert dest.shape == value.shape and dest.dtype == value.dtype, (
                    path,
                    tuple(dest.shape),
                    tuple(value.shape),
                    dest.dtype,
                    value.dtype,
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
            if path in scalars:
                assert scalars[path] == value, (
                    path,
                    scalars[path],
                    value,
                )
            else:
                scalars[path] = value
            return value

        return stable(metadata, "gdn")

    old_capture = Builder.build_for_cudagraph_capture

    def capture(self, metadata):
        if metadata.num_actual_tokens in PREFILLS:
            return self.build(0, metadata)
        return old_capture(self, metadata)

    old_attention = NPUModelRunner._build_attention_metadata

    def attention(self, *args, **kwargs):
        # Fixed large-bucket capture and real initial prefill share paged FIA,
        # avoiding capture with dummy DecodeOnly / runtime PrefillNoCache.
        if (
            prefill_bucket(kwargs.get("num_tokens", 0)) is not None
            and kwargs.get("num_reqs") == 1
        ):
            self.attn_state = AscendAttentionState.ChunkedPrefill
        return old_attention(self, *args, **kwargs)

    old_determine = NPUModelRunner._determine_batch_execution_and_padding

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
        if (
            SPEC
            and getattr(self, "_bucket_dummy_active", False)
            and kwargs.get("force_uniform_decode")
        ):
            pure_decode = True
        prompting = True
        if SPEC and not getattr(self, "_bucket_dummy_active", False):
            computed = self.input_batch.num_computed_tokens_cpu[:num_reqs]
            prompts = self.input_batch.num_prompt_tokens_cpu_tensor[:num_reqs].numpy()
            prompting = bool((computed < prompts).any())
            pure_decode = not prompting and max_num_scheduled_tokens <= SPEC + 1
        eligible = (
            prompting
            and not pure_decode
            and num_reqs == 1
            and prefill_bucket(num_tokens) is not None
        )
        from vllm.config import CUDAGraphMode

        piecewise_capture = (
            getattr(self, "_bucket_dummy_mode", None) == CUDAGraphMode.PIECEWISE
        )
        dispatch_tokens = (
            prefill_bucket(num_tokens) if PADDED and eligible else num_tokens
        )
        if (
            not piecewise_capture
            and not pure_decode
            and (not eligible or not getattr(self, "_prefill_full", True))
        ):
            kwargs["force_eager"] = True
        result = old_determine(
            self,
            dispatch_tokens,
            num_reqs,
            num_scheduled_tokens_np,
            max_num_scheduled_tokens,
            use_cascade_attn,
            **kwargs,
        )
        from collections import Counter

        if not hasattr(self, "_dispatch_counts"):
            self._dispatch_counts = Counter()
        self._dispatch_counts[(str(result[0]), num_tokens, num_reqs)] += 1
        if getattr(self, "_bucket_dummy_active", False):
            print(
                "BUCKET_CAPTURE_DISPATCH",
                dict(
                    tokens=num_tokens,
                    requests=num_reqs,
                    scheduled=num_scheduled_tokens_np.tolist(),
                    eligible=eligible,
                    force_eager=kwargs.get("force_eager"),
                    mode=str(result[0]),
                ),
                flush=True,
            )
        return result

    # FULL descriptors own request-count identity as well as token shape.
    # Register and look up the same single-prefill key, rather than mutating
    # scheduler seats after keys were registered for eight requests.
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

    old_descriptor = CudagraphDispatcher._create_padded_batch_descriptor

    def descriptor(self, *args, **kwargs):
        result = old_descriptor(self, *args, **kwargs)
        if result.num_tokens in PREFILLS:
            result = dataclasses.replace(result, num_reqs=1)
        return result

    CudagraphDispatcher._create_padded_batch_descriptor = descriptor

    if SPEC:
        # Keep native uniform speculative decode keys and add distinct prefill keys.
        old_initialize = CudagraphDispatcher.initialize_cudagraph_keys

        def initialize(self, *args, **kwargs):
            from vllm.config import CUDAGraphMode

            result = old_initialize(self, *args, **kwargs)
            for n in PREFILLS:
                if n in self.compilation_config.cudagraph_capture_sizes:
                    self.cudagraph_keys[CUDAGraphMode.FULL].add(
                        self._create_padded_batch_descriptor(n, False, False)
                    )
            return result

        CudagraphDispatcher.initialize_cudagraph_keys = initialize

        from vllm_ascend.spec_decode.llm_base_proposer import (
            AscendSpecDecodeBaseProposer,
        )

        old_propose = AscendSpecDecodeBaseProposer._propose

        def propose(self, *args, **kwargs):
            # Draft first-pass batching is not the target's single-prefill
            # contract. Keep native uniform decode FULL, not our prefill keys.
            from vllm.config import CUDAGraphMode

            dispatcher = self.runner.cudagraph_dispatcher
            keys = dispatcher.cudagraph_keys[CUDAGraphMode.FULL]
            try:
                dispatcher.cudagraph_keys[CUDAGraphMode.FULL] = {
                    k for k in keys if k.uniform
                }
                return old_propose(self, *args, **kwargs)
            finally:
                dispatcher.cudagraph_keys[CUDAGraphMode.FULL] = keys

        AscendSpecDecodeBaseProposer._propose = propose
    old_dummy = NPUModelRunner._dummy_run

    def dummy(self, num_tokens, *args, **kwargs):
        # Large graphs represent exactly one prefill. Restore the scheduler
        # configuration before serving; this does not restrict live concurrency.
        from vllm.config import CUDAGraphMode

        single = (
            num_tokens in PREFILLS
            and kwargs.get("cudagraph_runtime_mode") == CUDAGraphMode.FULL
        )
        old_seats = self.scheduler_config.max_num_seqs
        self._bucket_dummy_active = True
        self._bucket_dummy_mode = kwargs.get("cudagraph_runtime_mode")
        saved_full_keys = None
        if SPEC and self._bucket_dummy_mode == CUDAGraphMode.PIECEWISE:
            saved_full_keys = self.cudagraph_dispatcher.cudagraph_keys[
                CUDAGraphMode.FULL
            ]
            self.cudagraph_dispatcher.cudagraph_keys[CUDAGraphMode.FULL] = set()
        try:
            if single:
                self.scheduler_config.max_num_seqs = 1
            return old_dummy(self, num_tokens, *args, **kwargs)
        finally:
            if saved_full_keys is not None:
                self.cudagraph_dispatcher.cudagraph_keys[CUDAGraphMode.FULL] = (
                    saved_full_keys
                )
            self.scheduler_config.max_num_seqs = old_seats
            self._bucket_dummy_active = False
            self._bucket_dummy_mode = None

    NPUModelRunner._dummy_run = dummy
    NPUModelRunner._determine_batch_execution_and_padding = determine
    Builder.build = build
    Builder.build_for_cudagraph_capture = capture
    Builder._cudagraph_support = AttentionCGSupport.ALWAYS
    NPUModelRunner._build_attention_metadata = attention
    Builder._qwen_full_pilot = True


class Worker(NPUWorker):
    def __init__(self, vllm_config, *args, **kwargs):
        assert vllm_config.scheduler_config.max_num_seqs == 8
        if SPEC:
            assert vllm_config.speculative_config is not None
            assert vllm_config.speculative_config.num_speculative_tokens == SPEC
            assert vllm_config.speculative_config.method == "mtp"
        else:
            assert vllm_config.speculative_config is None
        assert vllm_config.parallel_config.tensor_parallel_size == 2
        install()
        super().__init__(vllm_config, *args, **kwargs)

    def set_prefill_full(self, mode):
        if mode not in ("full", "none"):
            raise ValueError(mode)
        self.model_runner._prefill_full = mode == "full"
        return dict(rank=self.rank, mode=mode)

    def set_prefill_policy(self, mode):
        # Diagnostic RPC: called only between fully completed requests.
        global PADDED
        if mode not in ("native", "padded-none", "full"):
            raise ValueError(mode)
        PADDED = mode != "native"
        self.model_runner._prefill_full = mode == "full"
        return dict(rank=self.rank, policy=mode)

    def get_dispatch_counts(self):
        return dict(
            rank=self.rank,
            counts=[
                dict(mode=k[0], tokens=k[1], requests=k[2], count=v)
                for k, v in self.model_runner._dispatch_counts.items()
            ],
        )

    def start_prefill_profile(self, label):
        from profiling import ProfileWindow

        if self.window is not None and not self.window.closed:
            raise RuntimeError("overlapping profile windows")
        self.window = ProfileWindow(
            Path(os.environ["SERVING_PROFILE"]) / label, self.rank, steps=4
        )
        return dict(rank=self.rank, label=label)
