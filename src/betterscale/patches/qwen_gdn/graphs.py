"""Elastic token-capacity graphs; request partitions are device metadata, not keys."""

import dataclasses

PREFILLS = (16, 32, 64, 128, 256, 512, 1024, 1536, 2048)


def install():
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    original_descriptor = CudagraphDispatcher._create_padded_batch_descriptor

    def descriptor(self, *args, **kwargs):
        result = original_descriptor(self, *args, **kwargs)
        if result.num_tokens in PREFILLS:
            result = dataclasses.replace(result, num_reqs=8)
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
        decode = num_tokens == num_reqs and bool((num_scheduled_tokens_np == 1).all())
        if decode and not getattr(self, "_elastic_dummy", False):
            computed = self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs]
            prompts = self.input_batch.num_prompt_tokens_cpu_tensor[:num_reqs]
            decode = bool((computed >= prompts).all())
        capacity = (
            num_tokens
            if decode
            else next((n for n in PREFILLS if n >= num_tokens), num_tokens)
        )
        return original_determine(
            self,
            capacity,
            num_reqs,
            num_scheduled_tokens_np,
            max_num_scheduled_tokens,
            use_cascade_attn,
            **kwargs,
        )

    original_attention = NPUModelRunner._build_attention_metadata

    def attention(self, *args, **kwargs):
        if kwargs.get("num_tokens_padded", 0) > 8:
            self.attn_state = AscendAttentionState.ChunkedPrefill
        return original_attention(self, *args, **kwargs)

    original_dummy = NPUModelRunner._dummy_run

    def dummy(self, *args, **kwargs):
        self._elastic_dummy = True
        try:
            return original_dummy(self, *args, **kwargs)
        finally:
            self._elastic_dummy = False

    CudagraphDispatcher._create_padded_batch_descriptor = descriptor
    NPUModelRunner._determine_batch_execution_and_padding = determine
    NPUModelRunner._build_attention_metadata = attention
    NPUModelRunner._dummy_run = dummy
