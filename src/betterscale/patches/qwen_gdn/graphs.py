"""Token capacity policy; publication owns the actual runner overrides."""

import dataclasses

PREFILLS = (16, 32, 64, 128, 256, 512, 1024, 1536, 2048)


def descriptor(result):
    return (
        dataclasses.replace(result, num_reqs=8)
        if result.num_tokens in PREFILLS
        else result
    )


def capacity(self, num_tokens, num_reqs, num_scheduled_tokens_np):
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
    return capacity


def attention(self, tokens):
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    if tokens > 8:
        self.attn_state = AscendAttentionState.ChunkedPrefill
