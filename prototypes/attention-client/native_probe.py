"""Reduced Qwen3 MoE native layer oracle; dummy weights, not serving performance."""
import os
from vllm import LLM, SamplingParams
llm = LLM(
    model='/data/shared_models/Qwen3-30B-A3B',
    hf_overrides=dict(num_hidden_layers=2, hidden_size=256, intermediate_size=512,
                      moe_intermediate_size=128, num_experts=8, num_experts_per_tok=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=64),
    load_format='dummy', dtype='bfloat16', tensor_parallel_size=1,
    worker_cls='native_worker.AttentionProbeWorker',
    enforce_eager=True, max_model_len=1024, max_num_batched_tokens=256,
    max_num_seqs=4, kv_cache_memory_bytes=128*1024**2,
    enable_prefix_caching=False, skip_tokenizer_init=True,
    async_scheduling=False,
    additional_config=dict(enable_cpu_binding=False),
)
llm.collective_rpc('arm')
for size in (16, 128, 384):
    result = llm.generate([dict(prompt_token_ids=[17]*size), dict(prompt_token_ids=[19]*(size//2))],
                         SamplingParams(temperature=0, max_tokens=3, ignore_eos=True, detokenize=False))
    assert len(result)==2 and all(len(r.outputs[0].token_ids)==3 for r in result)
print('ATTENTION_NATIVE_PASS',flush=True)
