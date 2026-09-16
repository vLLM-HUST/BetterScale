"""One-card native control; full layer dimensions, two dummy layers."""

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from vllm import LLM, SamplingParams

llm = LLM(
    model="/data/shared_models/Qwen3-30B-A3B",
    hf_overrides=dict(num_hidden_layers=2),
    load_format="dummy",
    dtype="bfloat16",
    tensor_parallel_size=1,
    distributed_executor_backend="uni",
    worker_cls="local_expert_worker.LocalExpertWorker",
    enforce_eager=True,
    max_model_len=256,
    max_num_batched_tokens=32,
    max_num_seqs=1,
    kv_cache_memory_bytes=128 * 1024**2,
    enable_prefix_caching=False,
    skip_tokenizer_init=True,
    async_scheduling=False,
    additional_config=dict(enable_cpu_binding=False),
)
llm.collective_rpc("arm")
params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True, detokenize=False)
for size in (16, 32):
    result = llm.generate([dict(prompt_token_ids=[19] * size)], params)
    assert len(result[0].outputs[0].token_ids) == 3
print("LOCAL_EXPERT_GRAPH_PASS", flush=True)
