"""Single-card unchanged donor hybrid baseline, isolated from remote transport."""

import json
import os
from pathlib import Path
from vllm import LLM, SamplingParams
from settings import MODEL

llm = LLM(
    model=MODEL,
    hf_overrides=dict(num_hidden_layers=4),
    load_format="dummy",
    dtype="bfloat16",
    tensor_parallel_size=1,
    distributed_executor_backend="uni",
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
results = llm.generate(
    [dict(prompt_token_ids=[19] * 32)],
    SamplingParams(temperature=0, max_tokens=3, ignore_eos=True, detokenize=False),
)
Path(os.environ["LOCAL_EXPERT_RESULT"]).write_text(
    json.dumps(dict(status="pass", tokens=results[0].outputs[0].token_ids))
)
