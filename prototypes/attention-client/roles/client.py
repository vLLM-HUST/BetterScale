"""Separate native vLLM client process; no inherited expert pipe or parent engine."""

import json
import os
from pathlib import Path
import time

from vllm import LLM, SamplingParams

source = int(os.environ["EXPERT_SOURCE_ID"])
llm = LLM(
    model="/data/shared_models/Qwen3-30B-A3B",
    hf_overrides=dict(num_hidden_layers=2),
    load_format="dummy",
    dtype="bfloat16",
    tensor_parallel_size=1,
    distributed_executor_backend="uni",
    worker_cls="role_worker.Worker",
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
outputs = []
# Unequal source lifetimes and more than32 source-layer jobs. No server task budget.
for index in range(7 + source):
    prompt = [19 + source] * (16 if index % 2 else 32)
    start = time.monotonic()
    result = llm.generate(
        [dict(prompt_token_ids=prompt)],
        SamplingParams(
            temperature=0, max_tokens=3 + source, ignore_eos=True, detokenize=False
        ),
    )
    ids = result[0].outputs[0].token_ids
    assert len(ids) == 3 + source
    outputs.append(dict(tokens=ids, seconds=time.monotonic() - start))
audit = (
    llm.collective_rpc("audit_experts")
    if os.environ.get("EXPERT_ROLE_AUDIT") == "1"
    else None
)
receipt = llm.collective_rpc("drain_experts")
Path(os.environ["EXPERT_ROLE_DIRECTORY"], f"client{source}.json").write_text(
    json.dumps(
        dict(status="pass", requests=outputs, drain=receipt, audit=audit), indent=2
    )
)
