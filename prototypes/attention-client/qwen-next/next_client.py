"""Separate native vLLM client process; no inherited expert pipe or parent engine."""

import json
import os
from pathlib import Path
import time

from vllm import LLM, SamplingParams
from settings import MODEL, LAYERS, REAL

source = int(os.environ["EXPERT_SOURCE_ID"])
llm = LLM(
    model=MODEL,
    hf_overrides=dict(num_hidden_layers=LAYERS),
    load_format="auto" if REAL else "dummy",
    dtype="bfloat16",
    tensor_parallel_size=1,
    distributed_executor_backend="uni",
    worker_cls="next_worker.Worker",
    enforce_eager=True,
    max_model_len=256,
    max_num_batched_tokens=32,
    max_num_seqs=1,
    kv_cache_memory_bytes=128 * 1024**2,
    enable_prefix_caching=False,
    skip_tokenizer_init=not REAL,
    async_scheduling=False,
    additional_config=dict(enable_cpu_binding=False),
)
outputs = []
# Unequal source lifetimes and more than32 source-layer jobs. No server task budget.
for index in range(2 + source):
    prompt = [19 + source] * (16 if index % 2 else 32)
    if REAL:
        question = [
            "What is 2 + 3? Answer with only the number.",
            "What is the capital of France? Answer briefly.",
            "Complete the sequence: 2, 4, 6, 8,",
        ][index]
        prompt = llm.get_tokenizer().apply_chat_template(
            [dict(role="user", content=question)],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    start = time.monotonic()
    result = llm.generate(
        [dict(prompt_token_ids=prompt)],
        SamplingParams(
            temperature=0,
            max_tokens=16 if REAL else 3 + source,
            ignore_eos=not REAL,
            detokenize=REAL,
        ),
    )
    ids = result[0].outputs[0].token_ids
    assert ids and (REAL or len(ids) == 3 + source)
    outputs.append(
        dict(
            tokens=ids, text=result[0].outputs[0].text, seconds=time.monotonic() - start
        )
    )
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
