"""Authored nonrepetitive chat inputs, not an accuracy or SWE-bench dataset."""
import json
from pathlib import Path
import sys
from transformers import AutoTokenizer

model = '/models/vllm-ascend-models/Qwen3.8-27B'
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
prompts = {
    'queue': 'A Python HTTP server uses a single worker thread to read a queue and execute database writes. Under load, response latency increases even though CPU utilization is low. Describe three plausible causes and one measurement for each. Do not propose increasing the thread count before identifying the bottleneck.',
    'cache': 'Review this cache design: keys are (user_id, document_id), values are the rendered document. The document can be edited and access permissions can change. Workers have independent local caches and no invalidation messages. List the correctness risks and propose a small design that preserves authorization checks without flushing every cache on every edit.',
    'prefill': 'Explain why a GPU graph can substantially reduce latency for a short transformer prefill but provide little speedup for a long prefill, even when both execute the same layers. Distinguish host submission time, device compute time, communication waiting, and an event synchronization wait. Give a concrete experimental control that would distinguish overlap from an ineffective graph.',
    'retry': 'An HTTP client retries a POST that transfers money after a network timeout. The server may have committed the transfer before the connection failed. Explain an idempotency-key protocol that avoids duplicate transfers, including concurrent retries, transaction boundaries, expiry, and how to return the original result. State one thing that an idempotency key alone cannot guarantee.',
}
rows = []
for name, prompt in prompts.items():
    text = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': 'You are a precise technical assistant. Answer directly and concisely.'}, {'role': 'user', 'content': prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    rows.append(dict(name=name, prompt_ids=ids))
    print(name, len(ids))
Path(sys.argv[1]).write_text(json.dumps(dict(model=model, scope=__doc__, prompts=rows)))
