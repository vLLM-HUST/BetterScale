"""Unprofiled fixed-batch target-only diagnostic, never a SWE leaderboard point."""
import argparse
import json
from pathlib import Path
import time
import requests
from transformers import AutoTokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
tokenizer = AutoTokenizer.from_pretrained('/models/Qwen3.5-35B-A3B')
prompts = [tokenizer.encode(f'Continue counting from {i}: 1, 2, 3,', add_special_tokens=False)
           for i in range(16)]
s = requests.Session()
rows = []
for repeat in range(3):
    payload = dict(model='qwen35-overlap', prompt=prompts, temperature=0,
        max_tokens=512, ignore_eos=True, return_token_ids=True,
        cache_salt=f'overlap-fixed-batch-{repeat}')
    start = time.perf_counter()
    response = s.post('http://127.0.0.1:27720/v1/completions', json=payload, timeout=180)
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    data = response.json()
    assert len(data['choices']) == 16 and data['usage']['completion_tokens'] == 8192
    (a.output / f'repeat{repeat}.json').write_text(json.dumps(data))
    rows.append(dict(repeat=repeat, seconds=elapsed, output_tokens=8192,
                     output_tps=8192/elapsed, output_tps_per_chip=4096/elapsed))
    print(json.dumps(rows[-1]), flush=True)
(a.output / 'summary.json').write_text(json.dumps(dict(samples=rows,
    scope='Three fixed short-prompt 16x512-token batches; not SWE or long-context throughput'), indent=2))
