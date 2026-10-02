"""Bounded real-model HTTP gate; timing here is not a leaderboard score."""
import argparse
import json
import os
from pathlib import Path
import time
import requests
from transformers import AutoTokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--profile', action='store_true')
p.add_argument('--server-output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
url = 'http://127.0.0.1:27720'
s = requests.Session()
deadline = time.monotonic() + 900
while True:
    launch = a.server_output / 'launch.json'
    if launch.exists():
        try:
            os.kill(json.loads(launch.read_text())['pid'], 0)
        except ProcessLookupError as error:
            raise RuntimeError('Owned server exited before readiness') from error
    try:
        if s.get(url + '/health', timeout=3).status_code == 200:
            break
    except requests.RequestException:
        pass
    if time.monotonic() > deadline:
        raise TimeoutError('Assigned server did not become healthy in 900s')
    time.sleep(5)

tokenizer = AutoTokenizer.from_pretrained('/models/Qwen3.5-35B-A3B')
receipts = []
for i, length in enumerate((257, 4097, 8193)):
    body = tokenizer.encode('Keep this note. The access code is MARBLE. ', add_special_tokens=False)
    tail = tokenizer.encode('\nWhat is the access code? Answer briefly:', add_special_tokens=False)
    tokens = (body * (length // len(body) + 1))[:length - len(tail)] + tail
    payload = dict(model='qwen35-overlap', prompt=tokens, max_tokens=64,
        temperature=0, ignore_eos=True, logprobs=5, return_token_ids=True,
        cache_salt=f'overlap-qualification-{i}')
    result = s.post(url + '/v1/completions', json=payload, timeout=180)
    result.raise_for_status()
    data = result.json()
    assert data['usage']['prompt_tokens'] == length
    assert data['usage']['completion_tokens'] == 64
    receipts.append(dict(case=i, prompt_tokens=length, response=data))
    (a.output / f'case{i}.json').write_text(json.dumps(receipts[-1]))
    print('QUALIFIED_REQUEST', i, length, flush=True)

if a.profile:
    response = s.post(url + '/start_profile', timeout=30)
    response.raise_for_status()
    prompts = [tokenizer.encode(f'Continue counting from {i}: 1, 2, 3,', add_special_tokens=False)
               for i in range(16)]
    result = s.post(url + '/v1/completions', json=dict(model='qwen35-overlap',
        prompt=prompts, temperature=0, max_tokens=256, ignore_eos=True,
        return_token_ids=True, cache_salt='overlap-profile'), timeout=180)
    result.raise_for_status()
    data = result.json()
    assert len(data['choices']) == 16 and data['usage']['completion_tokens'] == 4096
    (a.output / 'profile-batch.json').write_text(json.dumps(data))
    response = s.post(url + '/stop_profile', timeout=120)
    response.raise_for_status()
metrics = s.get(url + '/metrics', timeout=15)
metrics.raise_for_status()
(a.output / 'metrics.txt').write_text(metrics.text)
(a.output / 'complete.json').write_text(json.dumps(dict(requests=3, profile=a.profile,
    scope='HTTP budget checks; compare cases across arms separately for numerical parity')))
print('QUALIFICATION_DONE', a.output, flush=True)
