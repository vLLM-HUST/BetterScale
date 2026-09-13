"""Small online admission/cancel/drain workload using the native LLM engine."""
import json
import time
from vllm import SamplingParams


def run(llm, output, budget):
    engine = llm.llm_engine
    started = time.monotonic()
    events, completed = [], {}

    def submit(rid, length, outputs):
        engine.add_request(rid, dict(prompt_token_ids=[17] * length),
                           SamplingParams(temperature=0, max_tokens=outputs,
                                          ignore_eos=True, detokenize=False))
        events.append(dict(event='submit', id=rid, length=length,
                           seconds=time.monotonic() - started))

    submit('long-decode', 64, 64)
    submit('cancel-in-flight', budget + 17, 64)
    injected = False
    steps = 0
    while engine.has_unfinished_requests():
        assert steps < 512, 'turnover failed to drain'
        results = engine.step()
        steps += 1
        for result in results:
            if result.finished:
                completed[result.request_id] = len(result.outputs[0].token_ids)
            events.append(dict(event='output', id=result.request_id,
                               tokens=len(result.outputs[0].token_ids), finished=result.finished,
                               seconds=time.monotonic() - started))
        if not injected and any(r.request_id == 'long-decode' for r in results):
            engine.abort_request(['cancel-in-flight'])
            events.append(dict(event='abort', id='cancel-in-flight',
                               seconds=time.monotonic() - started))
            submit('new-prefill', budget + 33, 16)
            submit('new-short', 17, 16)
            injected = True
    assert injected and completed.get('long-decode') == 64
    assert completed.get('new-prefill') == completed.get('new-short') == 16
    assert 'cancel-in-flight' not in completed
    abort_index = next(i for i, e in enumerate(events) if e['event'] == 'abort')
    assert not any(e['event'] == 'output' and e['id'] == 'cancel-in-flight'
                   for e in events[abort_index + 1:]), 'stale cancelled output escaped'
    (output / 'turnover.json').write_text(json.dumps(dict(
        elapsed=time.monotonic() - started, steps=steps, completed=completed,
        events=events), indent=2))
