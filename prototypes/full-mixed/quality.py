"""Native execution of retained OpenCompass inputs; no synthetic token lengths."""
import json
import time
from pathlib import Path


def run(llm, output, source):
    from vllm import SamplingParams
    requests = json.loads(source.read_text())
    assert len(requests) == 32 and len({r['request_id'] for r in requests}) == 32
    (output/'quality-inputs.json').write_text(json.dumps(requests))
    capacity = llm.collective_rpc('quality_capacity')
    (output/'quality-capacity.json').write_text(json.dumps(capacity,indent=2))
    assert all(r['max_length_concurrency'] >= 4 for r in capacity), 'KV capacity cannot hold four full-length quality requests'
    results = []
    started = time.monotonic()
    for offset in range(0,len(requests),4):
        batch = requests[offset:offset+4]
        prompts = [dict(prompt_token_ids=r['prompt_token_ids']) for r in batch]
        params = [SamplingParams(temperature=0,max_tokens=r['max_new_tokens'],
                                stop_token_ids=[r['eos_token_id']],ignore_eos=False,
                                detokenize=False) for r in batch]
        llm.collective_rpc('start_decode_observation',args=(False,f'quality{offset//4}'))
        outputs = llm.generate(prompts,params)
        llm.collective_rpc('stop_decode_observation')
        assert len(outputs) == len(batch)
        for request, actual in zip(batch,outputs):
            assert list(actual.prompt_token_ids) == request['prompt_token_ids']
            assert actual.finished and len(actual.outputs)==1
            result=actual.outputs[0]
            results.append(dict(request_id=request['request_id'],prompt_tokens=len(request['prompt_token_ids']),
                                token_ids=list(result.token_ids),finish_reason=result.finish_reason,
                                stop_reason=result.stop_reason))
        (output/'quality-partial.json').write_text(json.dumps(results,indent=2))
        print(f'quality completed {len(results)}/{len(requests)}',flush=True)
    receipts=llm.collective_rpc('graph_receipt')
    assert all(any(e['captured'] and e['replays']>0 and e['tokens']>24
                   for wrapper in rank['wrappers'] for e in wrapper['entries']) for rank in receipts)
    result=dict(status='COMPLETED_UNSCORED',scope='Retained 32 OpenCompass LongBench English retrieval items',
                elapsed_seconds=time.monotonic()-started,requests=results,receipts=receipts)
    (output/'quality-result.json').write_text(json.dumps(result,indent=2))
