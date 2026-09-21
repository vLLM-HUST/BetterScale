"""Cold/warm and divergent-continuation checks; never run during timing."""
import json
from pathlib import Path
import urllib.request


def verify(url, tokens, run, capsule):
    def reset():
        req = urllib.request.Request(url+'/reset_prefix_cache',data=b'',method='POST')
        with urllib.request.urlopen(req,timeout=60) as response:
            assert response.status == 200

    def hits():
        return [json.loads(line) for line in (Path(capsule)/'apc-hits.jsonl').read_text().splitlines()]

    results = []
    for boundary in (1536,3072):
        prompt = tokens[:boundary+9]
        changed = prompt.copy()
        changed[boundary] = next(t for t in tokens if t != changed[boundary])
        reset()
        start = len(hits())
        cold = run(prompt,32)
        warm = run(prompt,32)
        branch_warm = run(changed,32)
        reset()
        branch_cold = run(changed,32)
        observed = [row['hits'] for row in hits()[start:]]
        assert observed == [0,boundary,boundary-1536,0], observed
        result = dict(boundary=boundary,hits=observed,
                      warm_equal=cold['text']==warm['text'],
                      branch_equal=branch_cold['text']==branch_warm['text'])
        assert result['warm_equal'] and result['branch_equal'], result
        results.append(result)
    reset()
    run(tokens[:3073],1)
    return results
