"""Unprofiled matched-case means; do not compare different mixed arrival shapes."""
import collections
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
comparison = json.loads((root / 'comparison.json').read_text())
assert comparison['status'] == 'PASS'
rounds = []
for index, arm in enumerate(('split', 'mc2', 'mc2', 'split')):
    d = json.loads((root / f'{index}-{arm}/receipt.json').read_text())
    assert d['status'] == 'PASS'
    groups = collections.defaultdict(list)
    for c in d['cohorts']:
        if c['warmup']:
            continue
        for rank in c['measurement']['results']:
            for s in rank['steps']:
                if not s.get('forward'):
                    continue
                decode = all(n == 1 and x >= p for n, x, p in zip(
                    s['scheduled'], s['computed'], s['prompt_tokens']))
                if c['case']['kind'] == 'decode' and decode and s['requests'] == c['case']['batch']:
                    key = ('decode', s['requests'], rank['rank'])
                elif c['case']['kind'] == 'prefill' and not decode:
                    key = ('prefill', s['tokens'], rank['rank'])
                else:
                    continue
                groups[key].append(s)
    rounds.append(dict(index=index, arm=arm, points=[dict(kind=k[0], tokens=k[1], rank=k[2],
        samples=len(v), forward_ms=statistics.mean(s['forward_ms'] for s in v),
        period_ms=statistics.mean(s['period_ms'] for s in v if 'period_ms' in s))
        for k,v in sorted(groups.items())]))
points = []
for point in rounds[0]['points']:
    key = {k:point[k] for k in ('kind','tokens','rank')}
    values = {arm:[p for r in rounds if r['arm']==arm for p in r['points']
                  if all(p[k]==v for k,v in key.items())] for arm in ('split','mc2')}
    assert all(len(v)==2 for v in values.values()), key
    means = {arm:{metric:statistics.mean(p[metric] for p in rows)
                  for metric in ('forward_ms','period_ms')} for arm,rows in values.items()}
    points.append(dict(**key, **means,
        forward_reduction_percent=100*(1-means['mc2']['forward_ms']/means['split']['forward_ms'])))
result=dict(scope='Same hw3 physical6/7, candidate-only ABBA. Four measured cohorts per case per round, cold APC, noMTP. Forward is event envelope, not bare graph body; period includes preparation/sampling. Mixed traffic exercises functionality but heterogeneous arrivals are not a matched speedup claim.', rounds=rounds, points=points)
(root/'timing-summary.json').write_text(json.dumps(result,indent=2)+'\n')
for p in points:
    if p['rank']==0: print(p)
