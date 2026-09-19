"""Export a bounded MC2 routing profile; leave numerical timings unprofiled."""
import argparse
import collections
import json
from pathlib import Path
import sqlite3
import subprocess

p = argparse.ArgumentParser()
p.add_argument('round', type=Path)
p.add_argument('--traceloom', required=True)
a = p.parse_args()
out = a.round / 'traceloom'
out.mkdir(exist_ok=True)
result = []
for rank in (0, 1):
    sources = list((a.round / 'profiles/profile').glob(f'rank{rank}_*/PROF_*'))
    assert len(sources) == 1
    source = sources[0]
    db = out / f'rank{rank}.db'
    if not db.exists():
        with (out / f'rank{rank}.log').open('w') as log:
            subprocess.run(['msprof', '--export=on', '--type=db', f'--output={source}'],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=240)
            subprocess.run([a.traceloom, str(source), '--threads', '4', '--output', str(db),
                            '--perfetto-out', str(out / f'rank{rank}.perfetto.json.gz')],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as c:
        rows = c.execute('''select t.startNs,t.endNs,s.value from TASK t
            join COMPUTE_TASK_INFO i using(globalTaskId)
            join STRING_IDS s on s.id=i.opType order by t.startNs''').fetchall()
    firsts = [r for r in rows if r[2].split('_', 1)[0] == 'GemmaRmsNorm']
    samples = [r for r in rows if r[2] == 'ArgMaxV2']
    assert len(firsts) == len(samples) == 6, (len(firsts), len(samples))
    schedule = json.loads((a.round / f'profiles/schedule-rank{rank}.json').read_text())
    dispatch = [r for r in schedule['events'] if r['event'] == 'dispatch']
    assert len(dispatch) == 6
    steps = []
    for first, sample, wave in zip(firsts, samples, dispatch):
        body = [r for r in rows if first[0] <= r[0] < sample[0]]
        counts = collections.Counter(r[2].split('_', 1)[0] for r in body)
        decode = all(n == 1 and c >= p for n, c, p in zip(
            wave['scheduled'], wave['computed'], wave['prompt_tokens']))
        assert counts['MatmulAllReduce'] == (0 if decode else 128), counts
        assert counts['MatMulV2'] + counts['MatMulV3'] == (305 if decode else 177), counts
        assert counts['FusedInferAttentionScore'] == 16, counts
        steps.append(dict(dispatch=wave, counts=counts))
    result.append(dict(rank=rank, steps=steps))
(out / 'routing.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps([dict(rank=r['rank'], steps=[dict(scheduled=s['dispatch']['scheduled'],
    ops={k:v for k,v in s['counts'].items() if 'Mat' in k or 'Reduce' in k}) for s in r['steps']]) for r in result], indent=2))
