"""Compare bounded real-model arms; never export a SWE leaderboard point."""
import argparse
import csv
import json
from pathlib import Path
import statistics

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('reference', type=Path, help='Server capsule; sibling -qualification/-measure')
p.add_argument('candidate', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()


def read(root, suffix, name):
    return json.loads(Path(str(root) + suffix, name).read_text())


def choices(root, suffix, name):
    data = read(root, suffix, name)
    return sorted(data.get('response', data)['choices'], key=lambda c: c['index'])


checks = []
for suffix, names in [('-qualification', ['case0.json', 'case1.json', 'case2.json',
                                         'profile-batch.json']),
                      ('-measure', [f'repeat{i}.json' for i in range(3)])]:
    for name in names:
        ref = choices(a.reference, suffix, name)
        got = choices(a.candidate, suffix, name)
        assert len(ref) == len(got)
        exact = [x['token_ids'] == y['token_ids'] for x, y in zip(ref, got)]
        deltas = [abs(v - w) for x, y in zip(ref, got)
                  if x.get('logprobs') and y.get('logprobs')
                  for v, w in zip(x['logprobs']['token_logprobs'],
                                  y['logprobs']['token_logprobs'])]
        checks.append(dict(file=suffix + '/' + name, rows=len(ref),
                           exact_rows=sum(exact), max_logprob_delta=max(deltas, default=None)))
        assert all(exact), checks[-1]


def profile(root):
    results = []
    for rank in range(2):
        paths = list((root / 'profile').glob(f'*rank{rank}_*/ASCEND_PROFILER_OUTPUT/kernel_details.csv'))
        assert len(paths) == 1
        rows = list(csv.DictReader(paths[0].open()))
        # Qwen's TP2 shared gate/up, down and scalar gate have unique shapes.
        # Select a single complete 40-layer/24-token graph occurrence, not the
        # five-window aggregate (which also includes prompt/startup work).
        shapes = ('"24,2048;512,2048"', '"24,256;2048,256"', '"24,2048;1,2048"')
        groups = {}
        for row in rows:
            groups.setdefault(row['Model ID'], []).append(row)
        candidates = [(gid, group) for gid, group in groups.items()
                      if sum(r['Name'].startswith('aclnnMatmul_') and r['Input Shapes'] == shapes[0]
                             for r in group) == 40]
        assert candidates
        def start(x): return float(x['Start Time(us)'])
        def end(x): return start(x) + float(x['Duration(us)'])
        for gid, group in candidates:
            shared = [r for r in group if r['Name'].startswith('aclnnMatmul_') and r['Input Shapes'] in shapes]
            gmm = [r for r in group if 'GroupedMatmul' in r['Name']]
            assert len(shared) == 120 and len(gmm) == 80
            overlap = {}
            for label, others in [('routed_gmm', gmm),
                ('collective', [r for r in group if r['Name'].startswith('hcom_')]),
                ('aiv', [r for r in group if r['Accelerator Core'] == 'AI_VECTOR_CORE'])]:
                overlap[label + '_intersection_us'] = sum(
                    max(0, min(end(x), end(y)) - max(start(x), start(y)))
                    for x in shared for y in others)
            results.append(dict(rank=rank, model_id=gid, shared_matmuls=120,
                routed_gmms=80, device_span_ms=(max(map(end, group)) - min(map(start, group))) / 1000,
                **overlap))
    return results


arms = []
for root in (a.reference, a.candidate):
    samples = read(root, '-measure', 'summary.json')['samples']
    arms.append(dict(capsule=str(root), samples=samples,
        median_output_tps=statistics.median(s['output_tps'] for s in samples),
        profile=profile(root)))
result = dict(scope='TP2/EP2 target-only, short-prompt C16; not SWE, MTP or multi-DP qualification',
    numerical_checks=checks, arms=arms,
    median_output_change_percent=(arms[1]['median_output_tps']/arms[0]['median_output_tps'] - 1)*100)
a.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(numerical_checks=checks,
                     median_output_change_percent=result['median_output_change_percent'])))
