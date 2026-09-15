"""Compact peak/shoulder ledger. Counter deltas are not operator allocation sizes."""
import argparse
import json
from pathlib import Path

GiB = 2**30


def summarize(path):
    ranks = []
    for file in sorted((path / 'memory').glob('phases-rank-*.json')):
        phases = json.loads(file.read_text())
        budget = next((r for r in phases if r['phase'] == 'budget'), {})
        ready = next((r for r in phases if r['phase'] == 'final-capture-and-draft:exit'), {})
        ranks.append(dict(rank=phases[0]['rank'],
            max_observed_allocated=max(r['peak'] for r in phases),
            max_observed_reserved=max(r['peak_reserved'] for r in phases),
            kv_budget=budget.get('kv_budget'), model_bytes=budget.get('model_bytes'),
            eager_activation=budget.get('activation'), non_torch=budget.get('non_torch'),
            ready_allocated=ready.get('allocated'),ready_reserved=ready.get('reserved'),
            phases=phases))
    traces = {}
    for file in sorted((path / 'memory').glob('*-rank-0.json')):
        if file.name.startswith('phases-'): continue
        records = json.loads(file.read_text())
        rises = [dict(r, record_index=i) for i,r in enumerate(records) if r['new_peak']]
        shoulders = sorted(enumerate(records),key=lambda p:p[1]['allocated'],reverse=True)[:16]
        traces[file.stem] = dict(records=len(records),high_water_marks=rises,
            largest_sampled_live=[dict(r,record_index=i) for i,r in shoulders])
    return dict(scope='Dummy, diagnostic counters; no latency or numerical claim',
                ranks=ranks,rank0_traces=traces)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    result=summarize(a.run)
    (a.run/'memory-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in result['ranks']:
        print('rank',r['rank'],{k:round(r[k]/GiB,4) for k in ('kv_budget','model_bytes','eager_activation','non_torch','ready_allocated','ready_reserved') if r.get(k) is not None})
    for phase,t in result['rank0_traces'].items():
        print(phase,'records',t['records'],'watermarks',len(t['high_water_marks']))
        for r in t['largest_sampled_live'][:6]:
            print(round(r['allocated']/GiB,4),round(r['reserved']/GiB,4),Path(r['file']).name,r['function'],r['previous_line'],'->',r['line'])
