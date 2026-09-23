"""Summarize probe waterlines and native pool ownership without double counting.

Usage: python summarize_memory.py RUN_ROOT > RUN_ROOT/summary.json
Raw bounded snapshots remain gzip artifacts, not model context or Git payloads.
"""
import argparse
import gzip
import json
from pathlib import Path


def pool_totals(segments):
    pools = {}
    for segment in segments:
        key = f"device{segment['device']}:pool{tuple(segment['segment_pool_id'])}"
        pool = pools.setdefault(key,dict(reserved=0,active=0,inactive=0,segments=0,streams=set()))
        pool['reserved'] += segment['total_size']
        pool['segments'] += 1
        pool['streams'].add(segment['stream'])
        total = 0
        for block in segment['blocks']:
            size,state = block['size'],block['state']
            total += size
            if state=='inactive':pool['inactive'] += size
            elif state in ('active_allocated','active_pending_free','active_awaiting_free'):
                pool['active'] += size
            else:raise ValueError(f'Unknown allocator state: {state}')
        if total!=segment['total_size']:
            raise ValueError('Allocator segment/block geometry does not close')
    for pool in pools.values():pool['streams']=sorted(pool['streams'])
    return pools


def summarize(root):
    result = {'scope':'matched memory diagnostic; not throughput or maximum KV capacity', 'arms':{}}
    for arm in ('native','full'):
        path=root/arm
        receipt=path/'receipt.json'
        if not receipt.exists():continue
        info=json.loads(receipt.read_text())
        record={k:info.get(k) for k in ('status','error','server_exit_code','semantic_failures','draft_tokens','accepted_tokens')}
        record['requests']=len(info.get('requests',[]))
        record['penalty_stress_requests']=len(info.get('penalty_stress',[]))
        record['ranks']={}
        for rank in sorted(path.glob('memory-rank*')):
            phases=[json.loads(x) for x in (rank/'startup-phases.jsonl').read_text().splitlines()]
            values=[x['after'] for x in phases if x['after']]
            checks={p.name.removesuffix('-waterline.json'):json.loads(p.read_text())
                    for p in sorted(rank.glob('*-waterline.json'))}
            values+=list(checks.values())
            peaks={field:max((v[field] for v in values),default=0)
                   for field in ('peak_allocated_bytes','peak_reserved_bytes')}
            pools={}
            for label in ('load','kv-initialize','worker-ready','http-ready','after-context-262080','after-c16-penalty'):
                p=rank/f'{label}-snapshot.json.gz'
                if p.exists():
                    with gzip.open(p,'rt') as f:snapshot=json.load(f)
                    pools[label]=pool_totals(snapshot['segments'])
            record['ranks'][rank.name]={'observed_peaks':peaks,'checkpoints':checks,
                'pool_totals':pools,'capture_phases':[p for p in phases if p['phase'].startswith('capture-')]}
        result['arms'][arm]=record
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root',type=Path)
    args=parser.parse_args()
    print(json.dumps(summarize(args.run_root),indent=2))
