"""Rank-local TASK costs over the full captured session; no edge exclusion."""
import json, sqlite3, statistics, sys
from collections import defaultdict
from pathlib import Path

def union(xs):
    out=[]
    for a,b in sorted(xs):
        if out and a<=out[-1][1]: out[-1][1]=max(out[-1][1],b)
        else: out.append([a,b])
    return out

def length(xs): return sum(b-a for a,b in union(xs))

def analyze(db,pid):
    c=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
    assert c.execute('pragma integrity_check').fetchone()==('ok',)
    strings=dict(c.execute('select id,value from STRING_IDS'))
    tasks=c.execute('select startNs,endNs,globalPid,deviceId,globalTaskId,streamId,taskType from TASK').fetchall()
    assert tasks and {x[2] for x in tasks}=={pid},'unexpected PID ownership'
    assert len({x[3] for x in tasks})==1,'unexpected multi-device owner'
    meta={x[0]:(strings[x[1]],strings[x[2]]) for x in c.execute('select globalTaskId,name,opType from COMPUTE_TASK_INFO')}
    rows=[dict(start=x[0],end=x[1],stream=x[5],name=meta[x[4]][0],op=meta[x[4]][1]) for x in tasks if x[4] in meta]
    assert rows,'no compute tasks'
    collect=[x for x in rows if 'neural_collect_reduced' in x['name'] or 'neural_collect_reduced' in x['op']]
    bounds=c.execute('select startTimeNs,endTimeNs from SESSION_TIME_INFO').fetchall()
    assert len(bounds)==1
    left,right=bounds[0]; assert all(left<=x['start']<=x['end']<=right for x in rows)
    def intervals(xs): return [(x['start'],x['end']) for x in xs]
    def summary(xs):
        ds=sorted((x['end']-x['start'])/1e3 for x in xs)
        return dict(count=len(xs),union_ms=length(intervals(xs))/1e6,sum_ms=sum(ds)/1000,
                    p50_us=statistics.median(ds) if ds else None,p95_us=ds[round((len(ds)-1)*.95)] if ds else None)
    groups=defaultdict(list)
    for x in rows: groups[x['op'] if x['op']!='N/A' else x['name']].append(x)
    collect_ids={id(x) for x in collect}
    other=[x for x in rows if id(x) not in collect_ids]
    # Set difference via interval unions, not sum of overlapping operator times.
    total=length(intervals(rows)); cu=length(intervals(collect)); ou=length(intervals(other))
    pending={}; gaps=[]; collect_durations=[]
    for x in sorted(rows,key=lambda x:x['start']):
        if x['name']=='neural_publish':
            assert x['stream'] not in pending,'more than one publish before collect'
            pending[x['stream']]=x['end']
        elif x['name']=='neural_collect_reduced':
            assert x['stream'] in pending,'collect without publish'
            begin=pending.pop(x['stream']); assert begin<=x['start']
            gaps.append((x['start']-begin)/1000)
            collect_durations.append((x['end']-x['start'])/1000)
    assert not pending,'uncollected publication in captured window'
    publication=dict(pairs=len(gaps),publish_end_to_collect_start_p50_us=statistics.median(gaps) if gaps else None,
        publish_end_to_collect_start_sum_ms=sum(gaps)/1000,
        collect_sum_ms=sum(collect_durations)/1000,
        interpretation='Local work between publication and collect; opportunity to hide remote latency, NOT measured remote compute overlap.')
    return dict(publication=publication,database=str(db),pid=pid,device=tasks[0][3],integrity='ok',capture_ns=[left,right],
        capture_seconds=(right-left)/1e9,compute=summary(rows),collect=summary(collect),
        collect_capture_fraction=cu/(right-left),collect_compute_union_fraction=cu/total,
        collect_other_overlap_ms=(cu+ou-total)/1e6,
        hot_families=[dict(family=k,**summary(v)) for k,v in sorted(groups.items(),key=lambda kv:length(intervals(kv[1])),reverse=True)[:20]],
        nonclaims='Collect includes queue, compute, movement and copy; not network-only. Capture includes host gaps; no cold-start or SWE throughput claim. No cross-rank clock comparison or exact graph membership.')

if __name__=='__main__':
    root=Path(sys.argv[1]); receipt=json.loads((root/'receipt.json').read_text())
    result=[]
    for role in ('attention0','attention1'):
        dbs=list((root/'raw'/role).rglob('msprof*.db')); assert len(dbs)==1,dbs
        result.append(dict(role=role,**analyze(dbs[0],receipt['owners'][role])))
    (root/'costs.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ['role','capture_seconds','collect','collect_capture_fraction','collect_compute_union_fraction','hot_families']} for r in result],indent=2))
