"""Rank-local replay seams; uncovered TASK time is not proven recoverable idle."""
import argparse
import json
from pathlib import Path
import sqlite3


def union_ns(intervals):
    total=0;end=None
    for left,right in sorted(intervals):
        total += max(0,right-max(left,end if end is not None else left))
        end=max(right,end if end is not None else right)
    return total


def inspect(db):
    summary=json.loads(db.with_name(db.stem+'-summary.json').read_text())
    graphs=summary['graphs'];rows=[]
    with sqlite3.connect(f'file:{db.resolve()}?mode=ro',uri=True) as c:
        for a,b in zip(graphs,graphs[1:]):
            if a['evidence']!='exact_direct' or b['evidence']!='exact_direct':continue
            start=int(a['start_ns']+a['duration_us']*1000);end=b['start_ns']
            if end<=start:continue
            tasks=c.execute("select max(start_ns,?),min(end_ns,?) from traceloom_event where source_table='TASK' and start_ns<? and end_ns>?",(start,end,end,start)).fetchall()
            # EVENT_WAIT/NOTIFY_WAIT are TASK rows too, not productive execution.
            work=c.execute("select max(start_ns,?),min(end_ns,?) from traceloom_event where source_table='TASK' and start_ns<? and end_ns>? and (task_type like 'KERNEL%' or task_type like '%MEMCPY%')",(start,end,end,start)).fetchall()
            api=c.execute('''select v.value,count(*),sum(min(a.endNs,?)-max(a.startNs,?))/1000.0
                from CANN_API a join STRING_IDS v on v.id=a.name where a.startNs<? and a.endNs>?
                group by v.value order by 3 desc''',(end,start,end,start)).fetchall()
            task_costs=c.execute('''select op_type,category,count(*),sum(min(end_ns,?)-max(start_ns,?))/1000.0
                from traceloom_event where source_table='TASK' and start_ns<? and end_ns>?
                group by op_type,category order by 4 desc''',(end,start,end,start)).fetchall()
            busy=union_ns(tasks)/1e6
            rows.append(dict(from_kind=a['kind'],to_kind=b['kind'],start_ns=start,end_ns=end,
                envelope_ms=(end-start)/1e6,device_task_union_ms=busy,
                kernel_memcpy_union_ms=union_ns(work)/1e6,
                no_TASK_coverage_ms=(end-start)/1e6-busy,apis=api,task_costs=task_costs))
    return dict(scope='Profiled clipped rank-local TASK union, not proof of pure idle or unprofiled recoverable time. API rows may nest: never sum them into an exclusive host ledger.',gaps=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('db',type=Path);a=p.parse_args()
    out=a.db.with_name(a.db.stem+'-gaps.json');out.write_text(json.dumps(inspect(a.db),indent=2));print(out)
