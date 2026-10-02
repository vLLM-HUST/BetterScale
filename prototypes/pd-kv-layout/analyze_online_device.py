"""Native device cycles/coverage from msprof DBs; communication includes waits."""
import argparse
import bisect
import collections
import json
import sqlite3
from pathlib import Path
from analyze_d_timeline import union_ns,category
from online_workload import distribution


def analyze(path):
    with sqlite3.connect(f"file:{path}?mode=ro",uri=True) as c:
        rows=c.execute("""select t.startNs,t.endNs,s.value,t.streamId
            from TASK t join COMPUTE_TASK_INFO i using(globalTaskId)
            join STRING_IDS s on i.opType=s.id where t.endNs>t.startNs order by t.startNs""").fetchall()
        replay=c.execute("""select count(*) from CANN_API a join STRING_IDS s on a.name=s.id
            where s.value='aclmdlRIExecuteAsync'""").fetchone()[0]
        device=[r[0] for r in c.execute("select distinct deviceId from TASK")]
        # The CANN replay correlation reaches device launch/envelope records
        # even when this owner is dummy and has no attention slot-map anchor.
        # Do not use host API start timestamps as device cadence.
        graph=c.execute("""
            select a.connectionId,min(t.startNs),max(t.endNs),count(*)
            from CANN_API a join STRING_IDS s on a.name=s.id
            join TASK t using(connectionId)
            where s.value='aclmdlRIExecuteAsync' and t.endNs>t.startNs
            group by a.connectionId order by min(t.startNs)
        """).fetchall()
    anchors=[a for a,b,op,stream in rows if op=="_compute_slot_mapping_kernel"]
    groups=collections.defaultdict(list)
    windows=[collections.defaultdict(list) for _ in anchors[:-1]]
    for a,b,op,stream in rows:
        i=max(0,bisect.bisect_right(anchors,a)-1)
        while i<len(windows) and anchors[i]<b:
            lo,hi=anchors[i:i+2]
            if a<hi and b>lo:windows[i][category(op)].append((max(lo,a),min(hi,b)))
            i+=1
    for (lo,hi),intervals in zip(zip(anchors,anchors[1:]),windows):
        if hi<=lo:continue
        compute=[v for k,vs in intervals.items() if k!="communication" for v in vs]
        comm=intervals["communication"];window=hi-lo
        groups["slow" if window>100e6 else "fast"].append(dict(
            cycle_ms=window/1e6,compute_ms=union_ns(compute)/1e6,
            communication_ms=union_ns(comm)/1e6,
            uncovered_ms=(window-union_ns(compute+comm))/1e6))
    graph_cycles=[(b[1]-a[1])/1e6 for a,b in zip(graph,graph[1:])]
    graph_summary=dict(correlated_replays=len(graph),
        device_records_per_replay=sorted({r[3] for r in graph}),
        cycle_ms=distribution(graph_cycles) if graph_cycles else {},
        envelope_ms=distribution([(r[2]-r[1])/1e6 for r in graph]) if graph else {},
        gap_ms=distribution([(b[1]-a[2])/1e6 for a,b in zip(graph,graph[1:])]) if graph_cycles else {},
        cycles_over100ms=sum(v>100 for v in graph_cycles),
        max_cycle_ms=max(graph_cycles,default=None),
        scope="Device envelope correlated to CANN replay; includes idle-owner participation. Gaps may include globally idle workload, not just scheduling overhead.")
    return dict(path=str(path),devices=device,replays=replay,anchors=len(anchors),graph_device=graph_summary,
        cycles={k:dict(count=len(v),**{field:distribution([r[field] for r in v]) for field in v[0]})
                for k,v in groups.items()},
        scope="Slot mapping anchors require a real attention request; dummy ranks have no such anchors. Coverage is interval union, not FLOP utilization. Communication includes waits; uncovered is not proved host overhead.")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("database",type=Path);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();result=analyze(a.database)
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
