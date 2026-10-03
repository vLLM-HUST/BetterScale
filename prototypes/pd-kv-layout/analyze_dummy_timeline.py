"""Eight-rank MTP dummy timeline: pair native target/draft device envelopes."""
import argparse
import gzip
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from analyze_d_timeline import union_ns,category

def analyze(root):
    exports=json.loads((root/"exports.json").read_text())
    summaries=[];events=[]
    for item in exports:
        rank=int(re.search(r"/rank(\d+)_",item["profile"])[1])
        with sqlite3.connect(f'file:{item["database"]}?mode=ro',uri=True) as c:
            assert c.execute("pragma quick_check").fetchone()[0]=="ok"
            devices=[r[0] for r in c.execute("select distinct deviceId from TASK")]
            assert devices==[rank],(rank,devices)
            rows=c.execute("""select t.startNs,t.endNs,s.value,n.value,t.streamId
                from TASK t join COMPUTE_TASK_INFO i using(globalTaskId)
                join STRING_IDS s on i.opType=s.id join STRING_IDS n on i.name=n.id
                where t.endNs>t.startNs order by t.startNs""").fetchall()
            apis=c.execute("""select a.startNs,a.endNs,s.value,a.globalTid
                from CANN_API a join STRING_IDS s on a.name=s.id
                where a.endNs>a.startNs""").fetchall()
            graphs=c.execute("""select a.connectionId,min(t.startNs),max(t.endNs),count(*)
                from CANN_API a join STRING_IDS s on a.name=s.id join TASK t using(connectionId)
                where s.value='aclmdlRIExecuteAsync' and t.endNs>t.startNs
                group by a.connectionId order by min(t.startNs)""").fetchall()
            copies=c.execute("""select t.startNs,t.endNs,s.value,t.streamId
                from TASK t join STRING_IDS s on t.taskType=s.id
                where s.value='MEMCPY_ASYNC' and t.endNs>t.startNs""").fetchall()
            # PMU repeats graph task IDs but carries no occurrence timestamp.
            # Aggregate the whole capture, joining the unique metadata only;
            # joining TASK here would multiply samples by replay occurrences.
            pmu=c.execute("""select op.value,n.value,count(*),avg(p.value),
                min(p.value),max(p.value) from TASK_PMU_INFO p
                join COMPUTE_TASK_INFO i using(globalTaskId)
                join STRING_IDS op on i.opType=op.id
                join STRING_IDS n on p.name=n.id
                where op.value in ('betterscale_context_parallel',
                    'GroupedMatmul','MatMulV2','fused_recurrent_gated_delta_rule_fwd_kernel')
                and n.value like '%ratio' group by op.value,n.value""").fetchall()
        assert len(graphs)==64,(rank,len(graphs))
        # Do not call each of64 replays a complete MTP cycle. Independent
        # operator census establishes target30 GDN layers vs merged draft0.
        for j,g in enumerate(graphs):
            count=sum("fused_recurrent_gated_delta_rule_fwd_kernel"==op
                      and g[1]<=a<g[2] for a,b,op,name,stream in rows)
            assert count==(30 if j%2==0 else 0),(rank,j,count)
        anchors=[g[1] for g in graphs[::2]]
        lo,hi=anchors[2],anchors[30];count=28
        groups=defaultdict(list);ops=defaultdict(list)
        for a,b,op,name,stream in rows:
            if a<hi and b>lo:
                pair=max(a,lo),min(b,hi)
                groups[category(op)].append(pair);ops[op].append(pair)
            events.append(dict(ph="X",name=op,cat=category(op),ts=a/1000,
                dur=(b-a)/1000,pid=rank,tid=f"NPU{rank} stream{stream}",args=dict(operator=name)))
        dma=[]
        for a,b,op,stream in copies:
            if a<hi and b>lo:dma.append((max(a,lo),min(b,hi)))
            events.append(dict(ph="X",name=op,cat="DMA",ts=a/1000,dur=(b-a)/1000,pid=rank,tid=f"NPU{rank} stream{stream}"))
        for a,b,name,tid in apis:
            events.append(dict(ph="X",name=name,cat="CANN_API",ts=a/1000,
                dur=(b-a)/1000,pid=rank,tid=f"host {tid}"))
        for j,(a,b) in enumerate(zip(anchors,anchors[1:])):
            events.append(dict(ph="X",name=f"MTP dummy cycle {j}",cat="cycle",
                ts=a/1000,dur=(b-a)/1000,pid=rank,tid="target-to-target"))
        compute=[v for k,vs in groups.items() if k!="communication" for v in vs]
        comm=groups["communication"];covered=union_ns(compute+comm+dma)
        summaries.append(dict(rank=rank,device=devices[0],replays=len(graphs),
            interior_cycles=count,mean_cycle_ms=(hi-lo)/1e6/count,
            median_cycle_ms=statistics.median([(b-a)/1e6 for a,b in zip(anchors[2:30],anchors[3:31])]),
            compute_union_ms=union_ns(compute)/1e6/count,
            communication_union_ms=union_ns(comm)/1e6/count,
            compute_comm_overlap_ms=(union_ns(compute)+union_ns(comm)-union_ns(compute+comm))/1e6/count,
            memcpy_union_ms=union_ns(dma)/1e6/count,
            uncovered_ms=((hi-lo)-covered)/1e6/count,
            categories_union_ms={k:union_ns(v)/1e6/count for k,v in groups.items()},
            target_envelope_ms=statistics.mean([(g[2]-g[1])/1e6 for g in graphs[4:60:2]]),
            draft_envelope_ms=statistics.mean([(g[2]-g[1])/1e6 for g in graphs[5:61:2]]),
            pmu_whole_capture=[dict(op=op,metric=metric,samples=n,mean=mean,min=low,max=high)
                               for op,metric,n,mean,low,high in pmu],
            attention_phase_ms={phase:sum(b-a for a,b,op,name,stream in rows
                if op=="betterscale_context_parallel" and any(g[1]<=a<g[2] for g in graphs[first:60:2]))/1e6/count
                for phase,first in (("target",4),("draft",5))},
            top_ops=sorted([dict(op=k,count=len(v),sum_ms=sum(b-a for a,b in v)/1e6/count,
                               union_ms=union_ns(v)/1e6/count) for k,v in ops.items()],
                          key=lambda v:v["sum_ms"],reverse=True)[:24]))
    assert sorted(r["rank"] for r in summaries)==list(range(8))
    summaries.sort(key=lambda r:r["rank"])
    result=dict(scope="Profiled synthetic target+MTP dummy,32 cycles;28 interior device target-to-target cycles. Compute/HCCL/DMA interval unions, not summed stream time or FLOP utilization. HCCL includes waiting. PMU ratios are unweighted whole-capture samples (not interior-window bandwidth). No DRAM traffic counters; no proof of bandwidth saturation. Separate unprofiled baseline is dummy-kv80-v5.",ranks=summaries)
    (root/"timeline-summary.json").write_text(json.dumps(result,indent=2))
    origin=min(e["ts"] for e in events)
    for e in events:e["ts"]-=origin
    for rank in range(8):
        events.append(dict(ph="M",name="process_name",pid=rank,args=dict(name=f"DP{rank//2} TP{rank%2} / NPU{rank}")))
    with gzip.open(root/"d8-mtp-dummy-timeline.json.gz","wt") as f:
        json.dump(dict(traceEvents=events,displayTimeUnit="ms"),f,separators=(",",":"))
    print(json.dumps([dict(rank=r["rank"],cycle=r["mean_cycle_ms"],compute=r["compute_union_ms"],
        comm=r["communication_union_ms"],dma=r["memcpy_union_ms"],uncovered=r["uncovered_ms"],
        categories=r["categories_union_ms"],target=r["target_envelope_ms"],draft=r["draft_envelope_ms"])
        for r in summaries],indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path)
    analyze(p.parse_args().root)
