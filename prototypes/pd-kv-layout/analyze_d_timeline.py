"""Inspect native msprof SQLite task intervals; do not sum overlapping streams."""
import argparse
import gzip
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from analyze_d_cluster import distribution

def union_ns(intervals):
    end=None;total=0
    for a,b in sorted(intervals):
        if b<=a:continue
        if end is None or a>end:
            total+=b-a;end=b
        elif b>end:
            total+=b-end;end=b
    return total

def category(op):
    if op.startswith("hcom_"):return "communication"
    if "Matmul" in op or "MatMul" in op:return "matmul"
    if "gated_delta" in op or op=="CausalConv1d":return "GDN_recurrent_conv"
    if op=="betterscale_context_parallel":return "attention"
    if op.startswith("Moe"):return "MoE_routing"
    return "other_compute"

def analyze(root):
    summaries=[];events=[]
    for rank in range(6):
        paths=list((root/"profile").glob(f"rank{rank}_*/PROF*/msprof*.db"))
        assert len(paths)==1,paths
        with sqlite3.connect(f"file:{paths[0]}?mode=ro",uri=True) as c:
            assert c.execute("pragma quick_check").fetchone()[0]=="ok"
            mapping=c.execute("select * from RANK_DEVICE_MAP").fetchall()
            devices={r[0] for r in c.execute("select distinct deviceId from TASK")}
            assert devices=={rank+2},(rank,devices)
            rows=c.execute("""select t.startNs,t.endNs,s.value,n.value,t.streamId
                from TASK t join COMPUTE_TASK_INFO i using(globalTaskId)
                join STRING_IDS s on i.opType=s.id join STRING_IDS n on i.name=n.id
                where t.endNs>t.startNs order by t.startNs""").fetchall()
            apis=c.execute("""select a.startNs,a.endNs,s.value,a.globalTid
                from CANN_API a join STRING_IDS s on a.name=s.id""").fetchall()
            anchors=[a for a,b,op,name,stream in rows if op=="_compute_slot_mapping_kernel"]
            replay=[a for a,b,name,tid in apis if name=="aclmdlRIExecuteAsync"]
            assert len(anchors)==len(replay)==16,(len(anchors),len(replay))
            assert sum(op=="betterscale_context_parallel" for a,b,op,n,s in rows)==160
            assert sum(op=="GroupedMatmul" for a,b,op,n,s in rows)==1280
            # Keep 12 complete interior device cycles; discard capture edges.
            lo,hi=anchors[2],anchors[14]
            by_op=defaultdict(list);groups=defaultdict(list)
            for a,b,op,name,stream in rows:
                if b>lo and a<hi:
                    pair=(max(a,lo),min(b,hi));by_op[op].append(pair)
                    groups[category(op)].append(pair)
                events.append(dict(name=op,cat=category(op),ph="X",ts=a/1000,
                    dur=(b-a)/1000,pid=rank,tid=f"NPU{rank+2} stream{stream}",
                    args=dict(operator=name)))
            for a,b,name,tid in apis:
                if b>a:
                    events.append(dict(name=name,cat="CANN_API",ph="X",
                        ts=a/1000,dur=(b-a)/1000,pid=rank,tid=f"host {tid}"))
            periods=[(b-a)/1e6 for a,b in zip(anchors[2:14],anchors[3:15])]
            compute=[p for k,v in groups.items() if k!="communication" for p in v]
            comm=groups["communication"]
            combined=union_ns(compute+comm)
            window=hi-lo
            ops=sorted([dict(op=k,count=len(v),sum_ms=sum(b-a for a,b in v)/1e6/12)
                        for k,v in by_op.items()],key=lambda r:r["sum_ms"],reverse=True)
            summary=dict(rank=rank,device=rank+2,native_rank_device_map=mapping,
                interior_cycles=12,device_cycle_ms=distribution(periods),
                mean_cycle_ms=window/1e6/12,
                compute_union_ms=union_ns(compute)/1e6/12,
                communication_union_ms=union_ns(comm)/1e6/12,
                compute_communication_overlap_ms=(union_ns(compute)+union_ns(comm)-combined)/1e6/12,
                uncovered_ms=(window-combined)/1e6/12,
                categories_union_ms={k:union_ns(v)/1e6/12 for k,v in groups.items()},
                top_ops=ops[:20])
            summaries.append(summary)
    result=dict(scope="Native CANN/msprof, 16 graph replays/rank, interior12 slot-mapping-to-slot-mapping "
                "device cycles. Task interval coverage is not achieved FLOP utilization. Communication "
                "includes waits; uncovered time has no recorded compute/HCCL kernel, not proven host overhead. "
                "Native rankId=-1; rank identity comes from worker EP rank and is checked against device2..7.",
                ranks=summaries)
    (root/"timeline-summary.json").write_text(json.dumps(result,indent=2))
    # Subtract one shared origin to preserve cross-rank alignment and precision.
    origin=min(e["ts"] for e in events)
    for e in events:e["ts"]-=origin
    with gzip.open(root/"d6-msprof-timeline.json.gz","wt") as f:
        json.dump(dict(traceEvents=events,displayTimeUnit="ms"),f)
    for r in summaries:
        print(json.dumps({k:v for k,v in r.items() if k not in ("top_ops","categories_union_ms")}))
    print("rank0 categories",summaries[0]["categories_union_ms"])
    print("rank0 top",summaries[0]["top_ops"][:10])

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path)
    analyze(p.parse_args().root)
