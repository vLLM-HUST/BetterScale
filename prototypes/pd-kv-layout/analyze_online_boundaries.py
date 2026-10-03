"""Join D-host monotonic output receipts, without calling arrivals device cadence."""
import argparse
import collections
import json
from pathlib import Path
from online_workload import distribution


def records(path):
    with path.open() as stream:
        for line in stream:
            try:yield json.loads(line)
            except ValueError:continue  # Active buffered receipt file may have a partial last line.


def analyze(timing,workload):
    queued=collections.defaultdict(list)
    workers={}
    for path in timing.glob("core-*.jsonl"):
        for row in records(path):
            if row["stage"]=="core-enqueue":
                for request,count in row["requests"]:
                    if count:queued[request].append((count,row["ns"]))
    for path in timing.glob("worker-*.jsonl"):
        rows=list(records(path))
        workers[path.stem]=dict(count=len(rows),
            dispatch_ms=distribution([(b["begin_ns"]-a["begin_ns"])/1e6 for a,b in zip(rows,rows[1:])]),
            host_call_ms=distribution([(r["ns"]-r["begin_ns"])/1e6 for r in rows]))
    turns=[]
    for row in records(workload/"results.jsonl"):
        trace=row["trace"][-1];rid=trace["request_id"]
        # Native input processor appends an internal uniqueness suffix.
        keys=[key for key in queued if key==rid or key.startswith(rid+"-")]
        if len(keys)!=1:raise ValueError(f"Ambiguous or absent native request: {rid}: {keys}")
        enqueues={};count=0
        for n,ns in queued[keys[0]]:
            count+=n;enqueues[count]=ns
        matched=[(n,ns,enqueues[n]) for n,ns in trace["arrivals"] if n in enqueues]
        gaps=[(n,(ns-enqueue)/1e6) for n,ns,enqueue in matched]
        actor=trace["arrivals"]
        turn=dict(index=row["index"],turn=row["turn"],request_id=rid,
            prompt_tokens=row["prompt_tokens"],actor_arrivals=len(actor),matched=len(matched),
            core_to_actor_ms=distribution([v for _,v in gaps]),
            enqueue_interval_ms=distribution([(b[2]-a[2])/1e6 for a,b in zip(matched,matched[1:]) if b[0]-a[0]==1]),
            actor_interval_ms=distribution([(b[1]-a[1])/1e6 for a,b in zip(actor,actor[1:]) if b[0]-a[0]==1]))
        turns.append(turn)
    result=dict(scope="Same-host perf_counter_ns; worker dispatch is host timing, not device timing. Profiler windows perturb this run.",
                workers=workers,turns=turns)
    (workload/"boundary-summary.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(turns=len(turns),matched=sum(r["matched"] for r in turns),
        max_turn_p95_core_to_actor_ms=max((r["core_to_actor_ms"].get("p95",0) for r in turns),default=None))))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("timing",type=Path);p.add_argument("workload",type=Path)
    args=p.parse_args();analyze(args.timing,args.workload)
