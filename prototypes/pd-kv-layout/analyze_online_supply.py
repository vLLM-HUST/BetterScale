"""Host schedule/State-service observations, never a device-cadence estimator."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path


def quantiles(values):
    values=sorted(values)
    if not values:return dict(n=0)
    return dict(n=len(values),mean=sum(values)/len(values),p50=values[len(values)//2],
                p95=values[int((len(values)-1)*.95)],max=values[-1])


def records(path):
    # A live writer may leave an unfinished final line; complete corrupt records
    # must remain visible as errors rather than silently biasing measurements.
    with path.open() as stream:
        for line in stream:
            try:yield json.loads(line)
            except json.JSONDecodeError:
                if line.endswith("\n"):raise
                return


def analyze(timing,control,*,start=None,end=None):
    result=dict(scope="worker dispatch filtered by the explicit monotonic window; controller whole-file cohorts including drain. Host timing, NOT device occupancy/cadence",
                worker_window=dict(start=start,end=end),
                workers={},cache_rpc_seconds={},object_phase_seconds={})
    for path in sorted(timing.glob("worker-*.jsonl")):
        if not path.stem[len("worker-"):].isdigit():continue
        rows=Counter();decode=[];proposals=0;begins=[]
        for x in records(path):
            if x.get("stage")!="worker-dispatch":continue
            now=x["begin_ns"]/1e9
            if (start is not None and now<start) or (end is not None and now>=end):continue
            begins.append(x["begin_ns"])
            grants=x["scheduled"];drafts=x.get("drafts",{})
            rows[len(grants)]+=1
            proposals+=sum(drafts.values())
            if grants and all(n<=1+drafts.get(rid,0) for rid,n in grants.items()):
                decode.append(len(grants))
        result["workers"][path.stem]=dict(dispatched_rows=dict(sorted(rows.items())),
            decode_dispatches=len(decode),mean_decode_rows=sum(decode)/len(decode) if decode else None,
            scheduled_proposals=proposals,
            rows=quantiles([n for n,count in rows.items() for _ in range(count)]),
            host_dispatch_interval_ms=quantiles([(b-a)/1e6 for a,b in zip(begins,begins[1:])]))
    groups=defaultdict(list);transfer=defaultdict(list);seen=set()
    counts=Counter();request_phases=defaultdict(list);host_peaks=defaultdict(int)
    host_end={};p_waits=defaultdict(list);sender_tokens=sender_chunks=0
    if control:
        for x in records(control):
            op=x.get("op");counts[op]+=1
            if op in ("host-cache-admitted","host-cache-evicted","owner-admitted",
                      "P-first-token","D-queue-admitted"):
                request_phases[op].append(x["end"]-x["start"])
            if op=="ingress-admitted":
                for field in ("session_commit_wait","permit_wait"):
                    request_phases[field].append(x[field])
            if op=="host-cache-admitted":
                host_end[x["session"]]=x["end"]
                for owner,used in x["used"].items():host_peaks[owner]=max(host_peaks[owner],used)
            if op=="owner-admitted" and "p_instance" in x:
                prior=host_end.pop(x["session"],None)
                wait=x.get("p_permit_wait")
                if wait is None and prior is not None:wait=x["end"]-prior
                if wait is not None:p_waits[x["p_instance"]].append(wait)
            if op=="turn-committed" and start is not None and end is not None:
                for event in x["events"]:
                    if event["stage"]!="D":continue
                    previous=0
                    for count,now in event["arrivals"]:
                        delta=count-previous;previous=count
                        if start<=now/1e9<end:
                            sender_tokens+=delta;sender_chunks+=delta>0
            if op=="cache-control":
                groups[x["kind"]+"-"+x["action"]].append(x["end"]-x["start"])
                receipt=x.get("receipt") or {}
                identity=(x["kind"],x["index"],receipt.get("operation"))
                if x["action"]=="wait" and identity not in seen:
                    seen.add(identity)
                    for rank,values in receipt.get("transfer_phases_per_rank",{}).items():
                        for phase,seconds in values.items():
                            transfer[x["kind"]+"-"+receipt["kind"]+":"+phase].append(seconds)
    result["p_permit_wait_seconds_by_instance"]={k:quantiles(v) for k,v in p_waits.items()}
    if start is not None and end is not None:
        if end<=start:raise ValueError("positive measurement window required")
        result["sender_window"]=dict(accepted_tokens=sender_tokens,
            tokens_per_second=sender_tokens/(end-start),
            accepted_per_positive_chunk=sender_tokens/sender_chunks if sender_chunks else None,
            scope="D actor yield timestamps in worker window, not client goodput or device cadence; requires committed/drained cohort")
    result["control_counts"]=dict(counts)
    result["request_phase_seconds"]={k:quantiles(v) for k,v in request_phases.items()}
    result["host_reservation_peak_bytes"]=dict(host_peaks)
    result["cache_rpc_seconds"]={k:quantiles(v) for k,v in groups.items()}
    result["state_phase_host_seconds"]={k:quantiles(v) for k,v in transfer.items()}
    phases=defaultdict(list);sizes=Counter()
    for path in timing.glob("objects-*.jsonl"):
        for x in records(path):
            if x.get("stage")!="object-service" or not x["success"]:continue
            method=x["method"];sizes[method]+=x["bytes"]
            for (a,t0),(b,t1) in zip(x["points"],x["points"][1:]):
                phases[method+":"+a+"->"+b].append((t1-t0)/1e9)
    result["object_phase_seconds"]={k:quantiles(v) for k,v in phases.items()}
    result["object_bytes"]=dict(sizes)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timing",type=Path,required=True)
    p.add_argument("--control",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--measurement-seconds",type=float,
                   help="filter workers using first controller ingress as approximate window origin")
    p.add_argument("--last-seconds",type=float,default=60)
    args=p.parse_args();start=end=None
    if args.measurement_seconds is not None:
        if args.control is None or not 0<args.last_seconds<=args.measurement_seconds:
            p.error("a control trace and positive bounded worker window are required")
        origins=[]
        for row in records(args.control):
            if row.get("op")=="ingress-admitted":origins.append(row["start"])
        if not origins:p.error("no ingress clock anchor")
        end=min(origins)+args.measurement_seconds;start=end-args.last_seconds
    result=analyze(args.timing,args.control,start=start,end=end)
    result["clock_boundary"]="first controller ingress approximates client window origin; includes initial transport/parse delay"
    args.output.write_text(json.dumps(result,indent=2))
