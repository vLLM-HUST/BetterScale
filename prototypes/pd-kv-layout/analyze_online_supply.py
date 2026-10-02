"""Host schedule/State-service observations, never a device-cadence estimator."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path


def quantiles(values):
    values=sorted(values)
    if not values:return dict(n=0)
    return dict(n=len(values),p50=values[len(values)//2],
                p95=values[int((len(values)-1)*.95)],max=values[-1])


def analyze(timing,control):
    result=dict(scope="entire supplied files; host dispatch rows, not live device occupancy or cadence",
                workers={},cache_rpc_seconds={},object_phase_seconds={})
    for path in sorted(timing.glob("worker-*.jsonl")):
        rows=Counter();decode=[];proposals=0
        for line in path.open():
            x=json.loads(line)
            if x.get("stage")!="worker-dispatch":continue
            grants=x["scheduled"];drafts=x.get("drafts",{})
            rows[len(grants)]+=1
            proposals+=sum(drafts.values())
            if grants and all(n<=1+drafts.get(rid,0) for rid,n in grants.items()):
                decode.append(len(grants))
        result["workers"][path.stem]=dict(dispatched_rows=dict(sorted(rows.items())),
            decode_dispatches=len(decode),mean_decode_rows=sum(decode)/len(decode) if decode else None,
            scheduled_proposals=proposals)
    groups=defaultdict(list);transfer=defaultdict(list);seen=set()
    if control:
        for line in control.open():
            x=json.loads(line)
            if x.get("op")=="cache-control":
                groups[x["kind"]+"-"+x["action"]].append(x["end"]-x["start"])
                receipt=x.get("receipt") or {}
                identity=(x["kind"],x["index"],receipt.get("operation"))
                if x["action"]=="wait" and identity not in seen:
                    seen.add(identity)
                    for rank,values in receipt.get("transfer_phases_per_rank",{}).items():
                        for phase,seconds in values.items():
                            transfer[x["kind"]+"-"+receipt["kind"]+":"+phase].append(seconds)
    result["cache_rpc_seconds"]={k:quantiles(v) for k,v in groups.items()}
    result["state_phase_host_seconds"]={k:quantiles(v) for k,v in transfer.items()}
    phases=defaultdict(list);sizes=Counter()
    for path in timing.glob("objects-*.jsonl"):
        for line in path.open():
            x=json.loads(line)
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
    args=p.parse_args()
    args.output.write_text(json.dumps(analyze(args.timing,args.control),indent=2))
