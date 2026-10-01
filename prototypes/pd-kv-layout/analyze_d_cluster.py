"""Reduce observation receipts; host dispatch is not device execution duration."""
import argparse
import json
import statistics
from pathlib import Path

def distribution(values):
    values=sorted(values)
    if not values:
        return dict(n=0)
    def percentile(p):
        at=(len(values)-1)*p
        lo=int(at);hi=min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(at-lo)
    return dict(n=len(values),median=statistics.median(values),
                p10=percentile(.1),p90=percentile(.9),p99=percentile(.99))

def summarize(root):
    results=[]
    for label in ("b1","b8","b16","b8-profile"):
        ranks=[]
        for path in sorted(root.glob(label+"-rank*.json")):
            r=json.loads(path.read_text())
            rows=r["rows"]
            # Interior uninterrupted equal-batch decode only, excluding both
            # capture transitions; a mixed/prefill row breaks the adjacency.
            periods=[]
            duration=[]
            lo,hi=(26,37) if r["capture"] else (40,110)
            for a,b in zip(rows,rows[1:]):
                if (a["steady"] and b["steady"] and
                    lo <= a["decode_index"] < b["decode_index"] <= hi):
                    periods.append((b["begin_ns"]-a["begin_ns"])/1e6)
                    duration.append((a["end_ns"]-a["begin_ns"])/1e6)
            ranks.append(dict(rank=r["rank"],host_dispatch_period_ms=distribution(periods),
                              host_execute_call_ms=distribution(duration),memory=r["memory"]))
        request_path=root/(label+"-requests.json")
        if not request_path.exists():
            continue
        request=json.loads(request_path.read_text())
        itl=[]
        for r in request["requests"]:
            for a,b in zip(r["arrivals"],r["arrivals"][1:]):
                if 40<=a[0]<b[0]<=110 and b[0]-a[0]==1:
                    itl.append((b[1]-a[1])/1e6)
        results.append(dict(label=label,observed_ranks=len(ranks),ranks=ranks,client_single_token_itl_ms=(None if request["capture"] else distribution(itl)),
                            cohort_seconds=(request["end_ns"]-request["begin_ns"])/1e9,
                            request_count=len(request["requests"])))
    return dict(scope="1024 prompt / 128 target-only decode tokens; balanced native DP/TP2/EP; rank receipts determine topology. "
                "Host call duration is NOT device busy time; profiled and non-profiled arms separate.",
                cohorts=results)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path)
    a=p.parse_args();result=summarize(a.root)
    (a.root/"timing-summary.json").write_text(json.dumps(result,indent=2))
    for r in result["cohorts"]:
        print(r["label"], "ITL", r["client_single_token_itl_ms"])
        for rank in r["ranks"]:print(rank["rank"],rank["host_dispatch_period_ms"])
