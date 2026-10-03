"""Bounded synthetic online PD diagnostic, not an official SWE benchmark."""
import argparse
import asyncio
import json
from pathlib import Path
import time
import uuid
from online_coordinator import Coordinator
from naive_pd_coordinator import owner_for


def distribution(values):
    values=sorted(values)
    if not values:return {}
    return {k:values[min(len(values)-1,int((len(values)-1)*q))]
            for k,q in (("p50",.5),("p95",.95),("p99",.99))}


async def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/"control-events.jsonl").open("w") as trace, (args.output/"results.jsonl").open("w") as results:
        def record(event):
            trace.write(json.dumps(event)+"\n");trace.flush()
        c=Coordinator(args.output/"directory.db",args.p,args.d,trace=record)
        await c.start()
        rows=[];label=uuid.uuid4().hex[:12];names=[];counts=[0]*4
        for i in range(10000):
            name=f"load-{label}-{i}";owner=owner_for(name)
            if counts[owner]<args.sessions//4:names.append(name);counts[owner]+=1
            if len(names)==args.sessions:break
        start=time.perf_counter()
        async def capture():
            await asyncio.sleep(args.profile_delay)
            peers=[(c.p,i) for i in range(4)]+[(c.d,0)]
            began=False
            try:
                # All native core utilities, not fabricated inference rounds.
                receipts=await asyncio.gather(*(peer.rpc(i,"profile",owner=0,start=True) for peer,i in peers))
                began=True
                (args.output/"profile-start.json").write_text(json.dumps(dict(seconds=time.perf_counter()-start,receipts=receipts)))
                await asyncio.sleep(args.profile_seconds)
            finally:
                if began:
                    receipts=await asyncio.gather(*(peer.rpc(i,"profile",owner=0,start=False) for peer,i in peers))
                    (args.output/"profile-stop.json").write_text(json.dumps(dict(seconds=time.perf_counter()-start,receipts=receipts)))
        profiler=asyncio.create_task(capture()) if args.profile_seconds else None
        async def session(index,name):
            await asyncio.sleep(index*args.stagger)
            length=(4096,32768)[index%2]
            prompt=[17]*(length-1)+[32+index]
            for turn in range(args.turns):
                issued=time.perf_counter()
                value=await c.submit(name,prompt,args.tokens)
                assert len(value["token_ids"])==args.tokens
                row=dict(index=index,turn=turn,issued=issued-start,completed=time.perf_counter()-start,
                         prompt_tokens=len(prompt),**value)
                results.write(json.dumps(row)+"\n");results.flush();rows.append(row)
                print(json.dumps(dict(event="turn",index=index,turn=turn,seconds=value["seconds"])),flush=True)
                prompt=value["full_tokens"]+[17]*128
        try:
            await asyncio.gather(*(session(i,name) for i,name in enumerate(names)))
            elapsed=time.perf_counter()-start
            if profiler:await profiler
            snapshots=[await c.cache("D",i,{"kind":"snapshot"}) for i in range(4)]
            assert all(not s["pending"] and all(x["owner"] is None and x["io_owner"] is None for x in s["seats"]) for s in snapshots)
            with c.directory.transaction() as db:
                assert db.execute("SELECT COUNT(*) FROM sessions WHERE owner!=? OR active!=0",("P",)).fetchone()[0]==0
            intervals=[]
            for row in rows:
                a=row["trace"][-1]["arrivals"]
                intervals.extend((y[1]-x[1])/1e6 for x,y in zip(a,a[1:]) if y[0]-x[0]==1)
            summary=dict(ok=True,workload="synthetic-4k-32k-staggered-closed-loop",
                         sessions=args.sessions,turns=args.turns,output_tokens=args.tokens,
                         stagger=args.stagger,seconds=elapsed,requests=len(rows),
                         tokens_per_second=len(rows)*args.tokens/elapsed,
                         turn_seconds=distribution([r["seconds"] for r in rows]),
                         single_token_arrival_interval_ms=distribution(intervals),
                         arrival_interval_count=len(intervals),snapshots=snapshots,
                         note="Actor arrival intervals are not device step timings; no TTFT is inferred from final RPC.")
            (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
            print(json.dumps({k:v for k,v in summary.items() if k!="snapshots"}),flush=True)
        finally:
            if profiler and not profiler.done():
                profiler.cancel()
                await asyncio.gather(profiler,return_exceptions=True)
            await c.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p",default="http://10.244.1.16:55581")
    p.add_argument("--d",default="http://10.244.2.32:55586")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--sessions",type=int,choices=(4,8,16,32,64),default=16)
    p.add_argument("--turns",type=int,default=2)
    p.add_argument("--tokens",type=int,default=512)
    p.add_argument("--stagger",type=float,default=.3)
    p.add_argument("--profile-seconds",type=float,default=0)
    p.add_argument("--profile-delay",type=float,default=15)
    args=p.parse_args()
    if not 0<=args.profile_seconds<=30 or args.profile_delay<0 or args.turns<1 or args.tokens<2 or args.tokens>512 or args.stagger<0:p.error("Invalid workload bound")
    asyncio.run(run(args))
