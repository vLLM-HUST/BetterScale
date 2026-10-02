"""Bounded online P4/D4 lifecycle gate, with real incremental transfer receipts."""
import argparse
import asyncio
import json
from pathlib import Path
import time
import uuid
from online_coordinator import Coordinator
from naive_pd_coordinator import owner_for

async def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/"control-events.jsonl").open("w") as trace:
        def record(event):trace.write(json.dumps(event)+"\n");trace.flush()
        c=Coordinator(args.output/"directory.db",args.p,args.d,trace=record)
        await c.start()
        try:
            names=[];counts=[0]*4
            for i in range(10000):
                name=f"{args.label}-{i}";owner=owner_for(name)
                if counts[owner]<args.sessions//4:names.append(name);counts[owner]+=1
                if len(names)==args.sessions:break
            started=time.perf_counter()
            first=await asyncio.gather(*(c.submit(name,[17]*(args.prompt-1)+[32+i],args.tokens)
                                        for i,name in enumerate(names)))
            (args.output/"first.json").write_text(json.dumps(first,indent=2))
            assert all(len(v["token_ids"])==args.tokens for v in first)
            assert all(v["trace"][-1]["cached"]==args.prompt for v in first)
            second=await asyncio.gather(*(c.submit(v["session"],v["full_tokens"]+[17]*16,8) for v in first))
            assert all(len(v["token_ids"])==8 and v["trace"][0]["stage"]=="P-load" for v in second)
            if args.exact_context:
                v=second[0]
                assert len(v["full_tokens"])<262142
                edge=await c.submit(v["session"],v["full_tokens"]+[17]*(262142-len(v["full_tokens"])),2)
                assert len(edge["full_tokens"])==262144 and len(edge["token_ids"])==2
                (args.output/"exact-context.json").write_text(json.dumps(edge,indent=2))
            snapshots=[await c.cache("D",i,{"kind":"snapshot"}) for i in range(4)]
            assert all(not s["pending"] and all(x["owner"] is None and x["io_owner"] is None for x in s["seats"]) for s in snapshots)
            p_snapshots=[await c.cache("P",i,{"kind":"snapshot"}) for i in range(4)]
            assert all(not s["pending"] for s in p_snapshots)
            if args.require_concurrent:
                assert all(s["peak_pending"]>1 for s in snapshots+p_snapshots), "State concurrency not exercised"
            with c.directory.transaction() as db:
                assert db.execute("SELECT COUNT(*) FROM sessions WHERE owner!=? OR active!=0",("P",)).fetchone()[0]==0
            summary=dict(ok=True,seconds=time.perf_counter()-started,first=first,second=second,snapshots=snapshots,p_snapshots=p_snapshots)
            (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
            print(json.dumps({k:summary[k] for k in ("ok","seconds")}),flush=True)
        finally:await c.close()

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p",default="http://10.244.1.16:55581")
    p.add_argument("--d",default="http://10.244.2.32:55586")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--exact-context",action="store_true")
    p.add_argument("--require-concurrent",action="store_true")
    p.add_argument("--label",default="online-"+uuid.uuid4().hex[:12])
    p.add_argument("--sessions",type=int,choices=(4,8,16,32,64,128,192,256,320),default=4)
    p.add_argument("--prompt",type=int,default=1024)
    p.add_argument("--tokens",type=int,default=128)
    asyncio.run(run(p.parse_args()))
