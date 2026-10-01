"""Prefill a complete64-request cohort, then qualify one capacity-admitted D wave.

The first-wave barrier is a probe device, NOT the production admission policy.
State readback was qualified separately; this gate exercises the configured
cohort and checks released page counts without repeating every large oracle.
"""
import argparse
import asyncio
import json
from pathlib import Path
import time

from naive_pd_coordinator import Coordinator,owner_for


async def run(args):
    def trace(value):
        with (args.output/"transport-events.jsonl").open("a") as stream:
            stream.write(json.dumps(value)+"\n")
    c=Coordinator(args.output/"directory.db",args.p,args.d,max_batch=64,trace=trace)
    gate=asyncio.Event();decode=c._decode
    async def admitted_cohort():
        await gate.wait()
        await decode()
    c._decode=admitted_cohort
    await c.start()
    tasks=[];started=time.perf_counter()
    try:
        names=[];counts=[0]*4
        for i in range(10000):
            name=f"cohort64-{i}";owner=owner_for(name)
            if counts[owner]<16:
                names.append(name);counts[owner]+=1
            if len(names)==64:break
        tasks=[asyncio.create_task(c.submit(name,[17]*1024+[20+i],16))
               for i,name in enumerate(names)]
        deadline=time.monotonic()+900
        while c.ready.qsize()!=64:
            if c.failure:raise RuntimeError(c.failure)
            if time.monotonic()>deadline:raise TimeoutError("P cohort admission deadline")
            await asyncio.sleep(.1)
        gate.set()
        results=await asyncio.gather(*tasks)
        assert all(len(r["token_ids"])==16 and r["trace"][-1]["wave_size"]==64
                   and r["trace"][-1]["cached"]==1025 for r in results)
        after=await c.d.rpc(0,"capacity")
        assert after==c.capacities,"D pool did not release the imported cohort"
        selected={}
        for r in results:selected.setdefault(r["owner"],r)
        for r in selected.values():
            key=c.directory.current(r["session"])
            assert await c.p.get(key)==await c.d.get(key)
        with c.directory.transaction() as db:
            assert db.execute("SELECT COUNT(*) FROM sessions WHERE active!=0 OR owner!='P'").fetchone()[0]==0
        report=dict(seconds=time.perf_counter()-started,capacities=after,requests=results,
                    scope="64 submitted in one D wave,16 per owner; not a simultaneous-row timeline proof")
        (args.output/"summary.json").write_text(json.dumps(report,indent=2))
        print(json.dumps(dict(ok=True,requests=len(results),seconds=report["seconds"])),flush=True)
    finally:
        await c.close()
        await asyncio.gather(*tasks,return_exceptions=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p",required=True);parser.add_argument("--d",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    asyncio.run(run(args))
