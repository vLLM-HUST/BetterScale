"""Retire one stopped frontend's idle cache, then prove native pools empty.

Never attach a fresh empty Directory to occupied native pools. This helper uses
only the existing frontend sessions.sqlite, its exclusive controller lock and
normal two-replica/TP retirement. It does not drop arbitrary worker objects.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sqlite3
from online_coordinator import Coordinator


async def retire(frontend, output, p_url, d_url):
    path=frontend/"sessions.sqlite"
    if not path.is_file():raise ValueError("existing frontend sessions.sqlite required")
    if output.exists():raise ValueError("never overwrite a retirement receipt")
    with sqlite3.connect(f"file:{path}?mode=ro",uri=True) as db:
        if db.execute("SELECT 1 FROM sessions WHERE active!=0 OR owner!='P' LIMIT 1").fetchone():
            raise ValueError("unfinished numerical ownership requires recovery")
    # Reconstruct no host ledger: this offline operation only retires known
    # checkpoints. start() still acquires the frontend's controller lock.
    os.environ["BETTERSCALE_PD_HOST_PRESSURE"]="0"
    c=Coordinator(path,p_url,d_url)
    try:
        await c.start()
        with c.directory.transaction() as db:
            names=[r[0] for r in db.execute("SELECT id FROM sessions WHERE manifest IS NOT NULL")]
        semaphore=asyncio.Semaphore(4)
        async def drop(name):
            async with semaphore:
                if not await c.evict_idle(name):raise RuntimeError("idle manifest disappeared")
        await asyncio.gather(*(drop(name) for name in names))
        memory=[]
        for peer,count in ((c.p,4),(c.d,1)):
            for instance in range(count):memory.extend(await peer.rpc(instance,"memory",owner=0))
        if len(memory)!=16 or any(
            r["pool"]["bytes"] or r["pool"]["checkpoints"] or r["host_arena"]["used_bytes"]
            for r in memory):
            raise RuntimeError("native cache residue; do not launch an empty frontend")
        output.write_text(json.dumps(dict(retired=len(names),memory=memory),indent=2))
        return len(names)
    finally:await c.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frontend",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--p",default="http://10.244.1.16:55581")
    p.add_argument("--d",default="http://10.244.2.32:55586")
    a=p.parse_args()
    print(json.dumps(dict(retired=asyncio.run(retire(a.frontend,a.output,a.p,a.d)))))
