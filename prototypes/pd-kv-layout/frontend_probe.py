"""Live loopback API checks: warm continuation, conflict and disconnect fencing."""
import argparse
import asyncio
import json
from pathlib import Path
import sqlite3
import time
import aiohttp


async def run(args):
    events=[];base=args.url.rstrip("/")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as client:
        async def post(session,prompt,n):
            response=await client.post(base+"/generate",json=dict(
                session=session,prompt_token_ids=prompt,max_tokens=n))
            data=await response.json() if response.status==200 else await response.text()
            return response.status,data
        response=await client.get(base+"/health");health=await response.json()
        assert response.status==200 and health["context_limit"]==262144
        first=asyncio.create_task(post("api-main",[17]*1024,16))
        deadline=time.monotonic()+30
        while True:
            response=await client.get(base+"/health")
            if (await response.json())["inflight"]:break
            if time.monotonic()>deadline:raise TimeoutError("HTTP request was not admitted")
            await asyncio.sleep(.05)
        status,_=await post("api-main",[17]*1024,2)
        assert status==409;events.append("overlap409")
        status,result=await first
        assert status==200 and len(result["token_ids"])==16
        status,warm=await post("api-main",result["full_tokens"]+[17],4)
        assert status==200 and warm["trace"][0]["restored"] and len(warm["token_ids"])==4
        events.append("warm-model-roundtrip")
        status,_=await post("too-long",[17]*262144,1)
        assert status==400;events.append("context400")
        response=await client.post(base+"/generate",json={"unsupported":"text"})
        assert response.status==400;events.append("schema400")

        # A disconnected HTTP caller does not make an in-flight remote writer
        # disappear. The controller must drain it before freeing its session.
        abandoned=asyncio.create_task(post("api-disconnect",[18]*1024,16))
        deadline=time.monotonic()+30
        while True:
            response=await client.get(base+"/health")
            if (await response.json())["inflight"]:break
            if time.monotonic()>deadline:raise TimeoutError("Disconnect case not admitted")
            await asyncio.sleep(.05)
        abandoned.cancel()
        await asyncio.gather(abandoned,return_exceptions=True)
        deadline=time.monotonic()+300
        while True:
            response=await client.get(base+"/health");state=await response.json()
            assert response.status==200 and not state["failed_closed"]
            if not state["inflight"]:break
            if time.monotonic()>deadline:raise TimeoutError("Disconnected request did not drain")
            await asyncio.sleep(.5)
        with sqlite3.connect("file:"+str(args.directory)+"?mode=ro",uri=True) as db:
            row=db.execute("SELECT owner,active FROM sessions WHERE id='api-disconnect'").fetchone()
            assert row==("P",0)
        events.append("disconnect-drained-and-published")
        args.output.write_text(json.dumps(dict(ok=True,events=events,warm=warm),indent=2))
        print(json.dumps(dict(ok=True,events=events)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url",default="http://127.0.0.1:55686")
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();asyncio.run(run(args))
