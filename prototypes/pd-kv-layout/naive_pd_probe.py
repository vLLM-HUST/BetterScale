"""Bounded two-host P4/D4 integration; no MTP or production-load claim."""
import argparse
import asyncio
import json
from pathlib import Path
import time
from naive_pd_coordinator import Coordinator,owner_for
from pd_limits import context_limit


async def run(args):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    seed=tokenizer.encode("Explain how immutable checkpoints preserve a conversation across prefill and decode workers. ",add_special_tokens=False)
    prompt=(seed*(args.prompt_tokens//len(seed)+1))[:args.prompt_tokens]
    if args.prompt_tokens+1+16+len(seed)+8>context_limit():
        raise ValueError("Both turns must fit the configured context")
    names=[];counts=[0]*4
    for i in range(10000):
        name=f"{args.prefix}-{i}";owner=owner_for(name)
        if counts[owner]<args.sessions//4:
            names.append(name);counts[owner]+=1
        if len(names)==args.sessions:break
    assert {owner_for(s) for s in names}==set(range(4))
    coordinator=await Coordinator(args.output/"directory.db",args.p,args.d,max_batch=args.sessions,verify_imports=True).start()
    receipts=[]
    started=time.perf_counter()
    try:
        first=await asyncio.gather(*(coordinator.submit(s,prompt+[20+i],16) for i,s in enumerate(names)))
        receipts.append(dict(turn=1,requests=first))
        (args.output/"turn1.json").write_text(json.dumps(first,indent=2))
        second=await asyncio.gather(*(coordinator.submit(s,r["full_tokens"]+seed,8)
                                     for s,r in zip(names,first)))
        receipts.append(dict(turn=2,requests=second))
        (args.output/"turn2.json").write_text(json.dumps(second,indent=2))
        one=await coordinator.submit(args.prefix+"-single-token",prompt,1)
        receipts.append(dict(turn="single-token",requests=[one]))
        assert {r["trace"][0]["instance"] for r in first}==set(range(4))
        for previous,current in zip(first,second):
            assert len(previous["token_ids"])==16 and len(current["token_ids"])==8
            assert previous["owner"]==current["owner"]
            assert current["trace"][0]["restored"]
            assert current["trace"][0]["cached"]>0
            assert all(r["trace"][-1]["cached"]>0 for r in (previous,current))
        for result in second+[one]:
            key=coordinator.directory.current(result["session"])
            assert await coordinator.p.get(key)==await coordinator.d.get(key)
        with coordinator.directory.transaction() as db:
            rows=db.execute("SELECT id,epoch,owner,active,manifest FROM sessions ORDER BY id").fetchall()
            assert all(row[2]=="P" and row[3]==0 for row in rows)
        report=dict(scope="P4 TP2 / D DP4TP2EP8, target-only, naive full checkpoint; not task-quality qualification",
                    prompt_tokens=args.prompt_tokens,context_limit=context_limit(),
                    seconds=time.perf_counter()-started,receipts=receipts,directory=rows)
        (args.output/"summary.json").write_text(json.dumps(report,indent=2))
        print(json.dumps(dict(ok=True,seconds=report["seconds"],sessions=len(rows))),flush=True)
    finally:await coordinator.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p",required=True);parser.add_argument("--d",required=True)
    parser.add_argument("--model",default="/workspace/models/Qwen3.5-35B-A3B")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prefix",default="joint")
    parser.add_argument("--sessions",type=int,choices=(4,8,16,32,64),default=8)
    parser.add_argument("--prompt-tokens",type=int,default=1024)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    asyncio.run(run(args))


if __name__=="__main__":main()
