"""One bounded all-eight-rank dummy KV-occupancy observation."""
import argparse
import asyncio
import json
from pathlib import Path

async def run(args):
    from online_actor import options
    opts,_=options("D",0)
    opts["worker_cls"]="dummy_decode_entry.Worker"
    opts["scheduler_cls"]="pool_state_entry.Scheduler"
    opts["additional_config"].update(state_cache_host_bytes=0,state_cache_control_rpc=False,state_cache_incremental=False,
                                     state_cache_policy=False)
    from betterscale.models import qwen35
    qwen35.STATE_SCHEDULER=opts["scheduler_cls"]
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    args.output.mkdir(parents=True,exist_ok=False)
    model=AsyncLLM.from_engine_args(AsyncEngineArgs(**opts))
    try:
        groups=await asyncio.gather(*(model.engine_core._call_utility_async(
            "collective_rpc","theory_decode",None,
            (args.context,args.fraction,args.steps,args.warmup,str(args.output/"profile") if args.profile else None),None,engine=e)
            for e in model.engine_core.core_engines))
        rows=[row for group in groups for row in group]
        if sorted(r["rank"] for r in rows)!=list(range(8)):raise RuntimeError("Missing ranks")
        (args.output/"result.json").write_text(json.dumps(rows,indent=2))
        print("DUMMY_COMPLETE",json.dumps([dict(rank=r["rank"],batch=r["batch"],
            occupied_fraction=r["occupied_fraction"],mean_ms=sum(r["step_ms"])/len(r["step_ms"])) for r in rows]),flush=True)
    finally:model.shutdown()

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--profile",action="store_true")
    p.add_argument("--context",type=int,default=100000)
    p.add_argument("--fraction",type=float,default=.8)
    p.add_argument("--steps",type=int,default=32)
    p.add_argument("--warmup",type=int,default=8)
    a=p.parse_args()
    if not 4096<=a.context<=200000 or not .1<=a.fraction<=.9 or not 8<=a.steps<=128 or not 2<=a.warmup<=32:
        raise ValueError("Unbounded dummy observation")
    asyncio.run(run(a))
