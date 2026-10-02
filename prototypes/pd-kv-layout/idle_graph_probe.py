"""D8-only skew gate: parked State equality and native graph replay receipts."""
import argparse
import asyncio
import json
from pathlib import Path
import time


async def run(output):
    from online_actor import options
    opts,_=options("D",0)
    opts["worker_cls"]="idle_guard_entry.Worker"
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    model=AsyncLLM.from_engine_args(AsyncEngineArgs(**opts))
    async def request(owner,label,n):
        arrivals=[];last=None
        async for item in model.generate(dict(prompt_token_ids=[17]*1023+[32+owner],cache_salt=label),
            SamplingParams(temperature=0,max_tokens=n,ignore_eos=True,output_kind=RequestOutputKind.CUMULATIVE),
            label,data_parallel_rank=owner):
            last=item;arrivals.append((len(item.outputs[0].token_ids),time.perf_counter_ns()))
        assert len(last.outputs[0].token_ids)==n
        return dict(owner=owner,arrivals=arrivals,ids=list(last.outputs[0].token_ids))
    try:
        await asyncio.gather(*(request(o,f"park-{o}",16) for o in range(4)))
        await model.wait_for_requests_to_drain()
        armed=await model.collective_rpc("idle_guard_arm",args=(0,))
        await model.start_profile("idle-skew")
        result=await request(0,"active-skew",128)
        await model.wait_for_requests_to_drain()
        await model.stop_profile()
        checked=await model.collective_rpc("idle_guard_check")
        recovered=await asyncio.gather(*(request(o,f"return-{o}",32) for o in range(4)))
        await model.wait_for_requests_to_drain()
        (output/"summary.json").write_text(json.dumps(dict(ok=True,armed=armed,checked=checked,
            single_owner=result,recovered=recovered),indent=2))
        print("IDLE_GRAPH_GATE_PASS",flush=True)
    finally:model.shutdown()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    asyncio.run(run(args.output))
