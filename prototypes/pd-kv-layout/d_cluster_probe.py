"""D-only balanced native DP3/TP2/EP6, target-only; no donor/runtime edits."""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path

async def main(output, batches, profile_batch):
    from pd_model_probe import prepare_worker, engine_options
    prepare_worker("D", native_async=True)
    options = engine_options(False)
    options.update(worker_cls="d_cluster_profile_entry.Worker",
                   data_parallel_size=3, data_parallel_size_local=3,
                   data_parallel_address="127.0.0.1", data_parallel_rpc_port=29673,
                   disable_log_stats=True)
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    model = AsyncLLM.from_engine_args(AsyncEngineArgs(**options))
    try:
        tokenizer = model.get_tokenizer()
        unit = tokenizer.encode("Explain why a distributed system needs clear ownership and completion fences. ", add_special_tokens=False)
        prompt = (unit * (1024 // len(unit) + 1))[:1024]
        async def request(label, owner, index, n):
            start = time.perf_counter_ns()
            arrivals, tokens = [], []
            async for item in model.generate(
                dict(prompt_token_ids=prompt, cache_salt=f"{label}-{owner}-{index}"),
                SamplingParams(temperature=0, max_tokens=n, ignore_eos=True,
                               output_kind=RequestOutputKind.DELTA),
                f"{label}-{owner}-{index}", data_parallel_rank=owner):
                ids = list(item.outputs[0].token_ids)
                if ids:
                    tokens.extend(ids)
                    arrivals.append([len(tokens), time.perf_counter_ns()])
            if len(tokens) != n:
                raise RuntimeError(f"Wrong output length: {len(tokens)} != {n}")
            return dict(owner=owner, index=index, start_ns=start, arrivals=arrivals, tokens=tokens)
        # Compile/runtime warmup is outside all measured cohorts.
        await asyncio.gather(*(request("warmup", o, 0, 32) for o in range(3)))
        await model.wait_for_requests_to_drain()
        reports = []
        cohorts = [(f"b{batch}", batch, False) for batch in batches]
        if profile_batch:
            cohorts.append((f"b{profile_batch}-profile", profile_batch, True))
        for label, batch, capture in cohorts:
            armed = await model.collective_rpc("d_probe_arm", args=(str(output),label,batch,capture))
            begin = time.perf_counter_ns()
            requests = await asyncio.gather(*(request(label,o,i,128)
                for o in range(3) for i in range(batch)))
            end = time.perf_counter_ns()
            await model.wait_for_requests_to_drain()
            receipt = await model.collective_rpc("d_probe_finish")
            report = dict(label=label,batch_per_owner=batch,capture=capture,
                          prompt_tokens=1024,output_tokens=128,begin_ns=begin,end_ns=end,
                          armed=armed,workers=receipt,requests=requests)
            (output / f"{label}-requests.json").write_text(json.dumps(report,indent=2))
            reports.append(dict(label=label,seconds=(end-begin)/1e9,requests=len(requests)))
            print("D_COHORT",json.dumps(reports[-1]),flush=True)
        (output/"complete.json").write_text(json.dumps(reports,indent=2))
    finally:
        model.shutdown()

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--batches",type=int,nargs="+",choices=range(1,17),default=[1,8,16])
    parser.add_argument("--profile-batch",type=int,choices=range(17),default=8)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    asyncio.run(main(args.output,args.batches,args.profile_batch))
