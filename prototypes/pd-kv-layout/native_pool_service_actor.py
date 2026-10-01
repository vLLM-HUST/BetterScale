"""Trusted local actor for naive PD. Network transport must not use pickle.

Each actor admits one command at a time. D generate_batch uses native parallel
owner scheduling, then drains the DP wave before acknowledging the command.
No utility mutates a live Core or bypasses its idle gate.
"""
import asyncio
import time
import traceback
import uuid
from pd_limits import context_limit

async def run(kind,instance,connection):
    from pool_topology import engine_options
    options,layout=engine_options(kind,instance)
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from native_async_pool import owner_utility
    model=None
    try:
        model=AsyncLLM.from_engine_args(AsyncEngineArgs(**options))
        assert len(model.engine_core.core_engines)==layout["dp"]
        connection.send(("ready",dict(kind=kind,instance=instance,layout=layout,
                                        context_limit=options["max_model_len"],
                                        state_bytes=options["kv_cache_memory_bytes"],target_only=True)))
        while True:
            op,args=await asyncio.to_thread(connection.recv)
            if op=="stop":
                connection.send(("ok",None));break
            if op=="generate_batch":
                items=args["items"]
                counts=[0]*layout["dp"]
                if not 1<=len(items)<=16*layout["dp"]:
                    raise ValueError("Batch exceeds configured owner capacity")
                for item in items:
                    owner=item["owner"]
                    if type(owner) is not int or not 0<=owner<layout["dp"]:
                        raise ValueError("Invalid attention owner")
                    counts[owner]+=1
                    if (counts[owner]>16 or type(item["n"]) is not int
                            or not 1<=item["n"]<=512 or not item["tokens"]
                            or len(item["tokens"])+item["n"]>context_limit()):
                        raise ValueError("Unqualified request envelope")
                async def generate(item):
                    start=time.perf_counter()
                    final=None
                    async for value in model.generate(
                        dict(prompt_token_ids=item["tokens"],cache_salt=item["salt"]),
                        SamplingParams(temperature=0,max_tokens=item["n"],ignore_eos=True,
                                       output_kind=RequestOutputKind.FINAL_ONLY),
                        uuid.uuid4().hex,data_parallel_rank=item["owner"]):
                        final=value
                    assert final is not None and final.finished
                    ids=list(final.outputs[0].token_ids)
                    if len(ids)!=item["n"]:raise RuntimeError("Incomplete generation")
                    return dict(token_ids=ids,prompt_token_ids=final.prompt_token_ids,
                                cached=final.num_cached_tokens,text=final.outputs[0].text,
                                seconds=time.perf_counter()-start)
                result=await asyncio.gather(*(generate(item) for item in items))
                if kind=="D":await model.wait_for_requests_to_drain()
            elif op in ("export","import","drop"):
                owner=args["owner"]
                if op=="export":
                    result=await owner_utility(model.engine_core,owner,"pd_export_retired",
                                               args["tokens"],args["salt"],0,None)
                elif op=="import":
                    result=await owner_utility(model.engine_core,owner,"pd_import_target",
                                               args["payload"],args["salt"])
                else:
                    result=await owner_utility(model.engine_core,owner,"pd_drop_target",args["salt"])
            else:
                raise ValueError("Unknown actor operation")
            connection.send(("ok",result))
    except BaseException:
        try:connection.send(("error",traceback.format_exc()))
        except (BrokenPipeError,EOFError):pass
        raise
    finally:
        if model is not None:model.shutdown()
        connection.close()

def worker(kind,instance,connection):
    import os
    os.setsid()
    asyncio.run(run(kind,instance,connection))
