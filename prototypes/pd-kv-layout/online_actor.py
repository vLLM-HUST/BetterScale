"""Multiplex independent native requests and owner-local State control.

One pipe reader, tagged replies, bounded callers. Native AsyncLLM remains the
only model scheduler; utility futures complete from actual TP DMA receipts.
"""
import asyncio
import json
import os
import traceback
import time
import uuid
from pathlib import Path


def options(kind,instance):
    from pool_topology import placement
    from pd_model_probe import prepare_worker,engine_options
    from pd_limits import context_limit,state_budget
    layout=placement(kind,instance)
    prepare_worker(kind,native_async=True)
    if os.environ.get("BETTERSCALE_PD_RANK_PRIVATE") == "1":
        # Resolve Torch from the pinned model environment first, then retain
        # the isolated Mooncake-only site for subsequently spawned workers.
        import torch
        import torch_npu
        from rank_state_runtime import transfer_engine
        transfer_engine()
    value=engine_options(kind=="P")
    import online_entry
    os.environ.update(ASCEND_RT_VISIBLE_DEVICES=layout["devices"],HCCL_IF_BASE_PORT=str(layout["hccl_port"]))
    value.update(worker_cls="online_entry.Worker",scheduler_cls="online_entry.Scheduler",
                 max_model_len=context_limit(),kv_cache_memory_bytes=state_budget(),
                 data_parallel_size=layout["dp"],data_parallel_size_local=layout["dp"],
                 data_parallel_address="127.0.0.1",data_parallel_rpc_port=layout["rpc_port"],
                 disable_log_stats=True)
    value["additional_config"].update(state_cache_host_bytes=128<<30,
        state_cache_incremental=True,state_cache_control_rpc=True,state_cache_policy=False,
        state_cache_max_pending=20,pd_mtp=os.environ.get("BETTERSCALE_PD_MTP")=="1")
    private = os.environ.get("BETTERSCALE_PD_RANK_PRIVATE", "0")
    if private not in ("0", "1"):
        raise ValueError("Invalid private rank backend flag")
    if private == "1":
        if value["additional_config"]["pd_mtp"] or os.environ.get("BETTERSCALE_PD_COMPRESS_RESIDENT", "0") != "0":
            raise ValueError("private rank qualification requires target-only uncompressed State")
        value["additional_config"].update(pd_rank_private=True, pd_rank_role=kind,
            pd_rank_instance=instance, state_cache_two_phase_store=True)
    profile_dir=os.environ.get("BETTERSCALE_PD_PROFILE_DIR")
    if profile_dir:
        value["profiler_config"]=dict(profiler="torch",torch_profiler_dir=profile_dir,
            torch_profiler_with_stack=False,torch_profiler_record_shapes=False,
            torch_profiler_with_memory=False,ignore_frontend=True)
    return value,layout


async def run(kind,instance,pipe):
    value,layout=options(kind,instance)
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    model=AsyncLLM.from_engine_args(AsyncEngineArgs(**value))
    tasks=set();failed=False
    async def call(tag,op,args):
        nonlocal failed
        try:
            owner=args["owner"]
            if type(owner) is not int or not 0<=owner<layout["dp"]:raise ValueError("Bad owner")
            if op=="generate":
                import time
                result=None;arrivals=[];emitted=0;started=time.perf_counter_ns();request_id=uuid.uuid4().hex
                async for item in model.generate(
                    dict(prompt_token_ids=args["tokens"],cache_salt=args["salt"]),
                    SamplingParams(temperature=0,max_tokens=args["n"],ignore_eos=True,
                                   output_kind=RequestOutputKind.CUMULATIVE),
                    request_id,data_parallel_rank=owner):
                    result=item
                    now=time.perf_counter_ns()
                    current=item.outputs[0].token_ids
                    arrivals.append((len(current),now))
                    if args.get("_stream") and len(current)>emitted:
                        pipe.send((tag,"tokens",dict(token_ids=list(current[emitted:]),
                            arrival_ns=now,request_id=request_id,cached=item.num_cached_tokens)))
                        emitted=len(current)
                ids=list(result.outputs[0].token_ids)
                if len(ids)!=args["n"]:raise RuntimeError("Incomplete generation")
                answer=dict(token_ids=ids,full_tokens=list(result.prompt_token_ids)+ids,
                            cached=result.num_cached_tokens,start_ns=started,arrivals=arrivals,request_id=request_id)
            elif op=="audit":
                rows=await asyncio.gather(*(model.engine_core._call_utility_async(
                    "collective_rpc","pd_object_audit",None,(args["enabled"],),None,engine=e)
                    for e in model.engine_core.core_engines))
                answer=[row for group in rows for row in group]
                expected=[(o,t) for o in range(layout["dp"]) for t in range(2)]
                if (sorted((r["owner"],r["rank"]) for r in answer)!=expected
                        or any(r["enabled"] is not args["enabled"] for r in answer)):
                    raise RuntimeError("Incomplete audit-mode quorum")
            elif op=="profile":
                if not os.environ.get("BETTERSCALE_PD_PROFILE_DIR"):
                    raise ValueError("Profiler disabled for this node")
                if args["start"]:await model.start_profile(f"{kind}{instance}")
                else:await model.stop_profile()
                answer=dict(kind=kind,instance=instance,started=args["start"])
            elif op=="cache":
                answer=await model.engine_core._call_utility_async("pd_online_cache",args["command"],
                    engine=model.engine_core.core_engines[owner])
            else:raise ValueError("Unknown operation")
            pipe.send((tag,"ok",answer))
        except BaseException:
            failed=True
            pipe.send((tag,"error",traceback.format_exc()))
    try:
        capacities=[await model.engine_core._call_utility_async("pd_online_cache",{"kind":"capacity"},
            engine=model.engine_core.core_engines[i]) for i in range(layout["dp"])]
        pipe.send(("ready","ready",dict(capacities=capacities,kind=kind,instance=instance,layout=layout,
            context_limit=value["max_model_len"],state_bytes=value["kv_cache_memory_bytes"],target_only=not value["additional_config"]["pd_mtp"],
            object_audit=os.environ.get("BETTERSCALE_PD_VERIFY_OBJECTS")=="1",
            state_wire=("rank-private-v1" if value["additional_config"].get("pd_rank_private") else
                ("mtp-prefix-" if value["additional_config"]["pd_mtp"] else "")+
                ("zstd-resident-v1" if os.environ.get("BETTERSCALE_PD_COMPRESS_RESIDENT")=="1" else "raw-v2")))))
        while True:
            tag,op,args=await asyncio.to_thread(pipe.recv)
            if op=="stop":
                await asyncio.gather(*tasks)
                pipe.send((tag,"ok",None));break
            if failed or len(tasks)>=160:
                pipe.send((tag,"error","Actor failed closed or capacity exceeded"));continue
            task=asyncio.create_task(call(tag,op,args));tasks.add(task)
            task.add_done_callback(tasks.discard)
    finally:
        model.shutdown();pipe.close()


def worker(kind,instance,pipe):
    os.setsid()
    asyncio.run(run(kind,instance,pipe))


class Actor:
    def __init__(self,kind,instance):
        import multiprocessing as mp
        context=mp.get_context("spawn")
        self.pipe,child=context.Pipe()
        self.process=context.Process(target=worker,args=(kind,instance,child),name=f"online-{kind}{instance}")
        self.process.start();child.close()
        self.pending={};self.progress={};self.quarantined=False;self.info=None;self.reader=None

    async def ready(self):
        tag,status,info=await asyncio.wait_for(asyncio.to_thread(self.pipe.recv),1200)
        if (tag,status)!=("ready","ready"):raise RuntimeError(str(info))
        self.info=info;self.reader=asyncio.create_task(self.receive())
        return info

    def receive_batch(self):
        # One blocking wait, then drain only already-ready messages. Native DP
        # returns bursts: do not pay a thread-pool wakeup for every token chunk.
        # The bound yields back to HTTP/State work even under sustained output.
        batch=[self.pipe.recv()]
        while len(batch)<64 and self.pipe.poll():
            try:batch.append(self.pipe.recv())
            except EOFError:break  # Deliver preceding replies before quarantine.
        return batch

    async def receive(self):
        try:
            while True:
                batch=await asyncio.to_thread(self.receive_batch)
                for tag,status,value in batch:
                    if status=="tokens":
                        callback=self.progress.get(tag)
                        if callback is not None:
                            value["node_arrival_ns"]=time.perf_counter_ns()
                            callback(value)
                        continue
                    self.progress.pop(tag,None)
                    future=self.pending.pop(tag)
                    if status!="ok":raise RuntimeError(value)
                    if not future.done():future.set_result(value)
        except BaseException as error:
            self.quarantined=True
            # Include the just-removed failing caller, not only remaining tags.
            if "future" in locals() and not future.done():future.set_exception(RuntimeError(str(error)))
            for f in self.pending.values():
                if not f.done():f.set_exception(RuntimeError(str(error)))
            self.pending.clear()

    async def call(self,op,args,on_tokens=None):
        if self.quarantined or not self.process.is_alive():raise RuntimeError("Actor unavailable")
        if len(self.pending)>=160:raise RuntimeError("Actor request bound exceeded")
        tag=uuid.uuid4().hex;future=asyncio.get_running_loop().create_future();self.pending[tag]=future
        if on_tokens is not None:
            if op!="generate":raise ValueError("Only generation streams tokens")
            self.progress[tag]=on_tokens;args=dict(args,_stream=True)
        self.pipe.send((tag,op,args))
        try:return await asyncio.wait_for(asyncio.shield(future),1800)
        except BaseException:
            self.quarantined=True
            raise

    async def close(self):
        import signal
        if self.process.is_alive() and not self.quarantined:
            try:await asyncio.wait_for(self.call("stop",{}),60)
            except Exception:pass
        await asyncio.to_thread(self.process.join,60)
        if self.process.is_alive():
            if os.getpgid(self.process.pid)!=self.process.pid:raise RuntimeError("Foreign actor process group")
            os.killpg(self.process.pid,signal.SIGTERM)
            await asyncio.to_thread(self.process.join,20)
        if self.process.is_alive():
            os.killpg(self.process.pid,signal.SIGKILL)
            await asyncio.to_thread(self.process.join,10)
        self.pipe.close()
        if self.reader:
            self.reader.cancel();await asyncio.gather(self.reader,return_exceptions=True)
