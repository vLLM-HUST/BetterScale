"""Opt-in target-only online State entry; model waves remain native DP/EP."""
from dataclasses import asdict
from collections import defaultdict
from pool_state_entry import Scheduler as BaseScheduler, Worker as BaseWorker
from betterscale.models import qwen35

import os
if os.environ.get("BETTERSCALE_PD_TIMING_DIR"):
    from online_timing import install_core
    install_core()

qwen35.STATE_SCHEDULER="online_entry.Scheduler"


def cache_command(core, command):
    # Calling snapshot initializes the native completion inbox and drains it on
    # this same Core thread. No controller thread mutates scheduler structures.
    core.state_cache({"kind":"snapshot"})
    c=core.scheduler.cache_actions
    kind=command["kind"]
    if kind=="capacity":
        scheduler=core.scheduler;pool=scheduler.kv_cache_manager.block_pool
        return dict(block_size=scheduler.block_size,free_blocks=pool.get_num_free_blocks(),
                    max_requests=scheduler.max_num_running_reqs,context_limit=core.vllm_config.model_config.max_model_len)
    if kind in ("snapshot","wait","cancel","drop"):
        return core.state_cache(command)
    if kind=="describe":return asdict(c.host[command["key"]])
    if c.pending:return None
    if kind=="store_match":
        for seat in core.scheduler.residents.seats:
            if (seat.cache_salt==command["salt"] and seat.tokens==tuple(command["tokens"])
                    and seat.owner is None and seat.io_owner is None
                    and seat.fence<=core.scheduler.processed_step_seq):
                return c.store(seat.index,command["key"])
        return None
    if kind=="adopt":
        value=command["checkpoint"]
        tokens=value["tokens"]
        if (not isinstance(tokens,(tuple,list)) or not 1<len(tokens)<=core.vllm_config.model_config.max_model_len
                or any(type(t) is not int or not 0<=t<248320 for t in tokens)
                or not isinstance(value["key"],str) or not 1<=len(value["key"])<=128
                or not isinstance(value["salt"],str) or not 1<=len(value["salt"])<=128):
            raise ValueError("Invalid remote checkpoint metadata")
        cp=c._checkpoint(value["key"],tokens,value["salt"],value["block_count"])
        if tuple(value["pages"])!=cp.pages or value["byte_length"]!=cp.byte_length:
            raise ValueError("Remote State geometry/version identity mismatch")
        old=c.host.get(cp.key)
        if old is not None and old!=cp:raise ValueError("Checkpoint identity collision")
        c.host[cp.key]=cp
        return True
    if kind=="load_match":
        cp=c.host[command["key"]]
        offer=core.scheduler.residents.offer(cp.tokens,cp.salt,core.scheduler.processed_step_seq)
        if offer is None:return None
        if offer.warm:return {"already_resident":True}
        pool=core.scheduler.kv_cache_manager.block_pool
        seat=core.scheduler.residents.seats[offer.seat]
        if pool.get_num_free_blocks()+(len(seat.blocks.blocks[0]) if seat.blocks else 0)<cp.block_count:
            return None
        return c.load(cp.key,offer.seat)
    raise ValueError("Unqualified online cache command")


class Scheduler(BaseScheduler):
    def __init__(self,*args,**kwargs):
        from vllm.v1.engine.core import EngineCoreProc
        EngineCoreProc.pd_online_cache=cache_command
        super().__init__(*args,**kwargs)
        if self.cache_actions is None or not self.cache_control_rpc:
            raise ValueError("Online PD requires incremental control-only cache")


class Worker(BaseWorker):
    def execute_model(self,scheduler_output):
        if not os.environ.get("BETTERSCALE_PD_TIMING_DIR"):
            return super().execute_model(scheduler_output)
        import time
        from online_timing import Recorder
        if not hasattr(self,"_pd_timing"):self._pd_timing=Recorder("worker")
        begin=time.perf_counter_ns()
        result=super().execute_model(scheduler_output)
        self._pd_timing.record("worker-dispatch",begin_ns=begin,
            scheduled=dict(scheduler_output.num_scheduled_tokens))
        return result


    def compile_or_warm_up_model(self):
        result=super().compile_or_warm_up_model()
        from betterscale.models.qwen35.cache_worker import CacheWorker
        from betterscale.live.runtime.page_transport import ObjectStateTransport
        from online_objects import PeerObjectSink
        import os,json
        runner=self.model_runner;root=runner._live_state_root
        worker=CacheWorker(runner,128<<20)
        controls={id(root.continuation.selection),id(root.conv_selection),id(root.remaining_outputs)}
        worker.states=tuple((name,state) for name,state in root.named_states()
                            if name.startswith("target.") or id(state) in controls)
        if len(worker.states)!=83:raise RuntimeError("Unexpected target State closure")
        # A transferred full conv window retains its selector. Treating it as a
        # prior decode normalizes on bulk prefill; selector1 is an identity copy.
        worker.verify=defaultdict(lambda:True)
        worker.page_backend=ObjectStateTransport(
            PeerObjectSink(json.loads(os.environ["BETTERSCALE_PD_OBJECT_URLS"])),
            f"qwen35-target-state-v2/tp2/head{worker.rank}",
            verify=os.environ.get("BETTERSCALE_PD_VERIFY_OBJECTS")=="1")
        runner._state_cache_worker=worker
        return result
