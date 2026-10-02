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


def _cache_command(core, command):
    # Calling snapshot initializes the native completion inbox and drains it on
    # this same Core thread. No controller thread mutates scheduler structures.
    core.state_cache({"kind":"snapshot"})
    c=core.scheduler.cache_actions
    kind=command["kind"]
    if kind=="capacity":
        scheduler=core.scheduler;pool=scheduler.kv_cache_manager.block_pool
        return dict(block_size=scheduler.block_size,free_blocks=pool.get_num_free_blocks(),
                    device_release=hasattr(c,"device_result"),max_requests=scheduler.max_num_running_reqs,context_limit=core.vllm_config.model_config.max_model_len)
    if kind in ("snapshot","wait","cancel","drop"):
        return core.state_cache(command)
    if kind=="wait_device":return c.device_result(command["operation"])
    if kind=="describe":return asdict(c.host[command["key"]])
    if kind=="store_match":
        for seat in core.scheduler.residents.seats:
            if (seat.cache_salt==command["salt"] and seat.tokens==tuple(command["tokens"])
                    and seat.owner is None and seat.io_owner is None
                    and seat.fence<=core.scheduler.processed_step_seq):
                if core.vllm_config.additional_config.get("pd_rank_private", False):
                    if "peer_group" not in command:
                        raise ValueError("private State store requires explicit peer placement")
                    group = command["peer_group"]
                    if group is not None and (type(group) is not int or not 0 <= group < 4):
                        raise ValueError("invalid sticky peer group")
                number = c.store(seat.index,command["key"])
                if core.vllm_config.additional_config.get("pd_rank_private", False):
                    # Placement travels in the SAME ordered worker command as
                    # the store; no separate synchronous executor RPC barrier.
                    c.pending[number].command["peer_group"] = group
                return number
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


def cache_command(core, command):
    from betterscale.models.qwen35.cache_actions import CacheBusy
    try:return _cache_command(core,command)
    except CacheBusy:return None


class Scheduler(BaseScheduler):
    def __init__(self,*args,**kwargs):
        from vllm.v1.engine.core import EngineCoreProc
        EngineCoreProc.pd_online_cache=cache_command
        super().__init__(*args,**kwargs)
        if self.cache_actions is None or not self.cache_control_rpc:
            raise ValueError("Online PD requires incremental control-only cache")


class Worker(BaseWorker):
    def __init__(self,vllm_config,*args,**kwargs):
        decode_only=vllm_config.additional_config.get("pd_decode_only",False)
        if decode_only:
            from decode_graph_policy import qualify
            qualify(vllm_config)
        super().__init__(vllm_config,*args,**kwargs)
        if decode_only:
            from decode_graph_policy import install
            install()

    def state_cache_actions(self, commands):
        for command in commands:
            if "peer_group" in command:
                if command["kind"] != "store":
                    raise ValueError("peer placement belongs only to store")
                self.pd_rank_peer(command["key"], command["peer_group"])
        return super().state_cache_actions(commands)

    def pd_rank_peer(self, key, peer_group):
        from betterscale.live.runtime.host_state import HostStateKey
        if not isinstance(key, str) or not 1 <= len(key) <= 128:
            raise ValueError("invalid private checkpoint key")
        runtime = self.model_runner._pd_rank_runtime
        runtime.replica.set_peer(HostStateKey(key, 1), peer_group)
        return dict(owner=runtime.pool.owner, key=key, peer_group=peer_group)

    def pd_memory(self):
        import torch
        runner=self.model_runner
        worker=runner._state_cache_worker
        host=torch.npu.host_memory_stats()
        free,total=torch.npu.mem_get_info(runner.device)
        row=dict(owner=self.vllm_config.parallel_config.data_parallel_rank,rank=worker.rank,
                 device_free=free,device_total=total,
                 device_allocated=torch.npu.memory_allocated(runner.device),
                 device_reserved=torch.npu.memory_reserved(runner.device),
                 host={k:v for k,v in host.items() if k.endswith((".current",".peak"))})
        runtime=getattr(runner,"_pd_rank_runtime",None)
        if runtime is not None:
            with runtime.pool.lock:
                row["pool"]=dict(bytes=runtime.pool.bytes,budget=runtime.pool.budget,
                    objects=len(runtime.pool.objects),checkpoints=len(runtime.pool.groups),
                    referenced=len(runtime.pool.references),
                    readers=sum(x.readers for x in runtime.pool.objects.values()))
            if runtime.host_arena is not None:
                row["host_arena"]=runtime.host_arena.stats()
                from native_state_frame import FramePlan
                root=runner._live_state_root
                span=2048//root.capacity.page_tokens
                if span*root.capacity.page_tokens!=2048:raise ValueError("host frame page ratio")
                shapes=[]
                for domain,blocks in ((root.residents,(0,)),(root.pages,tuple(range(span)))):
                    shapes.append(FramePlan.build([(name,state,blocks) for name,state in worker.states
                                                   if state.domain is domain]).byte_length)
                row["host_geometry"]=dict(resident_frame_bytes=shapes[0],
                    page_frame_bytes=shapes[1],max_transfers=worker.max_transfers)
        return row

    def pd_object_audit(self,enabled):
        if type(enabled) is not bool:raise ValueError("Audit mode must be boolean")
        worker=self.model_runner._state_cache_worker
        if worker.inflight:raise RuntimeError("Cannot change audit during State I/O")
        previous=worker.page_backend.verify
        worker.page_backend.verify=enabled
        return dict(owner=self.vllm_config.parallel_config.data_parallel_rank,
                    rank=worker.rank,previous=previous,enabled=enabled)

    def execute_model(self,scheduler_output):
        if not os.environ.get("BETTERSCALE_PD_TIMING_DIR"):
            return super().execute_model(scheduler_output)
        import time
        from online_timing import Recorder
        if not hasattr(self,"_pd_timing"):self._pd_timing=Recorder("worker")
        begin=time.perf_counter_ns()
        result=super().execute_model(scheduler_output)
        self._pd_timing.record("worker-dispatch",begin_ns=begin,
            scheduled=dict(scheduler_output.num_scheduled_tokens),
            drafts={rid:len(ids) for rid,ids in scheduler_output.scheduled_spec_decode_tokens.items()})
        return result


    def compile_or_warm_up_model(self):
        result=super().compile_or_warm_up_model()
        from betterscale.models.qwen35.cache_worker import CacheWorker
        from betterscale.live.runtime.page_transport import ObjectStateTransport
        from online_objects import PeerObjectSink
        import os,json
        runner=self.model_runner;root=runner._live_state_root
        if os.environ.get("BETTERSCALE_PD_TIMING_DIR"):
            from online_timing import observe_method
            observe_method(self,"state_cache_actions","worker-state","state-cache-actions")
            observe_method(self.model_runner,"_sync_metadata_across_dp",
                           "worker-metadata","dp-metadata")
        concurrency=self.vllm_config.additional_config["state_cache_max_pending"]
        from state_numa import initializer
        worker=CacheWorker(runner,128<<20,max_transfers=concurrency,
                           waiter_initializer=initializer(runner.device.index))
        mtp=self.vllm_config.additional_config.get("pd_mtp",False)
        controls={id(root.continuation.selection),id(root.conv_selection),id(root.remaining_outputs)}
        worker.states=tuple((name,state) for name,state in root.named_states()
                            if name.startswith("target.") or (mtp and name.startswith("draft.")) or id(state) in controls)
        if len(worker.states)!=(85 if mtp else 83):raise RuntimeError("Unexpected target State closure")
        # A transferred full conv window retains its selector. Treating it as a
        # prior decode normalizes on bulk prefill; selector1 is an identity copy.
        worker.verify=defaultdict(lambda:True)
        if self.vllm_config.additional_config.get("pd_rank_private", False):
            from rank_state_runtime import build
            config = dict(self.vllm_config.additional_config)
            config["pd_rank_dp"] = self.vllm_config.parallel_config.data_parallel_rank
            runner._pd_rank_runtime = build(runner, worker, config)
            worker.page_backend = runner._pd_rank_runtime.transport
            runner._state_cache_worker = worker
            return result
        sink=PeerObjectSink(json.loads(os.environ["BETTERSCALE_PD_OBJECT_URLS"]),
                            max_transfers=concurrency)
        namespace=f"qwen35-target-state-v2/tp2/head{worker.rank}"
        compression=os.environ.get("BETTERSCALE_PD_COMPRESS_RESIDENT","0")
        if compression not in ("0","1"):raise ValueError("Invalid resident wire compression flag")
        if compression=="1":
            from online_codec import ResidentWireSink
            sink=ResidentWireSink(sink)
            namespace=f"qwen35-target-state-v3-zstd-resident/tp2/head{worker.rank}"
        batch_dma=os.environ.get("BETTERSCALE_PD_BATCH_DMA","0")
        if batch_dma not in ("0","1"):raise ValueError("Invalid State batch DMA flag")
        enqueue_copies=None
        if batch_dma=="1":
            import vllm_ascend.vllm_ascend_C
            from betterscale.models.qwen35.state_dma import enqueue as enqueue_copies
        worker.page_backend=ObjectStateTransport(
            sink,namespace,max_transfers=concurrency,enqueue_copies=enqueue_copies,
            verify=os.environ.get("BETTERSCALE_PD_VERIFY_OBJECTS")=="1")
        if mtp:worker.page_backend.namespace="mtp-prefix-v1/"+worker.page_backend.namespace
        runner._state_cache_worker=worker
        return result
