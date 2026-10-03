"""Explicit hw81 P4 / hw86 D8 geometry; no host/network discovery."""
import os
from pathlib import Path

ROOT=Path("/workspace/betterscale-pd-runtime")

def placement(kind, instance):
    if kind=="P" and type(instance) is int and 0<=instance<4:
        return dict(devices=f"{2*instance},{2*instance+1}",dp=1,
                    hccl_port=29700+100*instance,rpc_port=30100+100*instance)
    if kind=="D" and instance==0:
        return dict(devices="0,1,2,3,4,5,6,7",dp=4,hccl_port=29675,rpc_port=29673)
    raise ValueError("Only four TP2 P instances or one DP4TP2EP8 D pool are admitted")

def engine_options(kind, instance):
    layout=placement(kind,instance)
    from pd_model_probe import prepare_worker,engine_options as base_options
    if kind=="D":
        os.environ["BETTERSCALE_PD_D_PACKAGE"]=str(ROOT/"candidate-package-10-ep8-state")
    prepare_worker(kind,native_async=True)
    options=base_options(kind=="P")
    import pool_state_entry
    os.environ.update(ASCEND_RT_VISIBLE_DEVICES=layout["devices"],
                      HCCL_IF_BASE_PORT=str(layout["hccl_port"]))
    from pd_limits import context_limit,state_budget
    options.update(max_model_len=context_limit(),kv_cache_memory_bytes=state_budget(),worker_cls="pool_state_entry.Worker",
                   scheduler_cls="pool_state_entry.Scheduler",
                   data_parallel_size=layout["dp"],data_parallel_size_local=layout["dp"],
                   data_parallel_address="127.0.0.1",data_parallel_rpc_port=layout["rpc_port"],
                   disable_log_stats=True)
    return options,layout
