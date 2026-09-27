"""Probe-only raw CANN capture and host markers; no copy audit or DMA barrier."""

import json
import os
from pathlib import Path
import threading
import time


def stamp(event, **fields):
    row = dict(
        event=event,
        wall_ns=time.time_ns(),
        monotonic_ns=time.perf_counter_ns(),
        pid=os.getpid(),
        tid=threading.get_native_id(),
        **fields,
    )
    path = Path(os.environ["CAPSULE"]) / f"host-events-{os.getpid()}.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def install():
    import torch_npu
    from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper
    from betterscale.models.qwen35.cache_worker import CacheWorker

    def create(config, trace_name):
        from vllm.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
        stamp(
            "profile_owner",
            rank=rank,
            device=torch_npu.npu.current_device(),
            trace_name=trace_name,
        )
        return torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                export_type=torch_npu.profiler.ExportType.Db,
                msprof_tx=True,
                data_simplification=False,
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                config.torch_profiler_dir, worker_name=f"rank{rank}", analyse_flag=False
            ),
        )

    TorchNPUProfilerWrapper._create_profiler = staticmethod(create)
    original_submit, original_send = CacheWorker.submit, CacheWorker._send

    def mark(worker, command, event):
        name = f"StateCache/{command['kind']}/{command['operation']}/{event}"
        torch_npu.npu.mstx.mark(name)
        stamp(name, rank=worker.rank, seat=command["seat"], epoch=command["epoch"])

    def submit(worker, command):
        mark(worker, command, "submit_begin")
        try:
            return original_submit(worker, command)
        finally:
            mark(worker, command, "submit_end")

    def send(worker, command, error=None):
        mark(worker, command, "receipt_error" if error else "receipt_ready")
        return original_send(worker, command, error)

    CacheWorker.submit, CacheWorker._send = submit, send
