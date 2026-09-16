"""Optional device-service profiling; parse offline after releasing the cards."""

import os
from pathlib import Path


def start(role):
    root = os.environ.get("DEVICE_SERVICE_PROFILE")
    if not root:
        return None
    import torch_npu

    profiler = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            str(Path(root)), worker_name=role, analyse_flag=False
        ),
    )
    profiler.start()
    return profiler


def stop(profiler):
    if profiler is not None:
        profiler.stop()
