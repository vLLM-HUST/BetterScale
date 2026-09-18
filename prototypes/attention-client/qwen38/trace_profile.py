"""Opt-in bounded attention capture; analyse after releasing accelerator roles."""

import os
from pathlib import Path
import torch_npu


def start(rank):
    output = Path(os.environ["QWEN38_PROFILE_DIR"])
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
            str(output), worker_name=f"attention{rank}", analyse_flag=False
        ),
    )
    profiler.start()
    return profiler
