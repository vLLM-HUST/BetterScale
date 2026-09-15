"""Observation RPCs only: native worker, scheduler and model execution unchanged."""
import json
from pathlib import Path


class Observer:
    def start_observation(self, directory):
        import torch
        import torch_npu

        torch.npu.synchronize()
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self._baseline_profiler = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True,
            with_stack=False,
            profile_memory=False,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                export_type=torch_npu.profiler.ExportType.Db,
                op_attr=True,
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                str(out), worker_name="rank0", analyse_flag=False
            ),
        )
        self._baseline_profiler.start()
        config = self.vllm_config
        return dict(
            rank=self.rank,
            model=config.model_config.model,
            dtype=str(config.model_config.dtype),
            graph_mode=str(config.compilation_config.cudagraph_mode),
            graph_sizes=config.compilation_config.cudagraph_capture_sizes,
            async_scheduling=config.scheduler_config.async_scheduling,
            allocated_bytes=torch.npu.memory_allocated(),
            reserved_bytes=torch.npu.memory_reserved(),
        )

    def stop_observation(self):
        import torch

        torch.npu.synchronize()
        self._baseline_profiler.stop()
        del self._baseline_profiler
