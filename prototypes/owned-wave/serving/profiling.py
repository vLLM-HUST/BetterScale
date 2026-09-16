"""Bounded native CANN/msprof collection; parse only after worker exit."""

from pathlib import Path


class ProfileWindow:
    def __init__(self, directory, rank, steps, *, warmup_steps=0):
        self.steps = steps
        self.warmup_steps = warmup_steps
        self.started = False
        self.count = 0
        self.prof = None
        self.closed = False
        if not steps:
            return
        import torch_npu

        self.prof = torch_npu.profiler.profile(
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
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                str(Path(directory) / "profile"),
                worker_name=f"rank{rank}",
                analyse_flag=False,
            ),
        )
        if not warmup_steps:
            self.prof.start()
            self.started = True

    def step(self):
        self.count += 1
        if self.prof is None or self.closed:
            return
        if not self.started and self.count >= self.warmup_steps:
            import torch

            torch.npu.synchronize()  # Drain warmup, not a timed performance path.
            self.prof.start()
            self.started = True
        elif self.started and self.count >= self.warmup_steps + self.steps:
            self.close()

    def close(self):
        if self.prof is not None and not self.closed:
            if self.started:
                import torch

                torch.npu.synchronize()  # Keep the last sampled replay in the window.
                self.prof.stop()
            self.closed = True
