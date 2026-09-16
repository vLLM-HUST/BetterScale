"""Bounded native CANN/msprof collection; parse only after worker exit."""

from pathlib import Path
import json
import time


class ProfileWindow:
    def __init__(self, directory, rank, steps, *, warmup_steps=0):
        self.steps = steps
        self.warmup_steps = warmup_steps
        self.started = False
        self.count = 0
        self.prof = None
        self.closed = False
        self.schedule = []
        self.schedule_path = Path(directory) / f"schedule-rank{rank}.json"
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
            self.mark("profile_start")
            self.started = True

    def mark(self, event, plan=None):
        if not self.steps:
            return
        row = dict(
            event=event, wall_ns=time.time_ns(), monotonic_ns=time.perf_counter_ns()
        )
        if plan is not None:
            row["sequence"] = plan["sequence"]
            if event == "submit_begin":
                row.update(
                    key=plan["key"],
                    kind=plan["kind"],
                    lengths=plan["lengths"],
                    query_tokens=plan.get("query_tokens"),
                    residents=[
                        dict(
                            slot=r.lease.slot,
                            generation=r.lease.generation,
                            session=r.session,
                            turn=r.turn,
                        )
                        for r in plan["residents"]
                    ],
                )
        self.schedule.append(row)

    def step(self):
        self.count += 1
        if self.prof is None or self.closed:
            return
        if not self.started and self.count >= self.warmup_steps:
            import torch

            self.mark("warmup_drain_begin")
            torch.npu.synchronize()  # Drain warmup, not a timed performance path.
            self.mark("warmup_drain_end")
            self.prof.start()
            self.mark("profile_start")
            self.started = True
        elif self.started and self.count >= self.warmup_steps + self.steps:
            self.close()

    def close(self):
        if self.prof is not None and not self.closed:
            if self.started:
                import torch

                self.mark("profile_drain_begin")
                torch.npu.synchronize()  # Keep the last sampled replay in the window.
                self.mark("profile_drain_end")
                self.prof.stop()
                self.mark("profile_stop")
            self.closed = True
        if self.steps:
            self.schedule_path.parent.mkdir(parents=True, exist_ok=True)
            self.schedule_path.write_text(
                json.dumps(
                    dict(
                        scope="diagnostic host scheduler timestamps; wall clock must be verified against provider APIs before overlay",
                        warmup_steps=self.warmup_steps,
                        profile_steps=self.steps,
                        events=self.schedule,
                    ),
                    indent=2,
                )
            )
