"""Diagnostic observer: external device events, resolved only after HTTP drain.

Not a production Worker or scheduler. Both arms run the same instrumentation.
The forward envelope includes submission/update waits, not just graph kernels.
"""

import time

from concurrency_worker import Worker as Base


class Worker(Base):
    measuring = False

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        runner = self.model_runner
        import os

        if os.environ.get("PROBE_NZ_GATE_UP") == "1":
            from nz_gate_up import apply

            apply(runner.model)
        if os.environ.get("PROBE_MC2_ROWS") == "1":
            from mc2_rows import apply

            apply(runner.model)
        determine = runner._determine_batch_execution_and_padding
        forward = runner._model_forward

        def observe_shape(num_tokens, num_reqs, num_scheduled_tokens_np, *args, **kwargs):
            result = determine(num_tokens, num_reqs, num_scheduled_tokens_np, *args, **kwargs)
            if self.measuring:
                self.current.update(
                    tokens=int(num_tokens),
                    requests=int(num_reqs),
                    scheduled=num_scheduled_tokens_np.tolist(),
                    computed=runner.input_batch.num_computed_tokens_cpu[
                        :num_reqs
                    ].tolist(),
                    prompt_tokens=runner.input_batch.num_prompt_tokens_cpu_tensor[
                        :num_reqs
                    ].tolist(),
                    mode=str(result[0]),
                    descriptor=repr(result[1]),
                )
            return result

        def observe_forward(*args, **kwargs):
            if not self.measuring:
                return forward(*args, **kwargs)
            assert not self.current.get("forward"), "multiple forwards in one step"
            self.current["forward"] = True
            events = self.events[len(self.rows) - 1]
            events[1].record()
            result = forward(*args, **kwargs)
            events[2].record()
            return result

        runner._determine_batch_execution_and_padding = observe_shape
        runner._model_forward = observe_forward
        return result

    def begin_step_measurement(self, label):
        import torch

        assert not self.measuring
        # Allocate and initialize outside the workload. No timed-step syncs.
        if not hasattr(self, "events"):
            self.events = [
                [torch.npu.Event(enable_timing=True) for _ in range(3)]
                for _ in range(256)
            ]
            for events in self.events:
                for event in events:
                    event.record()
        torch.npu.synchronize()
        self.label, self.rows = label, []
        self.measuring = True
        return {"rank": self.rank, "label": label}

    def execute_model(self, scheduler_output):
        if self.measuring and scheduler_output.num_scheduled_tokens:
            assert len(self.rows) < len(self.events), "bounded observer exhausted"
            self.current = dict(index=len(self.rows), host_start_ns=time.perf_counter_ns())
            self.events[len(self.rows)][0].record()
            self.rows.append(self.current)
        return super().execute_model(scheduler_output)

    def end_step_measurement(self):
        import torch

        self.measuring = False
        torch.npu.synchronize()  # RPC issued only after every HTTP response ends.
        for i, row in enumerate(self.rows):
            start, before, after = self.events[i]
            assert row.get("forward"), row
            row["prepare_ms"] = start.elapsed_time(before)
            row["forward_ms"] = before.elapsed_time(after)
            if i + 1 < len(self.rows):
                following = self.events[i + 1][0]
                row["period_ms"] = start.elapsed_time(following)
                row["after_forward_ms"] = after.elapsed_time(following)
                row["host_period_ms"] = (
                    self.rows[i + 1]["host_start_ns"] - row["host_start_ns"]
                ) / 1e6
        return dict(rank=self.rank, label=self.label, steps=self.rows)
