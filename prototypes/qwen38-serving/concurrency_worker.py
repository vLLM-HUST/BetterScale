"""Diagnostic-only observation of six mixed-concurrency steps; no policy changes."""

import os
import time
from profiling import ProfileWindow

if os.environ["COMPARE_NO_MTP"] == "candidate":
    if os.environ.get("PARTITION_CANDIDATE") == "1":
        from mixed_full_worker import Worker as Base
    else:
        from betterscale.qwen_worker import Worker as Base
else:
    from vllm_ascend.worker.worker import NPUWorker as Base


class Worker(Base):
    window = None

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        runner = self.model_runner
        original = runner._determine_batch_execution_and_padding

        def determine(num_tokens, num_reqs, num_scheduled_tokens_np, *args, **kwargs):
            result = original(
                num_tokens, num_reqs, num_scheduled_tokens_np, *args, **kwargs
            )
            w = self.window
            if w is not None and not w.closed:
                w.schedule.append(
                    dict(
                        event="dispatch",
                        wall_ns=time.time_ns(),
                        sequence=w.count,
                        tokens=num_tokens,
                        requests=num_reqs,
                        scheduled=num_scheduled_tokens_np.tolist(),
                        mode=str(result[0]),
                        descriptor=repr(result[1]),
                        request_ids=runner.input_batch.req_ids[:num_reqs],
                        computed=runner.input_batch.num_computed_tokens_cpu[
                            :num_reqs
                        ].tolist(),
                        prompt_tokens=runner.input_batch.num_prompt_tokens_cpu_tensor[
                            :num_reqs
                        ].tolist(),
                    )
                )
            return result

        runner._determine_batch_execution_and_padding = determine
        return result

    def profile(self, is_start=True, profile_prefix=None):
        if is_start:
            if self.window is not None:
                raise RuntimeError("one short capture only")
            self.window = ProfileWindow(
                os.environ["SERVING_PROFILE"], self.rank, steps=6
            )
        elif self.window is not None:
            self.window.close()

    def execute_model(self, scheduler_output):
        w = self.window
        if w is not None and not w.closed:
            w.schedule.append(
                dict(
                    event="execute_begin",
                    wall_ns=time.time_ns(),
                    sequence=w.count,
                    tokens=dict(scheduler_output.num_scheduled_tokens),
                    new_requests=[
                        r.req_id for r in scheduler_output.scheduled_new_reqs
                    ],
                )
            )
        result = super().execute_model(scheduler_output)
        if w is not None and not w.closed:
            w.schedule.append(
                dict(event="execute_end", wall_ns=time.time_ns(), sequence=w.count)
            )
        return result

    def sample_tokens(self, grammar_output):
        window = self.window
        before = window.count if window is not None else None
        result = super().sample_tokens(grammar_output)
        # The partition prototype already inherits the observation Worker's
        # step hook; native/package Workers do not. Advance exactly once.
        if window is not None and not window.closed and window.count == before:
            window.step()
        return result
