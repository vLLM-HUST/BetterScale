"""Observation only: four steady steps, native numerical/control paths unchanged."""

import os
import time
from vllm_ascend.worker.worker import NPUWorker


class Worker(NPUWorker):
    window = None

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        if os.environ.get("SERVING_PACK_CONV") == "1":
            from conv_layout import pack_conv_weights

            count = pack_conv_weights(self.model_runner.model)
            from vllm.logger import init_logger

            init_logger(__name__).info(
                "Packed %d immutable GDN convolution weights", count
            )
        return result

    def profile(self, is_start=True, profile_prefix=None):
        from profiling import ProfileWindow

        if is_start:
            if self.window is not None:
                raise RuntimeError("one bounded profile per worker lifetime")
            self.window = ProfileWindow(
                os.environ["SERVING_PROFILE"], self.rank, steps=4, warmup_steps=8
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
                    tokens=scheduler_output.num_scheduled_tokens,
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
        result = super().sample_tokens(grammar_output)
        if self.window is not None and not self.window.closed:
            self.window.step()
        return result
