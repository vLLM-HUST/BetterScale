"""Probe-only, per-rank CPU submission receipts; no device synchronization."""

import json
import time
from pathlib import Path


class StepObserver:
    def __init__(self, worker):
        self.file = (Path.cwd() / f"submission-rank{worker.rank}.jsonl").open("w")
        self.ordinal = 0

    def write(self, phase, **fields):
        self.file.write(
            json.dumps(
                dict(time=time.monotonic(), step=self.ordinal, phase=phase, **fields)
            )
            + "\n"
        )
        self.file.flush()

    def execute(self, original, schedule, *args, **kwargs):
        self.ordinal += 1
        cached = schedule.scheduled_cached_reqs
        self.write(
            "execute_enter",
            tokens=schedule.total_num_scheduled_tokens,
            counts=list(schedule.num_scheduled_tokens.values()),
            new=len(schedule.scheduled_new_reqs),
            resumed=len(cached.resumed_req_ids),
            finished=len(schedule.finished_req_ids),
        )
        result = original(schedule, *args, **kwargs)
        self.write("execute_return")
        return result
