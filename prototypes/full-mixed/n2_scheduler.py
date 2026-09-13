"""Native two-wave AsyncScheduler with explicit retirement and reuse fences.

The engine's existing batch queue implements N -> N+2 authorization. This
extension does not create another scheduler queue or move sampling onto CPU.
"""
from collections import deque
from pathlib import Path
import json
import os
from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class N2Scheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.parallel_config.pipeline_parallel_size == 1
        assert self.scheduler_config.async_scheduling
        self.defer_block_free = True
        self.frames = deque()
        self.issued = self.retired = self.max_inflight = 0
        self.deferred_requests = self.peak_deferred_frees = 0

    def _free_request_blocks(self, request):
        before = len(self.deferred_frees)
        super()._free_request_blocks(request)
        self.deferred_requests += len(self.deferred_frees) > before
        self.peak_deferred_frees = max(self.peak_deferred_frees, len(self.deferred_frees))

    def add_request(self, request):
        # Streaming sessions have separate semantics; keep this prototype's
        # identity contract narrow rather than aliasing an old receipt.
        assert not request.resumable, 'streaming-input sessions not yet qualified'
        rid = request.request_id
        assert rid not in self.requests, 'duplicate live request ID'
        assert not any(rid in ids for _, ids in self.frames), 'request ID still in flight'
        return super().add_request(request)

    def schedule(self, *args, **kwargs):
        assert len(self.frames) < 2, 'native two-wave queue exceeded'
        output = super().schedule(*args, **kwargs)
        self.frames.append((output, frozenset(output.num_scheduled_tokens)))
        self.issued += 1
        self.max_inflight = max(self.max_inflight, len(self.frames))
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        assert self.frames and self.frames[0][0] is scheduler_output, 'receipt out of order'
        result = super().update_from_output(scheduler_output, model_runner_output)
        self.frames.popleft()
        self.retired += 1
        assert self.processed_step_seq <= self.sched_step_seq
        # Small cohort-boundary receipt, not synchronous per-wave file IO.
        if not self.frames:
            self.receipt()
        return result

    def receipt(self):
        path = Path(os.environ['FULL_MIXED_OUTPUT']) / 'n2-scheduler.json'
        path.write_text(json.dumps(dict(issued=self.issued, retired=self.retired,
            max_inflight=self.max_inflight, pending=len(self.frames),
            scheduled_device_steps=self.sched_step_seq,
            retired_device_steps=self.processed_step_seq,
            deferred_frees=len(self.deferred_frees),
            deferred_requests=self.deferred_requests,
            peak_deferred_frees=self.peak_deferred_frees), indent=2))
