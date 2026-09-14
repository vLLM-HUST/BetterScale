"""Probe-only quiescent RPC receipts; never installed on the public Worker.

The harness binds the native development RPC endpoint to loopback only. Reset
peaks after final startup, then query after each drained cohort, not per step.
Allocator peaks cover execution; driver free memory is only an endpoint sample.
"""

import torch
from preflight_worker import PreflightWorker


class RuntimeMemoryWorker(PreflightWorker):
    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        self._execution_baseline = self.snapshot("execution_baseline")
        torch.npu.reset_peak_memory_stats()
        return result

    def capacity_snapshot(self, phase="after_cohort"):
        record = self.snapshot(phase)
        record["baseline"] = self._execution_baseline
        return record
