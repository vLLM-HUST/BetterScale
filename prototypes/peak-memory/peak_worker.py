"""Observation-only Worker: same 0.4 program and dummy fixture, no allocator edits."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import torch
import betterscale
import vllm
import vllm_ascend
from betterscale.worker import Worker as NativeProgram
from capture_memory_trace import CaptureMemoryTrace
from dummy_weights import install

install()


class Worker(NativeProgram):
    def _record(self, phase, **extra):
        # CPU allocator counters only; no additional device synchronization.
        row = dict(phase=phase, rank=self.rank,
                   allocated=torch.npu.memory_allocated(),
                   reserved=torch.npu.memory_reserved(),
                   peak=torch.npu.max_memory_allocated(),
                   peak_reserved=torch.npu.max_memory_reserved(), **extra)
        self._peak_rows.append(row)
        self._flush()
        return row

    def _flush(self):
        path = Path(os.environ['PEAK_OUTPUT']); path.mkdir(parents=True, exist_ok=True)
        (path / f'phases-rank-{self.rank}.json').write_text(json.dumps(self._peak_rows))

    @contextmanager
    def _observe(self, phase):
        self._record(phase + ':enter')
        roots = [Path(betterscale.__file__).parent,
                 Path(vllm_ascend.__file__).parent / 'models',
                 Path(vllm_ascend.__file__).parent / 'ops',
                 Path(vllm_ascend.__file__).parent / 'attention',
                 Path(vllm.__file__).parent / 'model_executor' / 'models',
                 Path(vllm.__file__).parent / 'model_executor' / 'layers']
        trace = CaptureMemoryTrace(roots, os.environ.get('PEAK_TRACE') == '1'
                                   and phase != 'profile-and-trial')
        try:
            with trace:
                yield
        finally:
            self._record(phase + ':exit')
            path = Path(os.environ['PEAK_OUTPUT']) / f'{phase}-rank-{self.rank}.json'
            path.write_text(json.dumps(trace.records))

    def determine_available_memory(self):
        self._peak_rows = []
        with self._observe('profile-and-trial'):
            result = super().determine_available_memory()
        self._record('budget', kv_budget=result,
                     activation=self.peak_activation_memory,
                     non_torch=self.non_torch_memory,
                     model_bytes=self.model_runner.model_memory_usage)
        return result

    def capture_trial_program(self):
        with self._observe('trial-capture-and-draft'):
            return super().capture_trial_program()

    def snapshot(self, phase, **extra):
        result = super().snapshot(phase, **extra)
        self._record(phase, **extra)
        return result

    def compile_or_warm_up_model(self):
        with self._observe('final-capture-and-draft'):
            result = super().compile_or_warm_up_model()
        self._execution_observations = 0
        return result

    def execute_model(self, *args, **kwargs):
        # Eight calls suffice to expose replay allocation growth; no timing claim.
        if self._execution_observations >= 8:
            return super().execute_model(*args, **kwargs)
        index = self._execution_observations
        self._execution_observations += 1
        self._record(f'execute-{index}:enter')
        result = super().execute_model(*args, **kwargs)
        self._record(f'execute-{index}:exit')
        return result
