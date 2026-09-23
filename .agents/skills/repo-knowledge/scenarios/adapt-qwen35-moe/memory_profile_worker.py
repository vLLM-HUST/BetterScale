"""Probe-only vLLM Worker using LiveInference's native memory attribution tools.

Put a pinned LiveInference src on PYTHONPATH. Select MEMORY_PROFILE_ARM=native
or full. No runtime memory policy changes; startup observations synchronize,
serving observations do not. Checkpoints must be requested only after drain.
"""
from contextlib import contextmanager
from dataclasses import asdict
from functools import wraps
import gzip
import json
import os
from pathlib import Path
import time

import torch
from livemodule.runtime.memory import TorchDeviceMemoryObserver, MemoryPhaseProfile
from livemodule.runtime.graph_memory import graph_pool_snapshot

if os.environ['MEMORY_PROFILE_ARM'] == 'native':
    from native_worker import Worker as Base
elif os.environ['MEMORY_PROFILE_ARM'] == 'full':
    from betterscale.worker import Worker as Base
else:
    raise ValueError('Select native or full explicitly')


class Worker(Base):
    def _memory_write(self, name, value):
        path = self._memory_dir / name
        path.write_text(json.dumps(value, indent=2)+'\n')

    def _memory_pools(self, segments):
        # Reuse the qualified pool/stream-aware reader, not a global delta.
        keys = sorted({(int(s['device']), tuple(s['segment_pool_id']), int(s['stream']))
                       for s in segments if tuple(s['segment_pool_id']) != (0,0)})
        pools = [graph_pool_snapshot(segments, device_index=d, pool_id=p, stream_id=s)
                 for d,p,s in keys]
        return [asdict(p) if p is not None else None for p in pools]

    def _memory_dump(self, label):
        snapshot = torch.npu.memory._snapshot()
        with gzip.open(self._memory_dir/f'{label}-snapshot.json.gz','wt') as f:
            json.dump(snapshot,f)
        self._memory_write(f'{label}-pools.json',self._memory_pools(snapshot['segments']))

    @contextmanager
    def _memory_phase(self, label, raw=False):
        before = self._memory_observer.snapshot()
        self._memory_observer.reset_peak_stats()
        error = None
        try:
            yield
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            after = self._memory_observer.snapshot()
            phase = MemoryPhaseProfile(label,None,before,after)
            row = asdict(phase)
            row.update(error=error, retained_allocated=phase.retained_allocated_bytes,
                       retained_reserved=phase.retained_reserved_bytes,
                       peak_allocated_delta=phase.peak_allocated_delta_bytes)
            with (self._memory_dir/'startup-phases.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            if raw or error:
                self._memory_dump(label)

    def load_model(self,*args,**kwargs):
        self._memory_dir = Path(os.environ['CAPSULE'])/f'memory-rank{self.rank}'
        self._memory_dir.mkdir()
        self._memory_observer = TorchDeviceMemoryObserver(self.device)
        torch.npu.memory._record_memory_history(stacks='python',max_entries=5000)
        with self._memory_phase('load',raw=True):
            result = super().load_model(*args,**kwargs)
        runner = self.model_runner
        original_dummy = runner._dummy_run
        self._memory_capturing = False
        self._memory_dummy_index = 0

        @wraps(original_dummy)
        def dummy(*args,**kwargs):
            if not self._memory_capturing:
                return original_dummy(*args,**kwargs)
            self._memory_dummy_index += 1
            tokens = args[0] if args else kwargs['num_tokens']
            mode = str(kwargs.get('cudagraph_runtime_mode','unspecified'))
            label = f'capture-{self._memory_dummy_index:03d}-t{tokens}-{mode}'
            with self._memory_phase(label,raw=self._memory_dummy_index<=2):
                return original_dummy(*args,**kwargs)
        runner._dummy_run = dummy
        original_capture = runner.capture_model
        @wraps(original_capture)
        def capture(*args,**kwargs):
            self._memory_capturing = True
            try:
                return original_capture(*args,**kwargs)
            finally:
                self._memory_capturing = False
                self._memory_dump('portfolio-finished')
        runner.capture_model = capture
        self._memory_steps = 0
        original_sample = runner.sample_tokens
        @wraps(original_sample)
        def sample(*args,**kwargs):
            result = original_sample(*args,**kwargs)
            self._memory_steps += 1
            # No added synchronize or peak reset in ordinary serving.
            if self._memory_steps<=4 or self._memory_steps%32==0:
                stats = torch.npu.memory_stats()
                row = {'step':self._memory_steps,'time_ns':time.monotonic_ns(),
                       'num_reqs':runner.input_batch.num_reqs,
                       **{k:int(stats[k]) for k in (
                           'allocated_bytes.all.current','allocated_bytes.all.peak',
                           'active_bytes.all.current','reserved_bytes.all.current',
                           'reserved_bytes.all.peak')}}
                with (self._memory_dir/'serving-waterlines.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
            return result
        runner.sample_tokens = sample
        return result

    def determine_available_memory(self):
        with self._memory_phase('profile-warmup',raw=True):
            return super().determine_available_memory()

    def initialize_from_config(self,*args,**kwargs):
        with self._memory_phase('kv-initialize',raw=True):
            return super().initialize_from_config(*args,**kwargs)

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        self.memory_checkpoint('worker-ready')
        return result

    def memory_checkpoint(self,label):
        if not label.replace('-','').isalnum():
            raise ValueError('Use an alphanumeric checkpoint label')
        value = asdict(self._memory_observer.snapshot())
        value['sample_steps'] = self._memory_steps
        self._memory_write(f'{label}-waterline.json',value)
        self._memory_dump(label)
        self._memory_observer.reset_peak_stats()
        return value
