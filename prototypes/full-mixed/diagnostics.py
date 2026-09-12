"""Opt-in decode-window observation; no execution/synchronization policy changes."""
import json
import os
import time
from pathlib import Path
import torch
from torch.profiler import record_function


def start_decode_observation(worker, profile=False, label="decode"):
    assert label.replace("-", "").isalnum()
    worker._decode_observation_label = label
    runner = worker.model_runner
    root = Path(os.environ['FULL_MIXED_OUTPUT'])
    drafter = getattr(runner, 'drafter', None)
    receipt = dict(rank=worker.rank, draft_class=type(drafter).__name__,
                   async_scheduling=runner.use_async_scheduling,
                   draft_use_graph=getattr(drafter, 'use_cuda_graph', None),
                   draft_runnable=type(getattr(drafter, '_runnable', None)).__name__,
                   target_wrapper=type(runner.model).__name__)
    (root / f'decode-coverage-rank{worker.rank}.json').write_text(json.dumps(receipt, indent=2))
    worker._decode_observation_originals = []
    worker._decode_wave_rows = []
    worker._decode_event_rows = []
    def wrap(obj, name, label):
        original = getattr(obj, name)
        def observed(*args, **kwargs):
            with record_function(label):
                event_pair = None
                if name in ('_model_forward', '_runnable'):
                    event_pair = [torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)]
                    event_pair[0].record()
                start = time.monotonic_ns()
                if name == 'execute_model' and args:
                    sched = args[0]
                    worker._decode_wave_rows.append(dict(
                        scheduled=dict(getattr(sched, 'num_scheduled_tokens', {})),
                        total=getattr(sched, 'total_num_scheduled_tokens', None)))
                result = original(*args, **kwargs)
                if event_pair is not None:
                    event_pair[1].record()
                    worker._decode_event_rows.append((label, start, time.monotonic_ns(), event_pair))
                return result
        worker._decode_observation_originals.append((obj, name, original))
        setattr(obj, name, observed)
    wrap(runner, 'execute_model', 'strengthen::target_step')
    wrap(runner, '_model_forward', 'strengthen::target_forward')
    wrap(runner, 'sample_tokens', 'strengthen::sample_and_draft')
    if drafter is not None:
        wrap(drafter, '_propose', 'strengthen::draft_prepare_and_forward')
        wrap(drafter, '_runnable', 'strengthen::draft_forward')
    if profile:
        import torch_npu
        worker._decode_profiler = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=False, profile_memory=False, with_stack=False,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                str(root / 'profile'), worker_name=f'rank{worker.rank}'))
        worker._decode_profiler.start()
    return receipt


def stop_decode_observation(worker):
    if hasattr(worker, '_decode_profiler'):
        worker._decode_profiler.stop()
        del worker._decode_profiler
    for obj, name, original in worker._decode_observation_originals:
        setattr(obj, name, original)
    torch.npu.synchronize()
    rows=worker._decode_event_rows
    origin=rows[0][3][0] if rows else None
    timed=[dict(label=label, host_start_ns=start, host_elapsed_ms=(end-start)/1e6,
                device_start_ms=origin.elapsed_time(pair[0]), device_elapsed_ms=pair[0].elapsed_time(pair[1]))
           for label,start,end,pair in rows]
    root = Path(os.environ['FULL_MIXED_OUTPUT'])
    phase=worker._decode_observation_label
    (root / f'{phase}-timing-rank{worker.rank}.json').write_text(json.dumps(timed,indent=2))
    (root / f'{phase}-waves-rank{worker.rank}.json').write_text(
        json.dumps(worker._decode_wave_rows, indent=2))
    return dict(rank=worker.rank, waves=len(worker._decode_wave_rows))
