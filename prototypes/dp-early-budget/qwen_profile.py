"""Bounded two-rank native profile, outside warmup; parse offline, not in workers."""

import json
from pathlib import Path


def install(runner, root):
    import torch
    import torch_npu
    from vllm.forward_context import get_forward_context

    rank = runner.vllm_config.parallel_config.data_parallel_rank
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    original = runner._model_forward
    rows = []
    sequence = 0
    profiler = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(wait=8, warmup=1, active=16, repeat=1),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            str(root / "profile"), worker_name=f"rank{rank}", analyse_flag=False
        ),
    )
    profiler.start()

    def forward(*args, **kwargs):
        nonlocal sequence
        if sequence >= 25:
            return original(*args, **kwargs)
        context = get_forward_context()
        rows.append(
            dict(
                sequence=sequence,
                requests=runner.input_batch.num_reqs,
                mode=str(context.cudagraph_runtime_mode),
            )
        )
        with torch.profiler.record_function("betterscale::qwen_target_forward"):
            result = original(*args, **kwargs)
        sequence += 1
        profiler.step()
        if sequence == 25:
            profiler.stop()
            (root / f"profile-window-rank{rank}.json").write_text(
                json.dumps(
                    dict(rank=rank, wait=8, warmup=1, active=16, rows=rows), indent=2
                )
            )
        return result

    runner._model_forward = forward
