"""Launch the qualified State-only route through native vLLM serving."""

import json
import os
from pathlib import Path
import sys

from . import capture_sizes, STATE_SCHEDULER
from .runtime import validate


def prepare(
    model,
    devices,
    port,
    cache_dir,
    runtime_dir,
    context_tokens,
    state_budget_bytes,
    served_model_name,
    distributed_port,
    execution_seats=16,
    resident_seats=20,
):
    if runtime_dir is None:
        raise ValueError(
            "Qwen35 live requires --qwen35-runtime-dir; first run "
            "python -m betterscale.models.qwen35.runtime --source DONOR --output RUNTIME"
        )
    runtime_dir = validate(Path(runtime_dir).resolve())
    if state_budget_bytes <= 0:
        raise ValueError(
            "State budget must be positive (resident State plus shared FA pages)"
        )
    from ...__main__ import prepare_libraries

    env = os.environ.copy()
    prepare_libraries(Path(__file__).parents[2], env)
    from ...patches.qwen_fia.context_parallel import configure

    configure(env)
    env.update(
        ASCEND_RT_VISIBLE_DEVICES=devices,
        PYTHON=sys.executable,
        VLLM_CACHE_ROOT=str(cache_dir.resolve()),
        BETTERSCALE_QWEN35_RUNTIME=str(runtime_dir),
        HCCL_IF_BASE_PORT=str(distributed_port),
    )
    command = [
        "bash",
        str(Path(__file__).with_name("serve.sh")),
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
        "--tensor-parallel-size",
        "2",
        "--distributed-executor-backend",
        "mp",
        "--worker-cls",
        "betterscale.qwen35_worker.Worker",
        "--dtype",
        "bfloat16",
        "--kv-cache-dtype",
        "auto",
        "--max-model-len",
        str(context_tokens),
        "--max-num-seqs",
        str(execution_seats),
        "--max-num-batched-tokens",
        "4096",
        "--gpu-memory-utilization",
        "0.9",
        "--seed",
        "17",
        "--enable-prefix-caching",
        "--mamba-cache-mode",
        "align",
        "--enable-prompt-tokens-details",
        "--async-scheduling",
        "--shutdown-timeout",
        "60",
        "--additional-config",
        json.dumps(dict(enable_cpu_binding=False, using_live_runtime=True,
                        state_resident_seats=resident_seats)),
        "--limit-mm-per-prompt",
        '{"image":0,"video":0}',
        "--compilation-config",
        json.dumps(
            dict(
                cudagraph_mode="FULL",
                cudagraph_capture_sizes=capture_sizes(execution_seats),
                max_cudagraph_capture_size=4096,
            )
        ),
        "--speculative-config",
        '{"method":"mtp","num_speculative_tokens":2}',
        "--scheduler-cls",
        STATE_SCHEDULER,
        "--kv-cache-memory-bytes",
        str(state_budget_bytes),
        "--generation-config",
        "vllm",
        "--override-generation-config",
        '{"temperature":0.0,"top_p":1.0,"top_k":-1,"presence_penalty":0.0}',
        "--default-chat-template-kwargs",
        '{"enable_thinking":true}',
    ]
    return command, env
