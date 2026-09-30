"""Launch the qualified State-only route through native vLLM serving."""

import json
import os
from pathlib import Path
import sys

from . import CAPTURE_SIZES, STATE_SCHEDULER, UNIFIED_CAPTURE_SIZES
from .runtime import matching_profile, validate


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
):
    if runtime_dir is None:
        raise ValueError(
            "Qwen35 live requires --qwen35-runtime-dir; first run "
            "python -m betterscale.models.qwen35.runtime --source DONOR --output RUNTIME"
        )
    runtime_dir = validate(Path(runtime_dir).resolve())
    profile = matching_profile(runtime_dir)
    unified = profile["pins_name"] == "qwen35_unified_pins.json"
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
        BETTERSCALE_QWEN35_CONTRACT="unified" if unified else "legacy",
        HCCL_IF_BASE_PORT=str(distributed_port),
    )
    if unified:
        # The fixed-host checkout carries a development distribution label.
        # Select the matching Ascend 0.25.1 API branches explicitly, as the
        # unified Native launch does, while source hashes remain authoritative.
        env["VLLM_VERSION"] = "0.25.1"
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
        "16",
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
        '{"enable_cpu_binding":false,"using_live_runtime":true}',
        "--limit-mm-per-prompt",
        '{"image":0,"video":0}',
        "--compilation-config",
        json.dumps(
            dict(
                cudagraph_mode=("FULL_AND_PIECEWISE" if unified else "FULL"),
                cudagraph_capture_sizes=(
                    UNIFIED_CAPTURE_SIZES if unified else CAPTURE_SIZES
                ),
                max_cudagraph_capture_size=(48 if unified else 4096),
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
