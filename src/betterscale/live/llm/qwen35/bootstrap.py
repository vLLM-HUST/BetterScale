"""Explicit physical/weight bootstrap, stopping before every native runner seam."""

import json
import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def open_model(
    model_path,
    *,
    context_tokens=512,
    resident_seats=20,
    token_pages=0,
    execution_seats=16,
    memory_floor_bytes=1 << 30,
    paged_attention=False,
    prefill_tokens=0,
):
    from datetime import timedelta

    import torch
    import torch_npu  # noqa: F401
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        get_world_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    from vllm_ascend.utils import enable_custom_op

    from betterscale.live import LiveRuntime, TorchStateBackend, live_runtime
    from betterscale.live.arch.ascend.graph import ACLGraphBackend
    from betterscale.live.runtime.capacity import GlooStateCapacityCoordinator
    from betterscale.live.runtime.memory import TorchDeviceMemoryObserver
    from betterscale.models.qwen import check_runtime

    from . import Capacity, Geometry
    from .graphs import QwenLiveLLMRoot
    from .loader import load_models

    path = Path(model_path).resolve()
    tp = int(os.environ["WORLD_SIZE"])
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    geometry = Geometry.from_config(
        json.loads((path / "config.json").read_text()), tensor_parallel_size=tp
    )
    capacity = Capacity(
        execution_seats,
        resident_seats,
        token_pages=token_pages or None,
        prefill_tokens=prefill_tokens,
    )
    if not 3 <= context_tokens <= (262144 if paged_attention else 4096):
        raise ValueError("live context exceeds the selected attention implementation")
    check_runtime("live_qwen_pins.json")
    if prefill_tokens:
        from betterscale.patches.qwen_gdn import check_library

        check_library()
    torch.npu.set_device(local_rank)
    if not enable_custom_op():
        raise RuntimeError("required Ascend numerical custom operators are unavailable")
    init_device_properties_triton()
    torch.set_default_dtype(torch.bfloat16)
    config = EngineArgs(
        model=str(path),
        dtype="bfloat16",
        tensor_parallel_size=tp,
        max_model_len=max(4096, context_tokens),
        max_num_seqs=execution_seats,
        enable_expert_parallel=False,
        enforce_eager=True,
        enable_prefix_caching=False,
        speculative_config={"method": "mtp", "num_speculative_tokens": 2},
    ).create_engine_config()
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=tp,
            rank=rank,
            local_rank=local_rank,
            backend="hccl",
            distributed_init_method="env://",
            timeout=timedelta(seconds=180),
        )
        initialize_model_parallel(tensor_model_parallel_size=tp, backend="hccl")
        device = f"npu:{local_rank}"
        root = None
        try:
            target, draft = load_models(config, path, device)
            with live_runtime(
                LiveRuntime(
                    device=device,
                    state_backend=TorchStateBackend(
                        device,
                        # Pinned torch_npu large-segment quantum. Reserve one
                        # quantum per KV tensor; never credit fragmented cache.
                        allocation_granularity_bytes=20 << 20,
                        # Exact resident State is charged first. The elastic
                        # attention domain receives the remaining physical
                        # budget after graph calibration and the free floor;
                        # neither seat count nor context is a page-pool quota.
                        capacity_coordinator=GlooStateCapacityCoordinator(
                            get_world_group().cpu_group
                        ),
                    ),
                    graph_backend=ACLGraphBackend(device=device),
                    memory_observer=TorchDeviceMemoryObserver(device=device),
                )
            ):
                root = QwenLiveLLMRoot(
                    geometry,
                    capacity,
                    target,
                    draft,
                    context_tokens=context_tokens,
                    greedy_only=True,
                    paged_attention=paged_attention,
                )
            root.activate(
                state_memory_floor_bytes=memory_floor_bytes if not token_pages else None
            )
            yield root
        finally:
            if root is not None:
                torch.npu.synchronize(device)
                root.close()
            torch.distributed.destroy_process_group()
