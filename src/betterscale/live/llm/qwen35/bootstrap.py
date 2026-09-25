"""Explicit physical/weight bootstrap, stopping before every native runner seam."""

from contextlib import contextmanager
import json
import os
from pathlib import Path


@contextmanager
def open_model(model_path, *, context_tokens=512, resident_seats=20, token_pages=64):
    import torch
    import torch_npu  # noqa: F401
    from datetime import timedelta
    from vllm.engine.arg_utils import EngineArgs
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm_ascend.utils import enable_custom_op
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    from betterscale.live import LiveRuntime, TorchStateBackend, live_runtime
    from betterscale.live.arch.ascend.graph import ACLGraphBackend
    from betterscale.models.qwen import check_runtime
    from . import Geometry, Capacity
    from .graphs import QwenLiveLLMRoot
    from .loader import load_models

    path = Path(model_path).resolve()
    tp = int(os.environ["WORLD_SIZE"])
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    geometry = Geometry.from_config(
        json.loads((path / "config.json").read_text()), tensor_parallel_size=tp
    )
    capacity = Capacity(1, resident_seats, token_pages=token_pages)
    if not 3 <= context_tokens <= 4096:
        raise ValueError("live context envelope must be in 3..4096")
    check_runtime("live_qwen_pins.json")
    torch.npu.set_device(local_rank)
    if not enable_custom_op():
        raise RuntimeError("required Ascend numerical custom operators are unavailable")
    init_device_properties_triton()
    torch.set_default_dtype(torch.bfloat16)
    config = EngineArgs(
        model=str(path),
        dtype="bfloat16",
        tensor_parallel_size=tp,
        max_model_len=4096,
        max_num_seqs=1,
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
                        device, memory_budget_bytes=3 << 30
                    ),
                    graph_backend=ACLGraphBackend(device=device),
                )
            ):
                root = QwenLiveLLMRoot(
                    geometry,
                    capacity,
                    target,
                    draft,
                    context_tokens=context_tokens,
                    greedy_only=True,
                )
            root.activate()
            yield root
        finally:
            if root is not None:
                torch.npu.synchronize(device)
                root.close()
            torch.distributed.destroy_process_group()
