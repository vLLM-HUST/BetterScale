"""Admitted Qwen35B TP2: owned full-model graphs versus owned eager target."""

import json
import os
from pathlib import Path
from datetime import timedelta
import torch
import torch_npu  # noqa: F401
from vllm.engine.arg_utils import EngineArgs
from vllm.config import set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm_ascend.utils import enable_custom_op
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from betterscale.live import LiveRuntime, TorchStateBackend, live_runtime
from betterscale.live.arch.ascend.graph import ACLGraphBackend
from betterscale.live.llm.qwen35 import Geometry, Capacity
from betterscale.live.llm.qwen35.loader import load_models
from betterscale.live.llm.qwen35.execution import QwenExecutionRoot
from betterscale.live.llm.qwen35.graphs import QwenLiveLLMRoot

run = Path(os.environ["CAPSULE"])
rank = int(os.environ["RANK"])
torch.npu.set_device(rank)
assert enable_custom_op()
init_device_properties_triton()
torch.set_default_dtype(torch.bfloat16)
model = Path("/workspace/models/Qwen3.5-35B-A3B")
config = EngineArgs(
    model=str(model),
    dtype="bfloat16",
    tensor_parallel_size=2,
    max_model_len=4096,
    max_num_seqs=2,
    enable_expert_parallel=False,
    enforce_eager=True,
    enable_prefix_caching=False,
    speculative_config={"method": "mtp", "num_speculative_tokens": 2},
).create_engine_config()
with set_current_vllm_config(config):
    init_distributed_environment(
        world_size=2,
        rank=rank,
        local_rank=rank,
        backend="hccl",
        distributed_init_method="env://",
        timeout=timedelta(seconds=180),
    )
    initialize_model_parallel(tensor_model_parallel_size=2, backend="hccl")
    device = f"npu:{rank}"
    target, draft = load_models(config, model, device)
    print(
        json.dumps(
            {
                "rank": rank,
                "weights_covered": True,
                "moe_type": type(target.model.layers[0].mlp.experts).__name__,
            }
        ),
        flush=True,
    )
    geometry = Geometry.from_config(
        json.loads((model / "config.json").read_text()), tensor_parallel_size=2
    )
    capacity = Capacity(2, 2, token_pages=32)

    def make(kind):
        with live_runtime(
            LiveRuntime(
                device=device,
                state_backend=TorchStateBackend(device, memory_budget_bytes=1 << 30),
                graph_backend=ACLGraphBackend(device=device)
                if kind is QwenLiveLLMRoot
                else None,
            )
        ):
            return kind(geometry, capacity, target, draft, greedy_only=True)

    # Teacher force the exact prefix at the observed warm/cold branch.
    sequence = (
        [9707, 11, 358, 1079, 264, 1786, 13] + [220, 16] + [15] * 10 + [271, 2, 220, 16]
    )
    records = {}
    for label, kind, width in [
        ("eager1", QwenExecutionRoot, 1),
        ("graph1", QwenLiveLLMRoot, 1),
        ("eager3", QwenExecutionRoot, 3),
        ("graph3", QwenLiveLLMRoot, 3),
    ]:
        root = make(kind)
        root.activate()
        target.logits_processor.use_all_gather = True
        rows = []
        position = 0
        previous = 1
        while position < len(sequence):
            count = width if position >= 7 and position + width <= len(sequence) else 1
            tokens = sequence[position : position + count]
            hidden, sampled = root.target_step(
                tokens,
                seat=0,
                position=position,
                slots=list(range(position + count)),
                accepted=previous,
            )
            full = target.compute_logits(hidden)
            assert torch.equal(sampled, full.argmax(-1)), (
                "pair argmax differs from full vocabulary"
            )
            values, indices = full.float().topk(8, dim=-1)
            for i in range(count):
                rows.append(
                    {
                        "position": position + i,
                        "ids": indices[i].tolist(),
                        "logits": values[i].tolist(),
                        "hidden": hidden[i].cpu(),
                    }
                )
            position += count
            previous = count
        selected = {
            name: leaf.recurrent.tensor[previous - 1].cpu().clone()
            for name, leaf in root.target.items()
            if hasattr(leaf, "recurrent")
        }
        records[label] = {"rows": rows, "selected_ssm": selected}
        print(
            json.dumps(
                {
                    "rank": rank,
                    "case": label,
                    "greedy": [r["ids"][0] for r in rows],
                    "last_top8": rows[-1]["ids"],
                    "last_logits": rows[-1]["logits"],
                }
            ),
            flush=True,
        )
        root.close()
    torch.save(records, run / f"diagnostics-rank{rank}.pt")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
