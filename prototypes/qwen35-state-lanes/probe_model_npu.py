"""Admitted small-model eager target gate, independent of native KV/runner."""

import json
import os
from datetime import timedelta
from pathlib import Path
import torch
import torch_npu  # noqa: F401
from vllm.engine.arg_utils import EngineArgs
from vllm.config import set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm_ascend.utils import enable_custom_op
from betterscale.live import LiveRuntime, TorchStateBackend, live_runtime
from betterscale.live.llm.qwen35 import Capacity, Geometry
from betterscale.live.llm.qwen35.execution import QwenExecutionRoot
from betterscale.live.llm.qwen35.loader import load_models
from betterscale.live.arch.ascend.graph import ACLGraphBackend
from betterscale.live.llm.qwen35.graphs import QwenLiveLLMRoot

use_graphs = os.environ.get("PROBE_GRAPHS") == "1"
Root = QwenLiveLLMRoot if use_graphs else QwenExecutionRoot

run = Path(os.environ["CAPSULE"])
model = Path("/workspace/models/Qwen3.5-0.8B")
assert enable_custom_op()
torch.npu.set_device(0)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

init_device_properties_triton()
torch.set_default_dtype(torch.bfloat16)
config = EngineArgs(
    model=str(model),
    dtype="bfloat16",
    max_model_len=4096,
    max_num_seqs=2,
    enforce_eager=True,
    enable_prefix_caching=False,
    speculative_config={"method": "mtp", "num_speculative_tokens": 2},
).create_engine_config()
with set_current_vllm_config(config):
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        backend="hccl",
        distributed_init_method="file://" + str(run / "distributed-init"),
        timeout=timedelta(seconds=90),
    )
    initialize_model_parallel(backend="hccl")
    target, draft = load_models(config, model, "npu")
    print("WEIGHTS_COVERED", flush=True)
    geometry = Geometry.from_config(json.loads((model / "config.json").read_text()))
    with live_runtime(
        LiveRuntime(
            device="npu",
            graph_backend=ACLGraphBackend(device="npu") if use_graphs else None,
            state_backend=TorchStateBackend("npu", memory_budget_bytes=512 << 20),
        )
    ):
        root = Root(
            geometry, Capacity(2, 5, token_pages=32), target, draft
        )
    root.activate()
    assert type(root) is Root
    assert len(list(root.named_graphs())) == (4 if use_graphs else 0)
    print("ROOT_ACTIVE", type(root).__name__, len(list(root.named_graphs())), flush=True)
    with torch.inference_mode():
        tokens = [9707, 11, 358, 1079, 264, 1786, 13]
        outputs = []
        for pos, token in enumerate(tokens):
            hidden, logits = root.target_step(
                [token], seat=0, position=pos, slots=list(range(pos + 1))
            )
            outputs.append({"hidden": hidden.cpu(), "logits": logits.cpu()})
            print(
                json.dumps(
                    {"position": pos, "next_token": int(logits.argmax(-1).item())}
                ),
                flush=True,
            )
        torch.save({"tokens": tokens, "outputs": outputs}, run / "target.pt")
    root.close()
    print("TARGET_COMPLETE", flush=True)
    if os.environ.get("PROBE_GENERATION") == "1":
        # Two resident seats make empty-first, hot-hit and eviction observable.
        with live_runtime(
            LiveRuntime(
                device="npu",
                graph_backend=ACLGraphBackend(device="npu") if use_graphs else None,
                state_backend=TorchStateBackend("npu", memory_budget_bytes=512 << 20),
            )
        ):
            root = Root(
                geometry, Capacity(2, 2, token_pages=32), target, draft
            )
        root.activate()
        baseline = root.generate(tokens, 12, speculative=False)
        root.close()
        root.activate()
        a = root.generate(tokens, 12)
        assert a["token_ids"] == baseline["token_ids"], ("mtp-vs-target", a, baseline)
        kept = [
            (leaf.conv.tensor[0].clone(), leaf.recurrent.tensor[:3].clone())
            for leaf in root.target.values()
            if hasattr(leaf, "recurrent")
        ]
        b = root.generate([785, 279, 3363, 374], 4)
        assert b["seat"] == 1
        for (conv, recurrent), leaf in zip(
            kept,
            (s for s in root.target.values() if hasattr(s, "recurrent")),
            strict=True,
        ):
            assert torch.equal(conv, leaf.conv.tensor[0])
            assert torch.equal(recurrent, leaf.recurrent.tensor[:3])
        del kept
        continuation = tokens + a["token_ids"] + [271]
        c = root.generate(continuation, 8)
        assert c["seat"] == 0 and c["cached_tokens"] == len(tokens) + len(
            a["token_ids"]
        )
        d = root.generate([16, 10, 17, 28], 4)
        assert d["seat"] == 1 and d["cached_tokens"] == 0
        root.close()
        root.activate()
        cold = root.generate(continuation, 8, speculative=False)
        assert c["token_ids"] == cold["token_ids"], ("warm-vs-cold", c, cold)
        root.close()
        receipt = {
            "baseline": baseline,
            "mtp": a,
            "unrelated": b,
            "warm": c,
            "eviction": d,
            "cold": cold,
            "passed": True,
            "root_type": type(root).__name__,
            "graph_names": [name for name, _ in root.named_graphs()],
            "scope": "real-weight TP1 owned graphs"
            if use_graphs
            else "real-weight synchronous eager TP1",
        }
        (run / "generation.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
