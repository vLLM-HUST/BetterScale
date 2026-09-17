"""Full-size attention-role ownership gate on meta; no NPU or weight payload."""

import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from livemodule import LiveRuntime, live_runtime
from livemodule.arch.binding import ArchBindings
from livemodule.llm.configuration import set_current_vllm_config
from livemodule.llm.qwen38 import Qwen38Config
from livemodule.llm.qwen38.moe import ArchQwen38MoE
from livemodule.llm.runtime_support import set_default_torch_dtype
from tests.test_qwen38_root_surface import _MetaArchitecture, _ModelConfig
from attention import AttentionRoot, RemoteMoE
from weights import MODEL


class RemoteMeta(_MetaArchitecture):
    bindings = ArchBindings(
        {**_MetaArchitecture.bindings.classes, ArchQwen38MoE: RemoteMoE}
    )


hf = Qwen38Config.from_pretrained(MODEL)
index = json.loads((MODEL / "quant_model_weights.safetensors.index.json").read_text())[
    "weight_map"
]
records = []
for rank in (0, 1):
    cfg = NS(
        model_config=_ModelConfig(
            hf_config=hf,
            hf_text_config=hf.get_text_config(),
            dtype=torch.bfloat16,
            max_model_len=4096,
            model=str(MODEL),
            enable_return_routed_experts=False,
            enforce_eager=False,
        ),
        cache_config=NS(block_size=64, cache_dtype="auto"),
        scheduler_config=NS(max_num_batched_tokens=32, max_num_seqs=1),
        parallel_config=NS(),
        compilation_config=NS(),
        speculative_config=None,
        device_config=NS(device=torch.device("meta")),
        load_config=NS(device=None, load_format="safetensors"),
        additional_config={},
        quant_config=None,
    )
    tp = NS(rank_in_group=rank, world_size=2, all_reduce=lambda x: x)
    with ExitStack() as stack:
        stack.enter_context(
            live_runtime(LiveRuntime(architecture=RemoteMeta(), device="meta"))
        )
        stack.enter_context(set_current_vllm_config(cfg))
        stack.enter_context(set_default_torch_dtype(torch.bfloat16))
        stack.enter_context(torch.device("meta"))
        stack.enter_context(
            patch("livemodule.llm.qwen38.causal_lm.get_tp_group", return_value=tp)
        )
        stack.enter_context(
            patch(
                "livemodule.llm.qwen38.causal_lm.get_pp_group",
                return_value=NS(world_size=1),
            )
        )
        stack.enter_context(
            patch(
                "livemodule.llm.qwen38.causal_lm.prepare_qwen38_qsa_island",
                return_value=NS(all_reduce=lambda x: x),
            )
        )
        for mod in ("linear_base", "linear_parallel", "vocab"):
            stem = "livemodule.llm.layers." + mod
            stack.enter_context(
                patch(stem + ".get_tensor_model_parallel_rank", return_value=rank)
            )
            stack.enter_context(
                patch(stem + ".get_tensor_model_parallel_world_size", return_value=2)
            )
        root = AttentionRoot(vllm_config=cfg)
    params = dict(root.named_parameters())
    assert not any(".experts." in name for name in params)
    expected = {
        name
        for name in index
        if root.accepts_checkpoint_tensor(name) and ".ple_embedding." not in name
    }
    assert set(params) == expected, dict(
        missing=sorted(expected - params.keys())[:8],
        extra=sorted(params.keys() - expected)[:8],
    )
    for name, module in root.quantized_qsa.items():
        assert (
            module.weight.dtype == torch.int8
            and module.weight_scale.dtype == torch.float32
        )
    records.append(
        dict(
            rank=rank,
            parameter_tensors=len(params),
            parameter_bytes=sum(t.numel() * t.element_size() for t in params.values()),
            quantized_projections=len(root.quantized_qsa),
            routed_expert_parameters=0,
        )
    )
print(
    json.dumps(dict(status="PASS", scope="CPU meta ownership only", ranks=records)),
    flush=True,
)
