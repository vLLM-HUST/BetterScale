"""Small research runtime configuration, borrowed from the owned Qwen38 gate.

No donor global patch or installed runtime modification. The caller binds its
TP2 process group and activates the explicitly selected native closure first.
"""

from types import SimpleNamespace as NS
import torch
from livemodule import LiveRuntime, TorchStateBackend
from livemodule.llm.configuration import CompilationMode, CUDAGraphMode
from livemodule.llm.distributed import capture_vllm_rank_binding
from livemodule.llm.qwen38 import Qwen38Config
from attention import RemoteAscend
from weights import MODEL


class ModelConfig(NS):
    def get_num_experts(self):
        return int(self.hf_text_config.num_experts)


def configure(rank, stage):
    hf = Qwen38Config.from_pretrained(MODEL)
    cfg = NS(
        model_config=ModelConfig(
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
        parallel_config=NS(
            rank=rank,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            use_sequence_parallel_moe=False,
            enable_eplb=False,
            enable_expert_parallel=True,
            enable_dbo=False,
            eplb_config=NS(num_redundant_experts=0),
        ),
        compilation_config=NS(
            static_forward_context={},
            static_all_moe_layers=[],
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
            cudagraph_capture_sizes=[],
            max_cudagraph_capture_size=32,
        ),
        speculative_config=None,
        device_config=NS(device=torch.device("npu:0")),
        load_config=NS(
            device=None,
            load_format="safetensors",
            progress_callback=lambda name, index, total: stage(
                "checkpoint", index=index, total=total
            ),
        ),
        additional_config={
            "mix_placement": False,
            "weight_nz_mode": 1,
            "enable_fused_mc2": 1,
            "qwen38_mapped_qsa_blocks": 0,
            "qwen38_dispatch_ffn_combine": False,
        },
        quant_config=None,
        remote_expert_transport=None,
        remote_expert_priority=0,
    )
    runtime = LiveRuntime(
        architecture=RemoteAscend(),
        device=cfg.device_config.device,
        state_backend=TorchStateBackend(
            cfg.device_config.device, memory_budget_bytes=4 * 2**30
        ),
        rank_binding=capture_vllm_rank_binding(),
    )
    return cfg, runtime
