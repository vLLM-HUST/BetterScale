"""Pinned native MoE baseline with only the already-qualified SD/V1 ABI bridge.

No BetterScale graph, numerical, weight-layout or scheduler optimization.
"""
from vllm_ascend.worker.worker import NPUWorker


class Worker(NPUWorker):
    def __init__(self, vllm_config, *args, **kwargs):
        from mamba_abi import install
        from betterscale.models.qwen import check_runtime

        hf = vllm_config.model_config.hf_text_config
        if (hf.model_type, hf.num_hidden_layers, hf.hidden_size) != (
            "qwen3_5_moe_text", 40, 2048
        ):
            raise ValueError("This baseline is only Qwen3.5-35B-A3B")
        check_runtime()
        install()
        super().__init__(vllm_config, *args, **kwargs)
