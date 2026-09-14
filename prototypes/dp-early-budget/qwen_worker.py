"""Qwen MoE protocol experiment: native serving, no DeepSeek patch installation."""

import os
from vllm_ascend.worker.worker import NPUWorker
from strengthen_dsv4.compat import check_runtime
from early_worker import EarlyBudgetMixin


class QwenControlWorker(NPUWorker):
    def __init__(self, vllm_config, *args, **kwargs):
        check_runtime()
        p = vllm_config.parallel_config
        assert vllm_config.model_config.hf_config.model_type == "qwen3_moe"
        assert (
            p.tensor_parallel_size,
            p.data_parallel_size,
            p.pipeline_parallel_size,
        ) == (1, 2, 1)
        assert p.enable_expert_parallel and p.data_parallel_size_local == 2
        assert vllm_config.speculative_config is None
        assert vllm_config.scheduler_config.async_scheduling
        assert not vllm_config.cache_config.enable_prefix_caching
        super().__init__(vllm_config, *args, **kwargs)

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        folder = os.environ.get("EARLY_BUDGET_PROFILE")
        if folder:
            from qwen_profile import install

            install(self.model_runner, folder)
        return result


class QwenEarlyWorker(EarlyBudgetMixin, QwenControlWorker):
    """Only the EngineCore-to-worker budget handoff differs from the control."""
