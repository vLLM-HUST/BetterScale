"""The only public entry: vllm serve ... --worker-cls strengthen_dsv4.worker.Worker."""

import logging

from .compat import check_runtime
from .config import PATCH_IDS, DP_PATCH_IDS, validate_worker_config
from vllm_ascend.worker.worker import NPUWorker

log = logging.getLogger(__name__)


class Worker(NPUWorker):
    # 选择本类即启用补丁；不用私有 profile、环境变量或服务启动器。
    # 先检查 pinned 私有 API 与配置，再安装初始化前必须生效的 target hooks。
    # 不替用户修改 CANN、动态库路径、HCCL、allocator、端口或 KV 预算。
    def __init__(self, vllm_config, *args, **kwargs):
        check_runtime()
        validate_worker_config(vllm_config)
        from .patches import compat_lcm, target_full

        compat_lcm.install()
        self._native_dp = vllm_config.parallel_config.tensor_parallel_size == 1
        if self._native_dp:
            from .patches import async_decode

            target_full.install(native_dsa=True)
            async_decode.install_capture()
        else:
            from .patches import ordered_replay, async_decode

            target_full.install()
            # Keep ordered large-prefill fallback underneath decode banks.
            # Both hooks precede capture; warmup must not overwrite them.
            ordered_replay.install_capture()
            async_decode.install_capture()
        super().__init__(vllm_config, *args, **kwargs)

    # 原生模型、KV、输入缓冲和 warmup 完成后，只在当前 worker 安装执行补丁。
    # 不依赖激活 RPC，不复制整个 KV 池，也不在生产路径写实验收据。
    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        from .patches import cross_step

        if self._native_dp:
            # Native TP1 DSA / DP8 keeps the eager DSpark path measured here.
            # Do not silently compose the TP-only split-draft patch with it.
            from .patches import async_decode

            cross_step.install(self, native_dsa=True, max_requests=2)
            async_decode.install(self)
            patches = DP_PATCH_IDS
        else:
            from .patches import split_draft, ordered_replay, qli_cpu, async_decode

            split_draft.install(self)
            cross_step.install(self)
            ordered_replay.install(self)
            qli_cpu.install(self)
            async_decode.install(self)
            patches = PATCH_IDS
        log.info("strengthen-dsv4 rank=%s READY patches=%s", self.rank, patches)
        return result
