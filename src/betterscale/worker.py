"""The only public entry: vllm serve ... --worker-cls betterscale.worker.Worker."""

import logging

from .patches.auto_kv import PhysicalMemoryMixin
from .compat import check_runtime
from .config import PATCH_IDS, DP_PATCH_IDS, validate_worker_config
from vllm_ascend.worker.worker import NPUWorker

log = logging.getLogger("vllm.betterscale.worker")


class Worker(PhysicalMemoryMixin, NPUWorker):
    # 选择本类即启用补丁；不用私有 profile、环境变量或服务启动器。
    # 先检查 pinned 私有 API 与配置，再安装初始化前必须生效的 target hooks。
    # 不替用户修改 CANN、动态库路径、HCCL、allocator 或端口。
    # 未指定固定 KV 字节时，由 auto_kv 按实际物理余量定容。
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
            from .patches import hc_workspace

            hc_workspace.install()
            target_full.install()
        super().__init__(vllm_config, *args, **kwargs)

    # KV 定容只管理试捕获/回收；具体 draft 实现仍由 worker 组合，
    # auto_kv 不导入另一个补丁包。试捕获与最终启动使用同一实现。
    def install_draft_program(self):
        from .patches import split_draft

        split_draft.install(self)

    def prepare_draft_program(self):
        from .patches.split_draft._warmup import prepare

        prepare(self)

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
            from .patches import split_draft, ordered_replay, qli_cpu

            split_draft.install(self)
            cross_step.install(self)
            ordered_replay.install(self)
            qli_cpu.install(self)
            patches = PATCH_IDS
        self.prepare_final_program()
        self.snapshot("ready_after_capture")
        log.info("BetterScale rank=%s READY patches=%s", self.rank, patches)
        return result
