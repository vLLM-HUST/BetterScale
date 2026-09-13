"""The only public entry: vllm serve ... --worker-cls strengthen_dsv4.worker.Worker."""

import logging

from .compat import check_runtime
from .config import PATCH_IDS, validate_worker_config
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
        target_full.install()
        super().__init__(vllm_config, *args, **kwargs)

    # 原生模型、KV、输入缓冲和 warmup 完成后，只在当前 worker 安装执行补丁。
    # 不依赖激活 RPC，不复制整个 KV 池，也不在生产路径写实验收据。
    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        from .patches import split_draft, cross_step, ordered_replay, qli_cpu

        # 每个模块自带实现与 install；worker 只选择组合与安装时机。
        # split_draft 自己拥有 graph/metadata，无需安装另一份 draft 补丁。
        split_draft.install(self)
        cross_step.install(self)
        ordered_replay.install(self)
        qli_cpu.install(self)
        log.info("strengthen-dsv4 rank=%s READY patches=%s", self.rank, PATCH_IDS)
        return result
