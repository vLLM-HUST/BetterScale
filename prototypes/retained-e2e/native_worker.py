"""Experiment-only native TP startup compatibility, no performance patches."""

from vllm_ascend.worker.worker import NPUWorker
from strengthen_dsv4.compat import check_runtime


class NativeTPWorker(NPUWorker):
    def __init__(self, vllm_config, *args, **kwargs):
        check_runtime()
        from strengthen_dsv4.patches.compat_lcm import install

        install()
        super().__init__(vllm_config, *args, **kwargs)
