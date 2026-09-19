"""DSV4 lifecycle composition; no Worker subclass or Qwen dependencies."""

import logging

from ..compat import check_runtime
from ..config import PATCH_IDS, DP_PATCH_IDS, validate_worker_config
from ..patches.auto_kv import init_device

log = logging.getLogger("vllm.betterscale.worker")


def validate(config):
    validate_worker_config(config)
    return "dsv4-dp" if config.parallel_config.tensor_parallel_size == 1 else "dsv4-tp"


def check(config):
    check_runtime()


def before_init(worker, config):
    from ..patches import compat_lcm, target_full

    compat_lcm.install()
    worker._native_dp = config.parallel_config.tensor_parallel_size == 1
    if worker._native_dp:
        from ..patches import async_decode

        target_full.install(native_dsa=True)
        async_decode.install_capture()
    else:
        from ..patches import hc_workspace

        hc_workspace.install()
        target_full.install()


def warmed(worker):
    from ..patches import cross_step
    from ..patches.auto_kv import snapshot

    if worker._native_dp:
        from ..patches import async_decode

        cross_step.install(worker, native_dsa=True, max_requests=2)
        async_decode.install(worker)
        patches = DP_PATCH_IDS
    else:
        from ..patches import split_draft, ordered_replay, qli_cpu
        from .dsv4_draft import prepare_final

        split_draft.install(worker)
        cross_step.install(worker)
        ordered_replay.install(worker)
        qli_cpu.install(worker)
        prepare_final(worker)
        patches = PATCH_IDS
    snapshot(worker, "ready_after_capture")
    log.info("BetterScale rank=%s READY patches=%s", worker.rank, patches)


def determine_available_memory(worker, native):
    from ..patches.auto_kv import determine_available_memory as budget
    from .dsv4_draft import capture_trial, retire_trial

    return budget(
        worker, native, capture_trial=capture_trial, retire_trial=retire_trial
    )
