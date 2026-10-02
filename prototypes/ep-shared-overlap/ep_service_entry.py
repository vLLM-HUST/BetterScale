"""Experimental TP2/EP2 admission for the real-model overlap gate only.

Preserve every baseline geometry/cache/MTP guard. Do not install this entry in
production or claim general DP/EP product admission from the two-card experiment.
"""
from copy import copy
import os
from betterscale.models import qwen35
from betterscale.qwen35_worker import Worker as BaseWorker
from betterscale.models.qwen35.seat_scheduler import LiveStateScheduler

TARGET_ONLY = os.environ.get("OVERLAP_TARGET_ONLY") == "1"

class TargetOnlyScheduler(LiveStateScheduler):
    pass

if TARGET_ONLY:
    from target_only import install
    install(TargetOnlyScheduler)

class Worker(BaseWorker):
    def __init__(self, *args, **kwargs):
        if TARGET_ONLY:
            from target_only import install_worker
            install_worker()
        super().__init__(*args, **kwargs)

_original = qwen35.validate


def validation_view(config):
    p = config.parallel_config
    if (p.tensor_parallel_size, p.data_parallel_size, p.pipeline_parallel_size,
            p.enable_expert_parallel, p.prefill_context_parallel_size,
            p.decode_context_parallel_size) != (2, 1, 1, True, 1, 1):
        raise ValueError("This experiment admits exactly TP2/DP1/PP1/EP2")
    # Validation-only view: the real runner always receives the original EP config.
    shadow = copy(config)
    shadow.parallel_config = copy(p)
    shadow.parallel_config.enable_expert_parallel = False
    if TARGET_ONLY:
        if config.scheduler_config.scheduler_cls != "ep_service_entry.TargetOnlyScheduler":
            raise ValueError("Target-only worker requires its matching scheduler")
        shadow.scheduler_config = copy(config.scheduler_config)
        shadow.scheduler_config.scheduler_cls = qwen35.STATE_SCHEDULER
    return shadow


def validate(config):
    return _original(validation_view(config))


# EP changes routed expert ownership, not the TP2 vocabulary partition used by
# the existing draft greedy processor. Keep all its object/shape/alias guards.
from betterscale.models.qwen35 import small_fish_runtime
_original_draft = small_fish_runtime.admit_draft


def admit_draft(proposer, target, ascend_config, lmhead_tp):
    shadow = copy(proposer)
    shadow.vllm_config = validation_view(proposer.vllm_config)
    return _original_draft(shadow, target, ascend_config, lmhead_tp)


qwen35.validate = validate
small_fish_runtime.admit_draft = admit_draft
