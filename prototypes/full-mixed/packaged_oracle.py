"""Same-state qualification of the packaged Worker, never a deployment entry."""

import os
from pathlib import Path
import torch
from vllm.forward_context import get_forward_context
from strengthen_dsv4.worker import Worker
from strengthen_dsv4.patches.async_decode import _target, _producer
from decode_shadow import device_fields, DecodeShadow
from pingpong_shadow import check

original_pair = _target.DecodePair.call


def audited_pair(pair, args, kwargs):
    ctx = get_forward_context()
    key = ctx.batch_descriptor
    if not hasattr(pair, "reference_catalog"):
        pair.reference_catalog = {}
        pair.checks = []
        pair.shadow_runner = None
    if key not in pair.packets[0]:
        w = pair.wrapper
        entries = w.concrete_aclgraph_entries
        try:
            w.concrete_aclgraph_entries = pair.reference_catalog
            _target.original_call(w, *args, **kwargs)
        finally:
            w.concrete_aclgraph_entries = entries
    if pair.shadow_runner is not None and len(pair.checks) < 48:
        return check(pair, lambda: original_pair(pair, args, kwargs))
    return original_pair(pair, args, kwargs)


class AcceptanceWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _target.DecodePair.call = audited_pair

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        r = self.model_runner
        r.model._decode_pair.shadow_runner = r
        producer, _ = r._async_decode
        producer.checks = []
        producer.path = (
            Path(os.environ["DONOR_DP_OUTPUT"])
            / f"decode-shadow-rank{r.vllm_config.parallel_config.data_parallel_rank}.json"
        )
        original_replay = _producer.Slot.replay

        def observed_replay(slot):
            if len(producer.checks) < 12:
                fields = device_fields(r, slot.n)
                producer._check_fields = fields, [x.clone() for x in fields]
            return original_replay(slot)

        _producer.Slot.replay = observed_replay
        original_prepare = r._cross_step_bounds.prepare

        def checked_prepare(schedule, counts):
            producer._check_fields = None
            output = original_prepare(schedule, counts)
            if producer._check_fields is not None:
                fields, before = producer._check_fields
                DecodeShadow.check(
                    producer,
                    schedule,
                    counts,
                    fields,
                    before,
                    output,
                    r.input_batch.num_reqs,
                )
            return output

        r._cross_step_bounds.prepare = checked_prepare
        return result
