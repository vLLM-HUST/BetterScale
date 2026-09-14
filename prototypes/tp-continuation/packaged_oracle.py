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
    if pair.shadow_runner is not None and not pair.checks:
        audit_sources(pair, args, kwargs)
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
            / f"decode-shadow-rank{r.vllm_config.parallel_config.data_parallel_rank * r.vllm_config.parallel_config.tensor_parallel_size + self.rank}.json"
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


def audit_sources(pair, args, kwargs):
    """Compare frozen capture sources against live metadata without changing them."""
    import dataclasses
    import json
    from strengthen_dsv4.patches.async_decode._packet import CallPacket
    from vllm.distributed import get_tp_group

    ctx = get_forward_context()
    packet = pair.packets[pair.sequence % 2][ctx.batch_descriptor]
    names, seen = {}, set()

    def visit(value, path):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            names[CallPacket.storage_key(value)] = str(path)
        elif dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                visit(getattr(value, field.name), path + (field.name,))
        elif type(value).__name__ == "RopeDataProxy":
            visit(value._data, path + ("rope",))
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, path + (key,))
        elif isinstance(value, (tuple, list)):
            for key, item in enumerate(value):
                visit(item, path + (key,))

    source = (args, kwargs, ctx.attn_metadata)
    visit(source, ())
    current = packet.captured_copies(source)
    old = {CallPacket.storage_key(d): s for d, s in packet.copy_sources}
    rows = []
    for dest, live in current:
        frozen = old[CallPacket.storage_key(dest)]
        rows.append(
            dict(
                path=names.get(CallPacket.storage_key(live)),
                live=CallPacket.storage_key(live),
                frozen=CallPacket.storage_key(frozen),
                equal=torch.equal(live, frozen),
            )
        )
    rank = get_tp_group().rank_in_group
    (Path(os.environ["DONOR_DP_OUTPUT"]) / f"sources-rank{rank}.json").write_text(
        json.dumps(rows, indent=2)
    )


class DummyAcceptanceWorker(AcceptanceWorker):
    """Exact four-layer fixture only; never widen the delivered admission gate."""

    def __init__(self, vllm_config, *args, **kwargs):
        config = vllm_config
        from unittest.mock import patch

        assert config.load_config.load_format == "dummy"
        assert config.model_config.hf_config.num_hidden_layers == 4
        assert config.parallel_config.tensor_parallel_size == 8
        with patch("strengthen_dsv4.worker.validate_worker_config", lambda c: None):
            super().__init__(config, *args, **kwargs)
