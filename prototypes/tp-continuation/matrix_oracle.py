"""One loaded engine: native self-check, packet, producer, metadata ablations.

Diagnostic only: record mismatches but always commit the FIRST native replay's
output/State. Never publish the failing candidate into generation or call this
performance evidence. This avoids reloading all weights for each failed hinge.
"""

import json
import os
from pathlib import Path
import torch
from torch.utils._pytree import tree_flatten, tree_map
from vllm.forward_context import get_forward_context
from vllm.distributed import get_tp_group
from strengthen_dsv4.worker import Worker
from strengthen_dsv4.patches.async_decode import _target
from dp_full_shadow import byte_pools

pair_call = _target.DecodePair.call


def compare(left, right, valid, capacity):
    result = []
    tp = get_tp_group()
    for x, y in zip(tree_flatten(left)[0], tree_flatten(right)[0]):
        if not isinstance(x, torch.Tensor):
            continue
        n = (
            valid
            if x.shape[0] == capacity
            else max(0, min(x.shape[0], valid - tp.rank_in_group * x.shape[0]))
        )
        x, y = x[:n], y[:n]
        result.append(
            dict(
                shape=list(x.shape),
                equal=torch.equal(x, y),
                outside=int((~torch.isclose(x, y, rtol=0.01, atol=1e-6)).sum().cpu()),
                max_diff=float((x.float() - y.float()).abs().max().cpu()) if n else 0,
            )
        )
    return result


def checked(pair, args, kwargs):
    ctx = get_forward_context()
    key = ctx.batch_descriptor
    w = pair.wrapper
    if not hasattr(pair, "reference_catalog"):
        pair.reference_catalog = {}
    if key not in pair.packets[0]:
        old = w.concrete_aclgraph_entries
        try:
            w.concrete_aclgraph_entries = pair.reference_catalog
            _target.original_call(w, *args, **kwargs)
        finally:
            w.concrete_aclgraph_entries = old
        return pair_call(pair, args, kwargs)
    worker = getattr(pair, "matrix_worker", None)
    if worker is None:
        return pair_call(pair, args, kwargs)
    r = worker.model_runner
    mode = worker.matrix_mode
    entry = pair.reference_catalog[key]
    count = worker.matrix_counts.get(mode, 0)
    req = next(iter(ctx.attn_metadata.values())).req_metadata
    valid = int(req.query_start_loc[-1])
    if count >= 12 or valid != 24:
        entry.aclgraph.replay()
        return entry.output
    torch.npu.synchronize()
    pools = byte_pools(r.kv_caches)
    state = pools + [
        m._mtp_hidden_buffer
        for m in r.get_model().modules()
        if hasattr(m, "_mtp_hidden_buffer")
    ]
    before = [x.clone() for x in state]

    def restore(values):
        for x, y in zip(state, values):
            x.copy_(y)

    def capture():
        entry.aclgraph.replay()
        torch.npu.synchronize()
        return tree_map(
            lambda x: x.clone() if isinstance(x, torch.Tensor) else x, entry.output
        )

    native = capture()
    after = [x.clone() for x in state]
    restore(before)
    again = capture()
    row = dict(
        mode=mode,
        sequence=count,
        native_self=compare(native, again, valid, key.num_tokens),
        native_kv_equal=all(torch.equal(x, y) for x, y in zip(pools, after)),
    )
    restore(before)
    actual = pair_call(pair, args, kwargs)
    torch.npu.synchronize()
    row.update(
        pair=compare(actual, native, valid, key.num_tokens),
        pair_kv_equal=all(torch.equal(x, y) for x, y in zip(pools, after)),
    )
    restore(after)
    for dst, src in zip(tree_flatten(entry.output)[0], tree_flatten(native)[0]):
        if isinstance(dst, torch.Tensor):
            dst.copy_(src)
    worker.matrix_rows.append(row)
    worker.matrix_counts[mode] = count + 1
    path = Path(os.environ["DONOR_DP_OUTPUT"]) / f"matrix-rank{worker.rank}.json"
    path.write_text(json.dumps(worker.matrix_rows, indent=2))
    return entry.output


class MatrixWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _target.DecodePair.call = checked

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        self.model_runner.model._decode_pair.matrix_worker = self
        self.matrix_rows = []
        self.matrix_counts = {}
        self.set_matrix_mode("packet")
        return result

    def set_matrix_mode(self, mode):
        assert mode in ("packet", "producer", "metadata")
        self.matrix_mode = mode
        r = self.model_runner
        producer, metadata = r._async_decode
        r._cross_step_bounds.enabled = mode != "packet"
        r._cross_step_bounds.prepare = (
            producer.native if mode == "packet" else producer.prepare
        )
        r._build_attention_metadata = (
            metadata.build if mode == "metadata" else metadata.native
        )
        return dict(mode=mode)
