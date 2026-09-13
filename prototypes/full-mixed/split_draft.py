"""Native context ingestion outside a bounded query-only graph; no large padding.

Keep the already-qualified fused small-context K5 route for normal decoding.
The split route uses the current stream for context writes and query reads.
"""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import torch
from torch.profiler import record_function
from vllm.forward_context import get_forward_context
from draft_graph import DraftGraphSet, ExactDraftGraph
from full_draft import graph_metadata


@contextmanager
def query_body(drafter):
    """Only the native merged-call context hook is skipped; restore on failure."""
    original = drafter.build_model_inputs_first_pass
    drafter.build_model_inputs_first_pass = lambda *args, **kwargs: None
    try:
        yield
    finally:
        drafter.build_model_inputs_first_pass = original


class SplitDraftGraphSet(DraftGraphSet):
    def __init__(self, worker):
        super().__init__(worker)
        assert self.drafter.parallel_drafting
        self.decode = DraftGraphSet(worker)
        self.context_calls = self.context_rows = 0
        self.context_max = 0

    def __call__(self, **kwargs):
        d, ctx = self.drafter, get_forward_context()
        assert not ctx.capturing
        count, actual = kwargs['batch_size'], d._dflash_num_context
        assert 1 <= count <= 4 and actual > 0
        if not self.enabled:
            return self.original(**kwargs)
        prefill = any(bool(m.num_prefills) for m in ctx.attn_metadata.values())
        if actual == 6 * count and not prefill and not kwargs.get('is_prefill', False):
            return self.decode(**kwargs)

        # Exactly the unpadded native operation, before ANY query graph capture.
        # No H2D round-trip or host fence is required between these same-stream ops.
        with record_function('strengthen::draft_context_ingest'):
            d.build_model_inputs_first_pass(kwargs['num_input_tokens'], d._context_slot_mapping_buffers)
        self.context_calls += 1
        self.context_rows += actual
        self.context_max = max(self.context_max, actual)
        original_metadata = ctx.attn_metadata
        ctx.attn_metadata = graph_metadata(original_metadata)
        kwargs = dict(kwargs, target_positions=None, is_prefill=False)
        assert kwargs['inputs_embeds'] is None
        if 'multi_steps_attn_metadata' in kwargs:
            kwargs['multi_steps_attn_metadata'] = graph_metadata(kwargs['multi_steps_attn_metadata'])
        modes = {(m.attn_state.name, bool(m.num_prefills)) for m in ctx.attn_metadata.values()}
        assert len(modes) == 1
        mode, prefill = modes.pop()
        key = (count, mode, prefill)
        try:
            with query_body(d), record_function('strengthen::draft_query_graph'):
                if key not in self.entries:
                    entry = ExactDraftGraph(self.worker, self.original, count)
                    entry.query_only = True
                    entry.strict_signature = True
                    entry.reference_kind = 'native-query-after-native-unpadded-context'
                    entry.path = Path(os.environ['FULL_MIXED_OUTPUT']) / f'split-draft-rank{self.worker.rank}-requests{count}-{mode}-{int(prefill)}.json'
                    self.entries[key] = entry
                output = self.entries[key](**kwargs)
                assert self.entries[key].fallbacks == 0
                return output
        finally:
            ctx.attn_metadata = original_metadata

    def receipt(self):
        self.decode.receipt()
        entries = []
        for key, entry in self.entries.items():
            entry.receipt()
            entries.append(dict(key=key,captured=entry.graph is not None,replays=entry.replays,
                                checks=entry.checks,capture_checked=entry.capture_checked,fallbacks=entry.fallbacks))
        result = dict(context_calls=self.context_calls,context_rows=self.context_rows,
                      context_max=self.context_max,context_padding=0,query_banks=entries)
        path = Path(os.environ['FULL_MIXED_OUTPUT']) / f'split-draft-rank{self.worker.rank}.json'
        path.write_text(json.dumps(result,indent=2))
        return result


def install(worker):
    torch.npu.synchronize()
    assert not hasattr(worker, '_exact_draft_graph')
    manager = SplitDraftGraphSet(worker)
    worker.model_runner.drafter._runnable = manager
    worker._exact_draft_graph = manager
    return dict(rank=worker.rank, policy='native-context-plus-query-graph')
