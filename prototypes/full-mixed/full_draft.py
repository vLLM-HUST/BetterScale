"""Bounded DSpark context banks for target prefill/mixed/decode waves.

Experimental: padding must pass the unpadded eager KV oracle on hardware.
This does not change the target budget or the number of generated draft tokens.
"""
from pathlib import Path
import copy
import os
import torch
from vllm.forward_context import get_forward_context
from draft_graph import DraftGraphSet, ExactDraftGraph


def graph_metadata(value):
    if isinstance(value, dict):
        return {k: graph_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [graph_metadata(v) for v in value]
    if type(value).__name__ == 'AscendDSAMetadata':
        result = copy.copy(value)
        # In the pinned DSACP forward, only num_prefills > 0 is consumed.
        # Exact counts were used by the builder already; tensor offsets and
        # req.num_reqs_actual remain untouched. Do not specialize graph bodies
        # on logging-only decode counts or the number of prefill requests.
        result.num_prefills = int(value.num_prefills > 0)
        result.num_decodes = result.num_decode_tokens = 0
        return result
    return value


class FullDraftGraphSet(DraftGraphSet):
    def __init__(self, worker):
        super().__init__(worker)
        assert self.drafter.parallel_drafting
        self.capacity = worker.model_runner.vllm_config.scheduler_config.max_num_batched_tokens
        self.actual_context = None
        self.original_metadata = None

    def reference(self, **kwargs):
        """Compare against real context length, never padded vs padded."""
        d = self.drafter
        capacity = d._dflash_num_context
        ctx = get_forward_context()
        metadata = ctx.attn_metadata
        d._dflash_num_context = self.actual_context
        ctx.attn_metadata = self.original_metadata
        try:
            return self.original(**kwargs)
        finally:
            d._dflash_num_context = capacity
            ctx.attn_metadata = metadata

    def __call__(self, **kwargs):
        if not self.enabled:
            return self.original(**kwargs)
        d = self.drafter
        assert not get_forward_context().capturing
        count, actual = kwargs['batch_size'], d._dflash_num_context
        assert 1 <= count <= 4 and 0 < actual <= self.capacity
        capacity = 6 * count if actual == 6 * count else self.capacity
        # DSpark's parallel _run_merged_draft branch does not consume either
        # argument. Their source shape/mode must not create new graph entries.
        kwargs = dict(kwargs, target_positions=None, is_prefill=False)
        assert kwargs['inputs_embeds'] is None
        assert self.actual_context is None, 'reentrant draft body'
        self.actual_context = actual
        ctx = get_forward_context()
        self.original_metadata = ctx.attn_metadata
        ctx.attn_metadata = graph_metadata(ctx.attn_metadata)
        if 'multi_steps_attn_metadata' in kwargs:
            kwargs['multi_steps_attn_metadata'] = graph_metadata(kwargs['multi_steps_attn_metadata'])
        try:
            assert capacity <= d._dflash_hidden_states.shape[0]
            assert capacity <= d._context_positions_buffer.shape[0]
            d._dflash_hidden_states[actual:capacity].zero_()
            d._context_positions_buffer[actual:capacity].zero_()
            for slots in d._context_slot_mapping_buffers:
                assert capacity <= slots.shape[0]
                slots[actual:capacity].fill_(-1)
            d._dflash_num_context = capacity
            modes = {(m.attn_state.name, bool(m.num_prefills)) for m in ctx.attn_metadata.values()}
            assert len(modes) == 1
            mode, prefill = modes.pop()
            key = (count, capacity, mode, prefill)
            if key not in self.entries:
                entry = ExactDraftGraph(self.worker, self.original, count,
                                        capacity, self.reference)
                entry.strict_signature = True
                entry.allow_addressed_signed_zero = True
                entry.path = Path(os.environ['FULL_MIXED_OUTPUT']) / (
                    f'full-draft-rank{self.worker.rank}-requests{count}-context{capacity}-{mode}-{int(prefill)}.json')
                self.entries[key] = entry
            entry = self.entries[key]
            output = entry(**kwargs)
            assert entry.fallbacks == 0, 'FULL draft bank signature changed'
            return output
        finally:
            d._dflash_num_context = actual
            self.actual_context = None
            ctx.attn_metadata = self.original_metadata
            self.original_metadata = None


def install(worker):
    torch.npu.synchronize()
    assert not hasattr(worker, '_exact_draft_graph'), 'install one draft policy only'
    manager = FullDraftGraphSet(worker)
    worker.model_runner.drafter._runnable = manager
    worker._exact_draft_graph = manager
    return dict(rank=worker.rank, context_capacity=manager.capacity,
                policy='bounded-all-mode-draft-experimental')
