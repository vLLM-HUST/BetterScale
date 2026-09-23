"""Bound merged MTP sampling by requests, not the prefill token envelope.

Installed before load_model creates the ACL runnable. Capture and its eager
warmups must see the same bounded shapes; slicing the returned IDs is too late.
The first/second draft model token envelopes and attention metadata stay intact.
"""
from functools import wraps


def sampling_inputs(proposer, num_input_tokens, batch_size, indices):
    capacity = min(num_input_tokens, proposer.runner.max_num_reqs)
    if capacity <= 0:
        raise ValueError('Draft sampling requires positive token/request capacities')
    live = proposer.runner.input_batch.num_reqs
    if live:
        if not (0 < batch_size <= live <= capacity) or indices.shape != (batch_size,):
            raise ValueError('Live MTP sampling exceeds its admitted request envelope')
        # Do not truncate real inputs, even on eager paths.
        return batch_size, indices
    # The shared FULL key is not uniform-decode-only: a token may belong to a
    # distinct request. Reserve min(tokens, requests), not tokens // MTP width.
    # dummy_run has already zeroed this persistent buffer, including its tail.
    indices = proposer.token_indices_to_sample[:capacity]
    if indices.shape != (capacity,):
        raise ValueError('Draft sampling buffer is smaller than request capacity')
    return capacity, indices


def install():
    from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer as Proposer
    if getattr(Proposer, '_betterscale_request_sampling', False):
        return
    original = Proposer._run_merged_draft

    @wraps(original)
    def run(self, num_input_tokens, batch_size, token_indices_to_sample,
            target_positions, inputs_embeds, multi_steps_attn_metadata,
            num_tokens, is_prefill=None):
        if (self.method != 'mtp' or self.num_speculative_tokens != 2
                or self.extra_slots_per_request != 1 or self.parallel_drafting):
            raise ValueError('Request-bounded sampling is qualified only for serial MTP2')
        batch_size, token_indices_to_sample = sampling_inputs(
            self, num_input_tokens, batch_size, token_indices_to_sample)
        return original(self, num_input_tokens, batch_size, token_indices_to_sample,
                        target_positions, inputs_embeds, multi_steps_attn_metadata,
                        num_tokens, is_prefill)

    Proposer._run_merged_draft = run
    Proposer._betterscale_request_sampling = True
