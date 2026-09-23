"""Benchmark-only greedy MTP2 adapter; TP2 leaf passed, serving not yet qualified.

Reuse the pinned core synthetic greedy kernel, not a fabricated token-count loop.
Ascend V1 ignores core synthetic settings; explicitly wire that missing path.
"""
import hashlib
import importlib
import json
from pathlib import Path
import torch


def sample(draft, cumulative, target, bonus, uniforms, conditional, max_spec_len):
    from vllm.v1.sample.rejection_sampler import rejection_greedy_sample_kernel
    assert 0 <= max_spec_len <= 2 and conditional.shape == (2,)
    assert draft.ndim == target.ndim == uniforms.ndim == cumulative.ndim == 1
    assert draft.numel() == target.numel() == uniforms.numel()
    assert all(x.device == draft.device and x.is_contiguous()
               for x in (draft, cumulative, target, bonus, uniforms, conditional))
    output = torch.full((cumulative.numel(), max_spec_len+1), -1,
                        dtype=torch.int32, device=draft.device)
    rejection_greedy_sample_kernel[(cumulative.numel(),)](
        output, cumulative, draft, target, bonus, None, max_spec_len,
        uniforms, conditional, SYNTHETIC_MODE=True)
    return output


def install(config):
    import vllm_ascend.sample.rejection_sampler as ascend
    previous = getattr(ascend, '_betterscale_synthetic_rates', None)
    spec = config.speculative_config
    if spec is None or spec.rejection_sample_method != 'synthetic':
        if previous is not None:
            raise RuntimeError('Cannot reuse a synthetic benchmark process for real sampling')
        return
    assert spec.method == 'mtp' and spec.num_speculative_tokens == 2
    assert config.parallel_config.tensor_parallel_size == 2
    assert spec.synthetic_acceptance_rates is not None
    from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates
    from vllm.distributed.parallel_state import get_tp_group
    from vllm_ascend.ascend_config import get_ascend_config
    identity = tuple(spec.synthetic_acceptance_rates)
    if previous is not None:
        if previous != identity:
            raise RuntimeError('Synthetic rates changed; start a fresh worker')
        return
    pins = json.loads(Path(__file__).with_name('synthetic_pins.json').read_text())
    for name, expected in pins.items():
        module = importlib.import_module(name[:-3].replace('/', '.'))
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Unqualified synthetic sampler source: {name}')
    rates = unconditional_to_conditional_rates(spec.synthetic_acceptance_rates)
    device_rates = None
    def synthetic(draft_token_ids, num_draft_tokens, max_spec_len, cu_num_draft_tokens,
                  draft_probs, target_logits_or_tuple, bonus_token_ids,
                  sampling_metadata, **kwargs):
        nonlocal device_rates
        assert sampling_metadata.all_greedy, 'benchmark adapter only qualifies greedy requests'
        assert all(0 <= n <= 2 for n in num_draft_tokens)
        assert sum(num_draft_tokens) == draft_token_ids.numel()
        ac = get_ascend_config()
        assert not ac.rejection_sampler_config.enable_block_verify
        assert not ac.rejection_sampler_config.enable_entropy_verify
        logits = target_logits_or_tuple
        if isinstance(logits, tuple):
            logits, indices = logits
            assert indices is None, 'selected-vocabulary path is not qualified'
        target = ascend.greedy_sample(logits) if ac.enable_reduce_sample else logits.argmax(-1)
        if device_rates is None:
            device_rates = torch.tensor(rates, dtype=torch.float32, device=logits.device)
        uniforms = torch.rand(draft_token_ids.numel(), dtype=torch.float32, device=logits.device)
        start = 0
        for i, n in enumerate(num_draft_tokens):
            if i in sampling_metadata.generators:
                uniforms[start:start+n].uniform_(generator=sampling_metadata.generators[i])
            start += n
        # Synthetic decisions must be identical on all TP ranks, even if their
        # earlier RNG consumption differed. Both comparison arms use this adapter.
        uniforms = get_tp_group().broadcast(uniforms, src=0)
        return sample(draft_token_ids, cu_num_draft_tokens, target, bonus_token_ids,
                      uniforms, device_rates, max_spec_len)
    ascend.rejection_sample = synthetic
    ascend._betterscale_synthetic_rates = identity
