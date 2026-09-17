"""Retain the actual bounded endpoint, not an un-emitted speculative bonus."""

import torch
from livemodule.serve.qwen38.wave import Qwen38GreedyWaveCommit


class RetainedCommit(Qwen38GreedyWaveCommit):
    def forward_bounded(
        self,
        proposal_ids,
        target_hidden,
        target_multi,
        active,
        remaining,
        eos_token_ids,
    ):
        result = super().forward_bounded(
            proposal_ids, target_hidden, target_multi, active, remaining, eos_token_ids
        )
        _, committed, verification, count, eos, length, _, _ = result
        endpoint = (count - 1).clamp_min(0).to(torch.long)
        rows = torch.arange(count.shape[0], device=count.device)
        self.model.publish_speculative_acceptance(endpoint, active & count.gt(0))
        return (
            endpoint,
            committed,
            verification,
            count,
            eos,
            length,
            committed[rows, endpoint],
            target_multi[rows, endpoint],
        )
