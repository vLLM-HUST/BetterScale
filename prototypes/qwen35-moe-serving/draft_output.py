"""Keep captured draft padding out of request-owned CPU history publication."""
from functools import wraps


def live_rows(result, count, width):
    if count == 0:  # Warmup/capture has no live requests.
        return result
    if result.ndim != 2 or result.shape[0] < count or result.shape[1] != width:
        raise RuntimeError(f'Invalid draft output {result.shape} for {count} requests/{width} steps')
    return result[:count]


def install():
    from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer as Proposer
    if getattr(Proposer, '_betterscale_live_draft_rows', False):
        return
    original = Proposer._propose

    @wraps(original)
    def propose(self, *args, **kwargs):
        count = self.runner.input_batch.num_reqs
        return live_rows(original(self, *args, **kwargs), count, self.num_speculative_tokens)

    Proposer._propose = propose
    Proposer._betterscale_live_draft_rows = True
