"""Experimental +1-token cache identity for shifted-input Qwen MTP.

A block ending at B includes target hidden[0:B] and draft inputs token[1:B+1].
It is publishable only after token[B] is known. Different continuations MUST
not share that block. No hidden-state side cache or partial state rollback.
"""


def extend_hashes(tokens, previous, block_size, hash_block, extra_keys):
    result = []
    start = len(previous) * block_size
    parent = previous[-1] if previous else None
    while start + block_size < len(tokens):
        end = start + block_size
        extra = tuple(extra_keys(start, end) or ()) + ('betterscale-mtp-lookahead-v1', tokens[end])
        parent = hash_block(parent, tokens[start:end], extra)
        result.append(parent)
        start = end
    return result
