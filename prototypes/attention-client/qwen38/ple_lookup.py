"""Host PLE lookup for mailbox lanes: last history position, grouped shard reads.

Hashing and checkpoint semantics remain owned by LiveInfer. The mailbox consumes
only the last n-gram position; do not fetch embeddings for discarded positions.
One sort replaces one full boolean-mask scan per touched checkpoint shard.
"""

import torch


def gather_rows(shards, flat_ids, rows_per_shard, head_dim):
    if flat_ids.device.type != "cpu" or flat_ids.ndim != 1:
        raise ValueError("PLE row IDs must be a flat CPU tensor")
    exemplar = next(iter(shards.values()))
    output = exemplar.new_empty(flat_ids.numel(), head_dim)
    ordered_ids, order = flat_ids.sort()
    shard_ids = ordered_ids.div(rows_per_shard, rounding_mode="floor")
    unique, counts = shard_ids.unique_consecutive(return_counts=True)
    start = 0
    for shard_id, count in zip(unique.tolist(), counts.tolist()):
        shard = shards.get(shard_id)
        if shard is None:
            raise RuntimeError(f"PLE lookup reached unloaded shard_{shard_id}")
        stop = start + count
        rows = ordered_ids[start:stop] - shard_id * rows_per_shard
        output.index_copy_(0, order[start:stop], shard.index_select(0, rows))
        start = stop
    return output


def lookup_last(table, token_history):
    from livemodule.llm.qwen38.ple_table import qwen38_ngram_ids

    table.require_ready()
    if token_history.device.type != "cpu":
        raise ValueError("PLE token history must remain on the host")
    ids = qwen38_ngram_ids(
        token_history,
        eos_token_id=table.eos_token_id,
        layer_multipliers=table._metadata["layer_multipliers"],
        head_vocab_sizes=table._metadata["ngram_heads_vocab_sizes"],
        head_offsets=table._metadata["ngram_heads_offsets"],
        heads_per_ngram=table.contract.heads_per_ngram,
    )[:, -1:, table.plan.ple_heads.start : table.plan.ple_heads.stop]
    values = gather_rows(
        table._shards,
        ids.reshape(-1),
        table.rows_per_shard,
        table.contract.ple_head_dim,
    )
    return values.view(*ids.shape, table.contract.ple_head_dim).flatten(-2)[:, 0]
