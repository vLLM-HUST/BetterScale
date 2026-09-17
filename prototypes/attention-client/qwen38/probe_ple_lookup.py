"""CPU oracle: exact native last-position PLE, EOS boundaries and broad shards."""

import json
import time
from types import SimpleNamespace

import torch
from livemodule.llm.qwen38.ple_table import Qwen38HostPLETable
from ple_lookup import lookup_last

torch.set_num_threads(2)
torch.manual_seed(1709)
sizes = torch.tensor([5003, 5009, 5011, 5021, 5023, 5039, 5051, 5059])
offsets = torch.cat((torch.zeros(1, dtype=torch.long), sizes.cumsum(0)[:-1]))
rows = 100
shards = {
    i: torch.randn(rows, 128, dtype=torch.bfloat16)
    for i in range((int(sizes.sum()) + rows - 1) // rows)
}
table = SimpleNamespace(
    require_ready=lambda: None,
    eos_token_id=0,
    _metadata=dict(
        layer_multipliers=torch.tensor([1, 101, 1009, 10007, 100003]),
        ngram_heads_vocab_sizes=sizes,
        ngram_heads_offsets=offsets,
    ),
    contract=SimpleNamespace(heads_per_ngram=2, ple_head_dim=128),
    plan=SimpleNamespace(ple_heads=SimpleNamespace(start=0, stop=8)),
    _shards=shards,
    rows_per_shard=rows,
    required_shard_indices=tuple(shards),
)
results = []
for first, stop in ((0, 8), (2, 4)):
    table.plan.ple_heads = SimpleNamespace(start=first, stop=stop)
    for n in (1, 32, 1024):
        tokens = torch.randint(0, 100000, (n, 5))
        tokens[::3, 2] = 0
        tokens[::7, -1] = 0
        begin = time.perf_counter()
        expected = Qwen38HostPLETable.lookup(table, tokens)[:, -1]
        old = time.perf_counter() - begin
        begin = time.perf_counter()
        actual = lookup_last(table, tokens)
        new = time.perf_counter() - begin
        assert torch.equal(expected, actual)
        results.append(
            dict(
                rows=n,
                heads=[first, stop],
                native_seconds=old,
                grouped_last_seconds=new,
                exact=True,
            )
        )
print(
    json.dumps(
        dict(
            scope="CPU synthetic shards; not full-checkpoint throughput",
            results=results,
        ),
        indent=2,
    )
)
