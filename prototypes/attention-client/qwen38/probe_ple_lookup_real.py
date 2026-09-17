"""CPU-only real-shard oracle; checkpoint paths follow the runtime environment."""

import argparse
import json, os, time, resource
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from safetensors import safe_open
from livemodule.llm.qwen38.ple_table import Qwen38HostPLETable
from ple_lookup import lookup_last

p = argparse.ArgumentParser()
p.add_argument("--trace-plan", required=True, type=Path)
a = p.parse_args()
torch.set_num_threads(2)
m = Path(os.environ["QWEN38_MODEL"])
old = Path(os.environ["QWEN38_ORIGINAL_MODEL"])
idx = json.load(open(old / "model.safetensors.index.json"))["weight_map"]
meta = {}
for name, file in idx.items():
    if ".ple.ple_embedding." in name and name.rsplit(".", 1)[-1] in (
        "layer_multipliers",
        "ngram_heads_offsets",
        "ngram_heads_vocab_sizes",
    ):
        with safe_open(old / file, framework="pt", device="cpu") as f:
            meta[name.rsplit(".", 1)[-1]] = f.get_tensor(name)
shards = {}
for file in sorted(m.glob("quant_model_weights-*.safetensors")):
    with safe_open(file, framework="pt", device="cpu") as f:
        for name in f.keys():
            if ".ple.ple_embedding.ngram_embedding.shard_" in name:
                shards[int(name.rsplit("_", 1)[-1].split(".")[0])] = f.get_tensor(name)
assert len(shards) == 128, len(shards)
rows, dim = shards[0].shape
r = json.load(open(m / "config.json"))["text_config"]
table = NS(
    require_ready=lambda: None,
    eos_token_id=151645,
    _metadata=meta,
    contract=NS(heads_per_ngram=r["heads_per_ngram"], ple_head_dim=dim),
    plan=NS(ple_heads=NS(start=0, stop=8)),
    _shards=shards,
    rows_per_shard=rows,
    required_shard_indices=tuple(shards),
)
plan = json.loads(a.trace_plan.read_text())
tokens = []
for s in plan["conversations"][:20]:
    ids = [table.eos_token_id] * 2 + s["turns"][0]["prompt_delta_token_ids"][:51]
    for i in range(2, len(ids)):
        tokens.append(ids[i - 2 : i + 1])
tokens = torch.tensor(tokens, dtype=torch.long)
print(json.dumps(dict(rows=rows, dim=dim, requests=list(tokens.shape))), flush=True)
ref = None
for label, fn in [
    ("native", lambda: Qwen38HostPLETable.lookup(table, tokens)[:, -1]),
    ("grouped-last", lambda: lookup_last(table, tokens)),
    ("native-warm", lambda: Qwen38HostPLETable.lookup(table, tokens)[:, -1]),
    ("grouped-last-warm", lambda: lookup_last(table, tokens)),
]:
    before = resource.getrusage(resource.RUSAGE_SELF)
    t = time.perf_counter()
    out = fn()
    sec = time.perf_counter() - t
    after = resource.getrusage(resource.RUSAGE_SELF)
    if ref is None:
        ref = out
    assert torch.equal(ref, out)
    print(
        json.dumps(
            dict(
                label=label,
                seconds=sec,
                major_faults=after.ru_majflt - before.ru_majflt,
                minor_faults=after.ru_minflt - before.ru_minflt,
                exact=True,
            )
        ),
        flush=True,
    )
