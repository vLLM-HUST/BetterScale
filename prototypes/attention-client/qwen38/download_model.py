"""Download pinned public snapshots on the execution host; never copy weights.

Only two original shards are required to recover exact PLE integer metadata.
This is not a complete original-model snapshot.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from modelscope import snapshot_download
from safetensors import safe_open

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
a = p.parse_args()
a.directory.mkdir(parents=True, exist_ok=True)
for model, revision, leaf, patterns in (
    (
        "Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp",
        "c76aa96a5e730de35211d1b0b3d05cc3c67efae5",
        "Qwen3.8-Flash-Next-w8a8-mtp",
        None,
    ),
    (
        "Qwen/Qwen3.8-Flash-Next",
        "2741eec155d03a8ce151b993ccce1a7b1e398d6b",
        "Qwen3.8-Flash-Next-ple",
        [
            "config.json",
            "model.safetensors.index.json",
            "model-00005-of-00131.safetensors",
            "model-00037-of-00131.safetensors",
        ],
    ),
):
    root = a.directory / leaf
    snapshot_download(
        model,
        revision=revision,
        local_dir=str(root),
        allow_patterns=patterns,
        max_workers=6,
    )
    tensors = 0
    for shard in root.glob("*.safetensors"):
        with safe_open(shard, framework="np") as f:
            tensors += len(f.keys())
    if patterns is None:
        index = json.loads(
            (root / "quant_model_weights.safetensors.index.json").read_text()
        )["weight_map"]
        assert all((root / shard).is_file() for shard in set(index.values()))
        assert tensors == 222866, tensors
    receipt = dict(
        model=model,
        revision=revision,
        directory=str(root),
        allow_patterns=patterns,
        safetensors_tensors=tensors,
        completed=datetime.now(timezone.utc).isoformat(),
        validation="SDK pinned download completed; safetensors headers readable; quantized index files present",
    )
    (a.directory / (leaf + ".receipt.json")).write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
