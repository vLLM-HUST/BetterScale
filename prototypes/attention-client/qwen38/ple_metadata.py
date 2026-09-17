"""Recover exact PLE integer buffers from the colocated original checkpoint.

The Eco-Tech snapshot stores these hash/index buffers in BF16, losing bits.
Casting them back is not a repair. Do not mutate either snapshot: validate
matching text geometry and the observed lossy cast, then use original integers.
This is a repaired-model execution lane, not unmodified Eco-Tech equivalence.
"""

import json
from pathlib import Path
import torch
from safetensors import safe_open
from weights import MODEL

ORIGINAL = Path("/data/shared_models/Qwen3.8-Flash-Next")
FIELDS = {"layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes"}


class ExactPLEMetadata:
    def __init__(self):
        old = json.loads((ORIGINAL / "config.json").read_text())
        new = json.loads((MODEL / "config.json").read_text())
        if old["text_config"] != new["text_config"]:
            raise ValueError("PLE repair requires matching original text configuration")
        self.index = json.loads(
            (ORIGINAL / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.repaired = []

    def restore(self, name, value):
        if ".ple.ple_embedding." not in name or name.rsplit(".", 1)[-1] not in FIELDS:
            return value
        if value.dtype == torch.int64:
            return value
        if value.device.type != "cpu" or value.dtype != torch.bfloat16:
            raise ValueError("Unexpected PLE metadata representation")
        with safe_open(ORIGINAL / self.index[name], framework="pt", device="cpu") as f:
            exact = f.get_tensor(name)
        if (
            exact.dtype != torch.int64
            or exact.shape != value.shape
            or not torch.equal(exact.to(value.dtype), value)
        ):
            raise ValueError(
                "PLE metadata differs from the observed original-to-BF16 conversion"
            )
        self.repaired.append(name)
        return exact.clone()
