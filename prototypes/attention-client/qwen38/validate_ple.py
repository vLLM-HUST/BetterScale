"""Validate the exact three-field PLE repair without loading model weights."""

import json
from safetensors import safe_open
from ple_metadata import ExactPLEMetadata, FIELDS
from weights import MODEL

repair = ExactPLEMetadata()
index = json.loads((MODEL / "quant_model_weights.safetensors.index.json").read_text())[
    "weight_map"
]
for name, shard in index.items():
    if ".ple.ple_embedding." in name and name.rsplit(".", 1)[-1] in FIELDS:
        with safe_open(MODEL / shard, framework="pt", device="cpu") as handle:
            repair.restore(name, handle.get_tensor(name))
assert len(repair.repaired) == 3, repair.repaired
print(
    json.dumps(
        dict(
            status="PASS",
            exact_integer_fields=repair.repaired,
            scope="repaired checkpoint, not untouched Eco-Tech equivalence",
        ),
        indent=2,
    )
)
