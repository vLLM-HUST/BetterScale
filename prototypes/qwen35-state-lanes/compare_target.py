"""Report all-vocabulary errors, cosine and greedy agreement, not text alone."""

import json
import sys
from pathlib import Path
import torch

reference = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
candidate = torch.load(sys.argv[2], map_location="cpu", weights_only=True)
assert reference["tokens"] == candidate["tokens"]
rows = []
for i, output in enumerate(candidate["outputs"]):
    a = output["logits"][0].float()
    b = reference["logits"][0, i].float()
    h = output["hidden"][0].float()
    rh = reference["hidden"][0, i].float()
    rows.append(
        {
            "position": i,
            "greedy": int(a.argmax()),
            "reference_greedy": int(b.argmax()),
            "logit_max": float((a - b).abs().max()),
            "logit_rms": float((a - b).square().mean().sqrt()),
            "hidden_cosine": float(torch.nn.functional.cosine_similarity(h, rh, dim=0)),
        }
    )
print(json.dumps(rows, indent=2))
Path(sys.argv[2]).with_suffix(".comparison.json").write_text(
    json.dumps(rows, indent=2) + "\n"
)
