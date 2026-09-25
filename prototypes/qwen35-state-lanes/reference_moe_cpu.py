"""Independent Transformers cold target oracle; no donor or accelerator imports."""

import json
import os
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM

path = Path("/workspace/models/Qwen3.5-35B-A3B")
config = Qwen3_5MoeTextConfig(
    **json.loads((path / "config.json").read_text())["text_config"]
)
config._attn_implementation = "eager"
torch.set_default_dtype(torch.bfloat16)
with torch.device("meta"):
    model = Qwen3_5MoeForCausalLM(config).eval()
weights = {}
files = sorted(
    set(
        json.loads((path / "model.safetensors.index.json").read_text())[
            "weight_map"
        ].values()
    )
)
for f in files:
    with safe_open(str(path / f), framework="pt", device="cpu") as reader:
        for name in reader.keys():
            if name.startswith("model.language_model."):
                weights[name.replace("model.language_model.", "model.", 1)] = (
                    reader.get_tensor(name)
                )
            elif name == "lm_head.weight":
                weights[name] = reader.get_tensor(name)
model.load_state_dict(weights, strict=True, assign=True)
del weights
model.model.rotary_emb = type(model.model.rotary_emb)(config, device="cpu")
assert not any(t.is_meta for t in model.buffers())
tokens = [9707, 11, 358, 1079, 264, 1786, 13]
with torch.inference_mode():
    result = model(torch.tensor([tokens]), output_hidden_states=True, use_cache=False)
out = {
    "tokens": tokens,
    "layers": [x.cpu() for x in result.hidden_states],
    "hidden": result.hidden_states[-1].cpu(),
    "logits": result.logits.cpu(),
}
torch.save(out, Path(os.environ["CAPSULE"]) / "reference.pt")
print(
    json.dumps(
        {
            "next_tokens": out["logits"].argmax(-1).tolist(),
            "reference": "Transformers eager BF16 CPU full-prefix",
        }
    )
)
