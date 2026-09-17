"""Read selected Qwen4Exp expert tensors without staging foreign experts.

The downloaded model has per-expert W8A8 target weights and fused BF16 MTP
weights. Keep the two contracts explicit rather than coercing the whole catalog.
"""

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

MODEL = Path(
    os.environ.get("QWEN38_MODEL", "/data/shared_models/Qwen3.8-Flash-Next-w8a8-mtp")
)


class Checkpoint:
    def __init__(self, root=MODEL):
        self.root = Path(root)
        self.index = json.loads(
            (self.root / "quant_model_weights.safetensors.index.json").read_text()
        )["weight_map"]

    def tensor(self, name):
        with safe_open(self.root / self.index[name], framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    def expert(self, layer, expert):
        if not 0 <= layer <= 48 or not 0 <= expert < 512:
            raise ValueError("Expected target layer0..47 or MTP48; expert0..511")
        if layer == 48:
            values = {}
            for projection in ("gate_up_proj", "down_proj"):
                name = f"mtp.layers.0.mlp.experts.{projection}"
                with safe_open(
                    self.root / self.index[name], framework="pt", device="cpu"
                ) as f:
                    values[projection] = f.get_slice(name)[expert]
            assert values["gate_up_proj"].shape == (1280, 2560)
            assert values["down_proj"].shape == (2560, 640)
            assert all(t.dtype == torch.bfloat16 for t in values.values())
            return "bf16", values
        values = {}
        for projection, shape in (
            ("gate_proj", (640, 2560)),
            ("up_proj", (640, 2560)),
            ("down_proj", (2560, 640)),
        ):
            prefix = (
                f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}"
            )
            w = self.tensor(prefix + ".weight")
            scale = self.tensor(prefix + ".weight_scale")
            offset = self.tensor(prefix + ".weight_offset")
            assert w.dtype == torch.int8 and w.shape == shape
            assert scale.shape == offset.shape == (shape[0], 1)
            if torch.count_nonzero(offset):
                raise ValueError(f"Asymmetric weight offset not implemented: {prefix}")
            assert torch.isfinite(scale).all() and (scale > 0).all()
            values[projection] = (w, scale.flatten().float())
        return "w8a8_dynamic", values
