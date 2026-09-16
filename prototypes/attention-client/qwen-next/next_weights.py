"""Layerwise role loader: never stage the full checkpoint on an attention NPU."""

import json
from pathlib import Path
import torch
import torch_npu
from settings import MODEL, LAYERS, REAL


def weights(owner, layers=None):
    catalog = []
    if REAL:
        from safetensors import safe_open

        index = json.loads((Path(MODEL) / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
    for layer in (range(LAYERS) if layers is None else layers):
        # Only one layer has temporary ND weights during NZ conversion.
        up = torch.empty((128, 2048, 1024), dtype=torch.bfloat16, device="npu")
        down = torch.empty((128, 512, 2048), dtype=torch.bfloat16, device="npu")
        if REAL:
            # Open only the shards containing our expert tensors. Host mappings
            # are transient; no client receives these weights.
            names = {}
            for local in range(128):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    name = f"model.layers.{layer}.mlp.experts.{owner*128+local}.{projection}.weight"
                    names.setdefault(index[name], []).append((name, local, projection))
            for shard, entries in names.items():
                with safe_open(
                    str(Path(MODEL) / shard), framework="pt", device="cpu"
                ) as handle:
                    for name, local, projection in entries:
                        value = handle.get_tensor(name).T
                        dest = (
                            down[local]
                            if projection == "down_proj"
                            else up[
                                local, :, : 512 if projection == "gate_proj" else 1024
                            ]
                        )
                        if projection == "up_proj":
                            dest = up[local, :, 512:]
                        dest.copy_(value)
        else:
            torch.manual_seed(812 + layer * 4 + owner)
            for tensor in (up, down):
                for expert in tensor:
                    expert.normal_(0, 0.01)
        catalog.append(
            (torch_npu.npu_format_cast(up, 29), torch_npu.npu_format_cast(down, 29))
        )
    return catalog
