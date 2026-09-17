"""Load only this server's experts, one layer/shard at a time.

No full-checkpoint host copy, no per-tensor reopening of the huge shard header.
The resident catalog is NZ; temporary ND weights live for one layer only.
"""

from collections import defaultdict

import torch
import torch_npu
from safetensors import safe_open
from weights import Checkpoint


def target_layer(checkpoint, layer, owner, *, owners=4):
    assert owners in (4, 8) and 0 <= owner < owners
    local_experts = 512 // owners
    up = torch.empty(local_experts, 2560, 1280, dtype=torch.int8)
    down = torch.empty(local_experts, 640, 2560, dtype=torch.int8)
    su = torch.empty(local_experts, 1280, dtype=torch.float32)
    sd = torch.empty(local_experts, 2560, dtype=torch.float32)
    shards = defaultdict(list)
    for local in range(local_experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            prefix = (
                f"model.language_model.layers.{layer}.mlp.experts."
                f"{owner * local_experts + local}.{projection}"
            )
            for suffix in ("weight", "weight_scale", "weight_offset"):
                name = f"{prefix}.{suffix}"
                shards[checkpoint.index[name]].append((name, local, projection, suffix))
    for shard, entries in shards.items():
        with safe_open(checkpoint.root / shard, framework="pt", device="cpu") as f:
            for name, local, projection, suffix in entries:
                value = f.get_tensor(name)
                is_down = projection == "down_proj"
                columns = (
                    slice(0, 640) if projection == "gate_proj" else slice(640, 1280)
                )
                if suffix == "weight":
                    dest = down[local] if is_down else up[local, :, columns]
                    assert value.dtype == torch.int8 and value.T.shape == dest.shape
                    dest.copy_(value.T)
                elif suffix == "weight_scale":
                    dest = sd[local] if is_down else su[local, columns]
                    assert value.shape == (dest.numel(), 1)
                    assert torch.isfinite(value).all() and (value > 0).all()
                    dest.copy_(value.flatten())
                elif torch.count_nonzero(value):
                    raise ValueError(f"Unsupported nonzero quantization offset: {name}")
    # Keep only one projection's transient NPU ND storage at a time.
    up_nz = torch_npu.npu_format_cast(up.to("npu"), 29)
    down_nz = torch_npu.npu_format_cast(down.to("npu"), 29)
    return up_nz, down_nz, su.to("npu"), sd.to("npu")


def mtp_layer(checkpoint, owner, *, owners=4):
    assert owners in (4, 8) and 0 <= owner < owners
    local_experts = 512 // owners
    result = []
    for projection in ("gate_up_proj", "down_proj"):
        name = f"mtp.layers.0.mlp.experts.{projection}"
        with safe_open(
            checkpoint.root / checkpoint.index[name], framework="pt", device="cpu"
        ) as f:
            value = f.get_slice(name)[
                owner * local_experts : (owner + 1) * local_experts
            ]
            assert value.dtype == torch.bfloat16
            result.append(
                torch_npu.npu_format_cast(value.transpose(1, 2).to("npu"), 29)
            )
    return *result, None, None


def load(owner, *, model=None, layers=48, mtp=False, owners=4):
    assert owners in (4, 8) and 0 <= owner < owners and 1 <= layers <= 48
    assert not mtp or layers == 48
    checkpoint = Checkpoint() if model is None else Checkpoint(model)
    result = []
    for layer in range(layers):
        result.append(target_layer(checkpoint, layer, owner, owners=owners))
        print(f"owner {owner}: loaded target layer {layer + 1}/{layers}", flush=True)
    if mtp:
        result.append(mtp_layer(checkpoint, owner, owners=owners))
    return result
