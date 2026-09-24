"""Selective original-BF16 expert loading for explicitly selected model geometry."""
import json
import os
from pathlib import Path
import torch
from safetensors import safe_open

from .model_geometry import GEOMETRY as G
H, M, E, K, LAYERS = G.hidden, G.inner, G.experts, G.topk, G.layers


class Checkpoint:
    def __init__(self, root=None):
        self.root = Path(root if root is not None else os.environ['BETTERSCALE_EXPERT_MODEL'])
        outer = json.loads((self.root / 'config.json').read_text())
        if outer.get('model_type') != 'qwen3_5_moe':
            raise ValueError('expert service requires Qwen3.5 MoE checkpoint')
        self.config = outer['text_config']
        expected = dict(model_type='qwen3_5_moe_text', hidden_size=H,
            moe_intermediate_size=M, num_experts=E, num_experts_per_tok=K,
            num_hidden_layers=40, shared_expert_intermediate_size=512,
            dtype='bfloat16')
        for key, value in expected.items():
            if self.config.get(key) != value:
                raise ValueError(f'unsupported Qwen35 config: {key}')
        self.index = json.loads((self.root / 'model.safetensors.index.json').read_text())['weight_map']
        # Check draft availability before allocating any expert payload.
        for layer in LAYERS:
            for name in expert_names(layer):
                if name not in self.index:
                    raise ValueError(f'Missing expert checkpoint tensor: {name}')

    def tensors(self, names):
        groups = {}
        for name in names:
            groups.setdefault(self.index[name], []).append(name)
        for shard, keys in groups.items():
            with safe_open(self.root / shard, framework='pt', device='cpu') as f:
                for key in keys:
                    yield key, f.get_tensor(key)

    def experts(self, layer, device='npu', bounds=(0, E)):
        if layer not in LAYERS:
            raise ValueError('layer is outside the configured routed layer set')
        first, last = bounds
        assert 0 <= first < last <= E
        prefix = expert_prefix(layer)
        if layer == 40:
            # This snapshot stores target experts fused, draft experts separately.
            shapes = {'gate_proj': (M,H), 'up_proj': (M,H), 'down_proj': (H,M)}
            weights = {name: torch.empty(last-first, *shape, dtype=torch.bfloat16)
                       for name,shape in shapes.items()}
            names = [f'{prefix}{expert}.{projection}.weight'
                     for expert in range(first,last) for projection in shapes]
            for name, tensor in self.tensors(names):
                expert, projection, _ = name[len(prefix):].split('.')
                dest = weights[projection][int(expert)-first]
                assert tensor.dtype == torch.bfloat16 and tensor.shape == dest.shape
                dest.copy_(tensor)
            return tuple(weights[p].to(device) for p in shapes)
        names = [prefix + p for p in ('gate_up_proj', 'down_proj')]
        values = {}
        for name in names:
            with safe_open(self.root / self.index[name], framework='pt', device='cpu') as f:
                values[name] = f.get_slice(name)[first:last].to(device)
        gate_up, down = (values[n] for n in names)
        assert gate_up.shape == (last-first, 2*M, H) and down.shape == (last-first, H, M)
        assert gate_up.dtype == down.dtype == torch.bfloat16
        return gate_up[:, :M], gate_up[:, M:], down


def expert_prefix(layer):
    if layer not in LAYERS:
        raise ValueError('layer outside admitted target/draft catalog')
    return (f'model.language_model.layers.{layer}.mlp.experts.' if layer < 40
            else 'mtp.layers.0.mlp.experts.')


def expert_names(layer):
    prefix = expert_prefix(layer)
    if layer < 40:
        return [prefix + p for p in ('gate_up_proj', 'down_proj')]
    return [f'{prefix}{expert}.{p}.weight' for expert in range(E)
            for p in ('gate_proj', 'up_proj', 'down_proj')]


def plain_experts(weights, x, ids, probs):
    """Independent unfused linear/SILU reference; FP32 weighted sum as author inference."""
    gate, up, down = weights
    routes = torch.empty(x.shape[0], K, H, dtype=x.dtype, device=x.device)
    # Route metadata is copied only in this untimed reference, not the service.
    cpu_ids = ids.cpu()
    for expert in cpu_ids.unique().tolist():
        row, slot = torch.where(cpu_ids == expert)
        row, slot = row.to(x.device), slot.to(x.device)
        inp = x[row]
        y = torch.nn.functional.linear(
            torch.nn.functional.silu(torch.nn.functional.linear(inp, gate[expert]))
            * torch.nn.functional.linear(inp, up[expert]), down[expert])
        routes[row, slot] = y
    return (routes.float() * probs.float().unsqueeze(-1)).sum(1).to(x.dtype)
