"""Batch attention isolation and declared shared capacity, CPU only."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from betterscale.live import LiveRuntime, TorchStateBackend, live_runtime
from betterscale.live.llm.qwen35 import Capacity, Geometry
from betterscale.live.llm.qwen35.numerics import attention
from betterscale.live.llm.qwen35.root import QwenStateRoot, state_budget_bytes


def test_attention_batches_preserve_independent_prefix_and_causal_masks():
    class Attention:
        num_heads = 2
        num_kv_heads = 1
        head_dim = 2
        scaling = 2**-0.5

        def qkv_proj(self, hidden):
            return hidden, None

        def _project_qkv_gate(self, qkv, positions):
            return qkv[:, :4], qkv[:, 4:6], qkv[:, 6:8], None

        def o_proj(self, hidden):
            return hidden, None

    rng = torch.Generator().manual_seed(77)
    keys = torch.randn(4, 4, 1, 2, generator=rng)
    values = torch.randn(4, 4, 1, 2, generator=rng)
    hidden = torch.randn(6, 8, generator=rng)
    positions = torch.tensor([1, 2, 3, 2, 3, 4]).expand(3, -1)
    reads = torch.tensor([[0, 1, 2, 3, 0, 0], [4, 5, 6, 7, 8, 0]])
    writes = torch.tensor([1, 2, 3, 6, 7, 8])

    def state():
        return SimpleNamespace(
            key=SimpleNamespace(tensor=keys.clone()),
            value=SimpleNamespace(tensor=values.clone()),
        )

    batch_state = state()
    batched = attention(Attention(), batch_state, hidden, positions, writes, reads)
    single_state = state()
    singles = torch.cat(
        [
            attention(
                Attention(),
                single_state,
                hidden[i : i + 3],
                positions[:, i : i + 3],
                writes[i : i + 3],
                reads[i // 3],
            )
            for i in (0, 3)
        ]
    )
    torch.testing.assert_close(batched, singles)
    torch.testing.assert_close(batch_state.key.tensor, single_state.key.tensor)
    torch.testing.assert_close(batch_state.value.tensor, single_state.value.tensor)


def test_state_budget_charges_actual_declarations_and_bounds_elastic_pages():
    config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "prototypes/qwen35-state-lanes/qwen35-35b-text-config.json"
        ).read_text()
    )
    # Small CPU geometry preserves lane relationships without allocating GBs.
    geometry = replace(
        Geometry.from_config(config, tensor_parallel_size=2),
        layer_types=("linear_attention", "full_attention"),
        hidden_size=8,
        gdn_key_heads=1,
        gdn_value_heads=1,
        gdn_key_dim=2,
        gdn_value_dim=2,
        attention_head_dim=2,
    )
    capacity = Capacity(2, 3, page_tokens=4, token_pages=None)
    budget = state_budget_bytes(geometry, capacity, 5)
    with live_runtime(
        LiveRuntime(
            device="cpu",
            state_backend=TorchStateBackend("cpu", memory_budget_bytes=budget),
        )
    ):
        root = QwenStateRoot(geometry, capacity)
    root.activate()
    assert root.pages.capacity == 5 and root.residents.capacity == 3
    assert (
        sum(
            state.tensor.numel() * state.tensor.element_size()
            for _, state in root.named_states()
        )
        == budget
    )
    root.close()
