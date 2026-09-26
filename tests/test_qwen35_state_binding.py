"""The baseline borrows declared independent State without running new math."""

import pytest
import torch
from torch import nn

from betterscale.live import LiveRuntime, TorchStateBackend, live_runtime
from betterscale.live.llm.qwen35.state import (
    AttentionState,
    Capacity,
    GDNState,
    Geometry,
)
from betterscale.models.qwen35.state_binding import BaselineStateRoot


def fixture():
    geometry = Geometry(("linear_attention", "full_attention"), 1, 4, 1, 1, 2, 2, 4, 4)
    capacity = Capacity(2, 3, token_pages=4)
    consumers = {}
    for name, abi in (
        ("model.layers.0.linear_attn", GDNState.binding_abi),
        ("model.layers.1.self_attn.attn", AttentionState.binding_abi),
        ("mtp.layers.0.self_attn.attn", AttentionState.binding_abi),
    ):
        leaf = nn.Module()
        leaf.state_binding_abi = abi
        leaf.kv_cache = torch.tensor([]) if "self_attn" in name else ()
        consumers[name] = leaf
    return geometry, capacity, consumers


def declare(geometry, capacity, consumers):
    with live_runtime(
        LiveRuntime(device="cpu", state_backend=TorchStateBackend("cpu"))
    ):
        return BaselineStateRoot(
            geometry,
            capacity,
            consumers=consumers,
            draft_names={"mtp.layers.0.self_attn.attn"},
        )


def test_native_empty_sentinels_receive_independent_domains():
    geometry, capacity, consumers = fixture()
    root = declare(geometry, capacity, consumers)
    assert all(not state.is_bound for _, state in root.named_states())
    root.activate()
    try:
        tensors = root.kv_caches()
        assert tensors.keys() == consumers.keys()
        for name, value in tensors.items():
            assert consumers[name].kv_cache is value
        conv, recurrent = tensors["model.layers.0.linear_attn"]
        assert conv.shape == (3, 5, 6)
        assert recurrent.shape == (9, 1, 2, 2)
        assert not conv.count_nonzero() and not recurrent.count_nonzero()
        key, value = tensors["model.layers.1.self_attn.attn"]
        assert key.shape == value.shape == (4, 128, 1, 4)
        assert key.data_ptr() != value.data_ptr()
        assert tensors["mtp.layers.0.self_attn.attn"][0].data_ptr() != key.data_ptr()
        assert root.conv_selection.tensor.tolist() == [1, 1, 1]
    finally:
        root.close()
    assert all(c.kv_cache == [] for c in consumers.values())


def test_existing_storage_and_missing_address_abi_fail_before_allocation():
    geometry, capacity, consumers = fixture()
    consumers["model.layers.1.self_attn.attn"].kv_cache = torch.zeros(1)
    with pytest.raises(ValueError, match="already allocated"):
        declare(geometry, capacity, consumers)
    geometry, capacity, consumers = fixture()
    del consumers["model.layers.0.linear_attn"].state_binding_abi
    with pytest.raises(ValueError, match="addressing ABI"):
        declare(geometry, capacity, consumers)


def test_draft_layer_zero_is_not_target_layer_zero():
    geometry, capacity, consumers = fixture()
    del consumers["model.layers.1.self_attn.attn"]
    with pytest.raises(ValueError, match="every target layer"):
        declare(geometry, capacity, consumers)
