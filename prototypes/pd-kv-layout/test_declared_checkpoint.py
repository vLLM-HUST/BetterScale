"""Bridge the CPU checkpoint oracle to actual BetterScale State declarations."""
import os
from pathlib import Path
import sys

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
from torch import nn

from betterscale.live import LiveRuntime, TorchStateBackend, live_runtime
from betterscale.live.llm.qwen35.state import (
    Geometry as StateGeometry, Capacity, AttentionState, GDNState,
)
from betterscale.models.qwen35.state_binding import BaselineStateRoot
from gdn_checkpoint import Geometry, capture, install


def root():
    geometry = StateGeometry(("linear_attention", "full_attention"), 1, 256,
                             2, 4, 8, 8, 4, 16)
    capacity = Capacity(1, 2, token_pages=2)
    consumers = {}
    for name, abi in [
        ("model.layers.0.linear_attn", GDNState.binding_abi),
        ("model.layers.1.self_attn.attn", AttentionState.binding_abi),
        ("mtp.layers.0.self_attn.attn", AttentionState.binding_abi),
    ]:
        leaf = nn.Module()
        leaf.state_binding_abi = abi
        leaf.kv_cache = [] if "linear_attn" in name else torch.empty(0)
        consumers[name] = leaf
    with live_runtime(LiveRuntime(device="cpu", state_backend=TorchStateBackend("cpu"))):
        value = BaselineStateRoot(geometry, capacity, consumers=consumers,
                                  draft_names={"mtp.layers.0.self_attn.attn"})
    value.activate()
    return value


def views(value):
    leaf = value.target["0"]
    return [(leaf.conv.tensor.view(torch.uint16).numpy(), leaf.recurrent.tensor.numpy())]


def test_actual_declared_state_export_restore_without_mtp():
    source, target = root(), root()
    try:
        window, matrix = source.target["0"].numerical_tensors()
        window.copy_(torch.arange(window.numel()).reshape(window.shape).to(torch.bfloat16))
        matrix.copy_(torch.arange(matrix.numel()).reshape(matrix.shape).float())
        source.continuation.selection.tensor[1] = 3
        source.conv_selection.tensor[1] = 2
        source.draft.key.tensor.fill_(7)
        source.draft.value.tensor.fill_(8)
        target.draft.key.tensor.fill_(19)
        target.draft.value.tensor.fill_(23)
        before = target.target["0"].conv.tensor[1].clone()
        snapshot = capture(views(source), 1, 3, 2, 129, Geometry(2, 4, 8, 8),
                           writer_retired=True)
        result = install(snapshot, views(target), 0, destination_quiescent=True)
        target.continuation.selection.tensor[0] = result["selected"]
        target.conv_selection.tensor[0] = result["conv_selected"]
        torch.testing.assert_close(target.target["0"].recurrent.tensor[0],
                                   source.target["0"].recurrent.tensor[5], rtol=0, atol=0)
        torch.testing.assert_close(target.target["0"].conv.tensor[0, :3],
                                   source.target["0"].conv.tensor[1, 1:4], rtol=0, atol=0)
        torch.testing.assert_close(target.target["0"].conv.tensor[1], before, rtol=0, atol=0)
        assert torch.all(target.draft.key.tensor == 19)
        assert torch.all(target.draft.value.tensor == 23)
        assert result["draft_valid"] is False
        np.testing.assert_array_equal(snapshot.conv[0], views(source)[0][0][1, 1:4])
    finally:
        source.close()
        target.close()
