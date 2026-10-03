"""CPU address-oracle tests; native async lifetime is covered by the NPU probe."""
import ctypes
from types import SimpleNamespace as S

import pytest
import torch

from betterscale.models.qwen35.state_dma import descriptors
from betterscale.live.runtime.host_state import TorchHostStateBackend, HostStateKey, HostStateSelection, HostStateDomainSelection
from betterscale.live.runtime.page_transport import ObjectStateTransport


def lane():
    tensor = torch.arange(44, dtype=torch.int32).reshape(11, 4)
    state = S(tensor=tensor, physical_blocks_per_logical_block=2, leading_physical_blocks=1)
    payload = S(tensor=torch.empty((6, 4), dtype=torch.int32))
    return state, payload


@pytest.mark.parametrize("to_host", [True, False])
def test_coalesced_and_reordered_runs_preserve_leading_rows(to_host):
    state, payload = lane()
    blocks = (3, 0, 1)
    expected = torch.cat([state.tensor[7:9], state.tensor[1:5]])
    if not to_host:
        payload.tensor.copy_(expected)
        state.tensor.fill_(-1)
    src, dst, sizes = descriptors([("lane", state, blocks)], {"lane": payload}, to_host=to_host)
    assert sizes.tolist() == [32, 64]
    for s, d, n in zip(src.tolist(), dst.tolist(), sizes.tolist()):
        ctypes.memmove(d, s, n)
    if to_host:
        assert torch.equal(payload.tensor, expected)
    else:
        assert torch.equal(state.tensor[7:9], expected[:2])
        assert torch.equal(state.tensor[1:5], expected[2:])
        assert (state.tensor[[0, 5, 6, 9, 10]] == -1).all()


@pytest.mark.parametrize("damage", ["stride", "dtype", "shape", "length", "bounds", "duplicate", "negative"])
def test_invalid_descriptors_rejected(damage):
    state, payload = lane()
    blocks = (0, 1, 2)
    if damage == "stride": state.tensor = state.tensor[:, ::2]
    if damage == "dtype": payload.tensor = payload.tensor.float()
    if damage == "shape": payload.tensor = torch.empty((6, 5), dtype=torch.int32)
    if damage == "length": payload.tensor = payload.tensor[:4]
    if damage == "bounds": blocks = (0, 1, 5)
    if damage == "duplicate": blocks = (0, 1, 1)
    if damage == "negative": blocks = (0, 1, -1)
    with pytest.raises(ValueError):
        descriptors([("lane", state, blocks)], {"lane": payload}, to_host=True)


def test_injected_submitter_reaches_every_transport_lane_and_audit():
    domain = object()
    tensor = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    state = S(tensor=tensor, domain=domain, storage_dtype=tensor.dtype, block_shape=(4,),
              num_blocks=8, physical_blocks_per_logical_block=1, leading_physical_blocks=0,
              logical_block_bytes=16)
    calls = []
    def copy(lanes, payloads, *, to_host):
        calls.append(to_host)
        TorchHostStateBackend._enqueue_copies(lanes, payloads, to_host=to_host)
    class Sink:
        def __init__(self): self.data = {}
        def ensure(self, key): return key in self.data
        def put(self, key, value): self.data[key] = value
        def get(self, key): return self.data[key]
    transport = ObjectStateTransport(Sink(), "test", max_transfers=3, staging_bytes=32,
                                     verify=True, enqueue_copies=copy)
    for staging, restore in transport.available:
        assert staging.backend._enqueue_copies is copy
        assert restore._enqueue_copies is copy
    selection = lambda i: HostStateSelection((HostStateDomainSelection(domain, (i,)),))
    key = HostStateKey("A", 1)
    transport.transfer(key, [("lane", state)], {"x": selection(0)}, store=True, stream=None).result()
    transport.transfer(key, [("lane", state)], {"x": selection(7)}, store=False, stream=None).result()
    assert torch.equal(tensor[0], tensor[7])
    assert calls == [True, False, True]
