"""CPU transport double: native placement and non-tensor continuation metadata."""

from contextlib import nullcontext
from queue import Queue
from types import SimpleNamespace as S

import pytest
import torch

from betterscale.models.qwen35 import cache_worker
from betterscale.live.runtime.host_state import TorchHostStateBackend


@pytest.mark.parametrize("incremental", [False, True])
def test_native_restore_lowers_pages_and_preserves_new_epoch_and_verify_role(
    monkeypatch,
    incremental,
):
    import vllm.distributed

    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_rank", lambda: 0)
    events = []

    class Stream:
        def __init__(self, **kwargs):
            pass

        def wait_event(self, event):
            events.append(event)

    class Event:
        def record(self, *args):
            pass

    npu = S(
        Stream=Stream,
        Event=Event,
        stream=lambda _: nullcontext(),
        current_stream=lambda _: None,
        set_device=lambda _: None,
    )
    monkeypatch.setattr(torch, "npu", npu, raising=False)

    class CPUBackend(TorchHostStateBackend):
        def _copy_lanes(self, lanes, payloads, *, to_host, stream):
            return super()._copy_lanes(lanes, payloads, to_host=to_host, stream=None)

    monkeypatch.setattr(cache_worker, "TorchHostStateBackend", CPUBackend)
    residents, pages = object(), object()

    def state(tensor, domain, count, span=1, leading=0):
        return S(
            tensor=tensor,
            domain=domain,
            storage_dtype=tensor.dtype,
            block_shape=tuple(tensor.shape[1:]),
            num_blocks=count,
            physical_blocks_per_logical_block=span,
            leading_physical_blocks=leading,
            logical_block_bytes=tensor[0].numel() * tensor.element_size() * span,
        )

    gdn = state(
        torch.arange(26, dtype=torch.float32).reshape(13, 2), residents, 4, 3, 1
    )
    fa = state(torch.arange(128, dtype=torch.float32).reshape(64, 2), pages, 64)
    epoch = state(torch.tensor([3, 0, 0, 0]), residents, 4)
    root = S(
        residents=residents,
        pages=pages,
        capacity=S(page_tokens=128),
        continuation=S(resident_epoch=epoch),
        named_states=lambda: [("gdn", gdn), ("fa", fa), ("epoch", epoch)],
    )
    runner = S(
        device="cpu",
        _live_state_root=root,
        _live_resident_epochs=[3, 0, 0, 0],
        _live_previous_verify={0},
        _mtp_apc_done=Event(),
    )
    worker = cache_worker.CacheWorker(runner, 4096)
    receipts = Queue()
    worker._send = lambda cmd, error=None: receipts.put((cmd, error))
    original_gdn, original_fa = gdn.tensor[1:4].clone(), fa.tensor[16:32].clone()
    command = dict(
        operation=1, kind="store", seat=0, epoch=3, key="A", blocks=[1], block_size=2048
    )
    if incremental:
        command.update(pages=["tail:A:0"], missing=[0])
    try:
        worker.submit(command)
        assert receipts.get(timeout=3)[1] is None
        gdn.tensor[1:4].zero_()
        fa.tensor[16:32].zero_()
        runner._live_previous_verify.clear()
        worker.submit(
            dict(command, operation=2, kind="load", seat=2, epoch=7, blocks=[2])
        )
        assert receipts.get(timeout=3)[1] is None
        assert torch.equal(gdn.tensor[7:10], original_gdn)
        assert torch.equal(fa.tensor[32:48], original_fa)
        assert epoch.tensor.tolist() == [3, 0, 7, 0]
        assert runner._live_resident_epochs[2] == 7
        assert runner._live_previous_verify == {2}
        assert events[0] is runner._mtp_apc_done and not worker.inflight
        worker.submit(dict(command, operation=3, kind="drop", seat=None, epoch=None))
        assert receipts.get(timeout=3)[1] is None and not worker.verify
        assert worker.backend.committed_bytes == 0
        if incremental:
            assert worker.page_backend.committed_bytes == 0
    finally:
        worker.waiters.shutdown(wait=True)
