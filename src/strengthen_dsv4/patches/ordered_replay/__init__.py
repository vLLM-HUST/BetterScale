"""Pinned DSV4: device-stream ordering instead of a host replay fence.

This is NOT a whole-engine two-step-ahead scheduling protocol. Target metadata
is produced on the same stream; DSV4 runner skips host graph-task updates.
Shared-expert stream dependencies inside the graph are unchanged.
"""

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

_original_call = ACLGraphWrapper.__call__
_installed = False


def install(worker):
    """Own both the wrapper hook and this worker's same-stream admission."""
    global _installed
    runner = worker.model_runner
    assert isinstance(runner.model, ACLGraphWrapper)
    assert runner.use_compress, "Only the pinned DSV4 compressed-attention path"
    assert runner.vllm_config.model_config.hf_config.model_type == "deepseek_v4"
    torch.npu.synchronize()  # explicit phase transition, never per replay
    runner.model._ordered_replay_stream = torch.npu.current_stream().npu_stream
    if not _installed:
        ACLGraphWrapper.__call__ = _ordered_call
        _installed = True
    return dict(rank=worker.rank, ordered_replay=True)


def call(original, wrapper, *args, **kwargs):
    stream = wrapper.__dict__.get("_ordered_replay_stream")
    if stream is None:
        return original(wrapper, *args, **kwargs)
    context = get_forward_context()
    entry = wrapper.concrete_aclgraph_entries.get(context.batch_descriptor)
    if (
        context.cudagraph_runtime_mode != CUDAGraphMode.FULL
        or wrapper.runtime_mode != CUDAGraphMode.FULL
        or context.capturing
        or entry is None
        or entry.aclgraph is None
    ):
        return original(wrapper, *args, **kwargs)
    assert (
        torch.npu.current_stream().npu_stream == stream
    ), "Unverified producer/replay stream"
    if wrapper.is_debugging_mode:
        assert [
            x.data_ptr() for x in args if isinstance(x, torch.Tensor)
        ] == entry.input_addresses
    entry.aclgraph.replay()
    return entry.output


def _ordered_call(self, *args, **kwargs):
    return call(_original_call, self, *args, **kwargs)
