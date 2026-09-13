"""Two private target packets, with their publication copies captured once.

Input ownership is banked separately by _host/_producer. Numerical State stays
single-copy, and target/sampler/draft consumers remain compute-stream ordered.
These target intermediates are not the native sampler's asynchronous D2H output.
"""

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from ._packet import CallPacket

original_call = ACLGraphWrapper.__call__


class DecodePair:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.catalogs = [wrapper.concrete_aclgraph_entries, {}]
        self.packets = [{}, {}]
        self.sequence = 0
        self.replays = [0, 0]
        self.stream = None  # native startup capture uses its own temporary stream

    def call(self, args, kwargs):
        w = self.wrapper
        ctx = get_forward_context()
        key = ctx.batch_descriptor
        source = (args, kwargs, ctx.attn_metadata)
        old_entries, old_meta, was_capturing = (
            w.concrete_aclgraph_entries,
            ctx.attn_metadata,
            ctx.capturing,
        )
        runnable = w.runnable
        assert (
            self.stream is None or torch.npu.current_stream().npu_stream == self.stream
        )
        try:
            if key not in self.packets[0]:
                # Only native startup capture may add a shape. Both graphs
                # use the native capture exemplar (conservative scalar bounds),
                # not the first live request's potentially short sequence.
                assert (
                    key not in self.catalogs[0]
                    or self.catalogs[0][key].aclgraph is None
                )
                for bank in (0, 1):
                    packet = self.packets[bank][key] = CallPacket(
                        source,
                        shared_fields=(
                            "full_compress_cos",
                            "full_compress_sin",
                            "hadamard",
                        ),
                        native_views=w.vllm_config.parallel_config.tensor_parallel_size
                        > 1,
                    )
                    a, k, ctx.attn_metadata = packet.tree
                    w.concrete_aclgraph_entries = self.catalogs[bank]
                    copies = packet.captured_copies(source)
                    # Keep native source allocations alive for the graph's
                    # entire lifetime, just as the untouched native FULL entry.
                    packet.copy_sources = copies

                    def copied_forward(*args, _copies=copies, **kwargs):
                        for destination, origin in _copies:
                            destination.copy_(origin, non_blocking=True)
                        return runnable(*args, **kwargs)

                    w.runnable = copied_forward
                    output = original_call(w, *a, **k)
                    w.runnable = runnable
                    assert self.catalogs[bank][key].aclgraph is not None
                return output
            bank = self.sequence % 2
            packet = self.packets[bank][key]
            # The recorded D2D publication reads native State-derived inputs
            # on the compute stream, into this bank's private target packet.
            a, k, ctx.attn_metadata = packet.tree
            w.concrete_aclgraph_entries = self.catalogs[bank]
            entry = self.catalogs[bank][key]
            # DSV4 does not execute host graph-task updates here. All producers,
            # target, sampler and draft stay ordered on the admitted stream.
            entry.aclgraph.replay()
            self.sequence += 1
            self.replays[bank] += 1
            return entry.output
        finally:
            w.concrete_aclgraph_entries = old_entries
            ctx.attn_metadata = old_meta
            ctx.capturing = was_capturing
            w.runnable = runnable


def call(wrapper, *args, **kwargs):
    ctx = get_forward_context()
    config = wrapper.vllm_config
    limit = config.scheduler_config.max_num_seqs * 6
    if (
        _EXTRA_CTX.is_draft_model
        or wrapper.runtime_mode != CUDAGraphMode.FULL
        or ctx.cudagraph_runtime_mode != CUDAGraphMode.FULL
        or ctx.batch_descriptor.num_tokens > limit
    ):
        return original_call(wrapper, *args, **kwargs)
    assert config.model_config.hf_config.model_type == "deepseek_v4"
    assert config.parallel_config.pipeline_parallel_size == 1
    assert config.parallel_config.decode_context_parallel_size == 1
    pair = wrapper.__dict__.get("_decode_pair")
    if pair is None:
        pair = wrapper._decode_pair = DecodePair(wrapper)
    return pair.call(args, kwargs)


_installed = False


def install():
    """Before native startup capture; import alone leaves the donor untouched."""
    global _installed
    if not _installed:
        ACLGraphWrapper.__call__ = call
        _installed = True
