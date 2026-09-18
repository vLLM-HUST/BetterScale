"""Pinned packed GDN publication with separate host-DMA and device-reader fences.

The native runner owns model inputs, KV state and result collection. This ring
only owns GDN metadata. It does not make concurrent runner threads safe.
"""

from contextlib import contextmanager
from dataclasses import dataclass, replace
from math import prod

from .metadata import Metadata, chunk_rows


class Frame:
    def __init__(self, tokens, groups, device, stream):
        import torch

        self.stream = stream
        self.uploaded = torch.npu.Event()
        self.consumed = torch.npu.Event()
        self.has_upload = self.has_consumer = False
        self.metas = {key: Metadata(tokens, tokens <= 8, device) for key in groups}
        fields, offset = [], 0
        for meta in self.metas.values():
            for name in (
                "cu",
                "slots",
                "state",
                "conv_cu",
                "conv_slots",
                "conv_initial",
            ):
                value = getattr(meta, name)
                offset = (offset + 7) // 8 * 8
                fields.append((meta, name, value.shape, value.dtype, offset))
                offset += value.numel() * value.element_size()
            for size, value in meta.indices.items():
                offset = (offset + 7) // 8 * 8
                fields.append((meta, size, value.shape, value.dtype, offset))
                offset += value.numel() * value.element_size()
            meta.host = {}
        self.host = torch.empty(offset, dtype=torch.uint8, pin_memory=True)
        self.device = torch.empty(offset, dtype=torch.uint8, device=device)
        for meta, name, shape, dtype, offset in fields:
            count = prod(shape)
            size = torch.empty((), dtype=dtype).element_size() * count
            host = self.host[offset : offset + size].view(dtype).view(shape)
            value = self.device[offset : offset + size].view(dtype).view(shape)
            meta.host[name] = host.numpy()
            if isinstance(name, int):
                meta.indices[name] = value
            else:
                setattr(meta, name, value)
        # Establish allocator lifetime ordering once, not a per-wave compute fence.
        self.stream.wait_stream(torch.npu.current_stream())

    def acquire(self):
        # Reusing pinned memory while DMA reads it corrupts the previous wave.
        # This fence is about the source, not model completion.
        if self.has_upload and not self.uploaded.query():
            self.uploaded.synchronize()

    def fill(self, key, m, lengths, slots, *, aligned_block_size=None):
        import numpy as np

        meta = self.metas[key]
        n = len(lengths)
        h = meta.host
        if n > meta.requests or sum(lengths) > meta.tokens:
            raise ValueError("GDN publication capacity exceeded")
        if m.seq_lens_cpu is None:
            raise ValueError("Owned publication requires non-speculative CPU lengths")
        seq_lens = m.seq_lens_cpu[:n].numpy()
        h["cu"][0] = 0
        np.cumsum(lengths, out=h["cu"][1 : n + 1])
        h["cu"][n + 1 :] = sum(lengths)
        h["conv_cu"][:] = h["cu"]
        h["slots"][:] = -1
        # Mirror native mamba_get_block_table_tensor's non-speculative align
        # selection on CPU. The runner owns copying a cached/previous state into
        # this destination before forward; never mutate the shared prefix slot.
        columns = (
            np.maximum((seq_lens - 1) // aligned_block_size, 0)
            if aligned_block_size is not None
            else np.zeros(n, dtype=np.int64)
        )
        h["slots"][:n] = slots[np.arange(n), columns]
        h["conv_slots"][:, 0] = h["slots"]
        h["conv_initial"][:] = False
        # Same seq_lens - query_lens formula as CommonAttentionMetadata,
        # including capture's synthetic lengths; never use GPU-derived copies.
        h["conv_initial"][:n] = seq_lens > lengths
        h["state"][:, 0] = h["slots"]
        h["state"][:, 1] = h["conv_initial"]
        for size in meta.indices:
            h[size][:] = chunk_rows(lengths, size, meta.tokens)
        return meta

    def publish(self):
        import torch

        with torch.npu.stream(self.stream):
            if self.has_consumer:
                self.stream.wait_event(self.consumed)
            self.device.copy_(self.host, non_blocking=True)
            self.uploaded.record(self.stream)
        self.has_upload = True
        # A device-side dependency, not a host synchronize. Also orders observers
        # that read metadata before entering _model_forward.
        torch.npu.current_stream().wait_event(self.uploaded)

    def release(self):
        import torch

        self.consumed.record(torch.npu.current_stream())
        self.has_consumer = True


@contextmanager
def graph_resources(runner, bank, tokens):
    import vllm_ascend.compilation.acl_graph as acl

    if not hasattr(runner, "_owned_graph_resources"):
        runner._owned_graph_resources = {}
    key = bank, tokens
    if key not in runner._owned_graph_resources:
        runner._owned_graph_resources[key] = acl.GraphParams(
            events={tokens: []},
            workspaces={tokens: None},
            handles={tokens: []},
            attn_params={tokens: []},
        )
    previous = acl._graph_params
    acl._graph_params = runner._owned_graph_resources[key]
    try:
        yield
    finally:
        acl._graph_params = previous


def install():
    import torch
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor, get_forward_context
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher as Dispatcher
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner as Runner
    from vllm_ascend.ops.gdn_attn_builder import (
        AscendGDNAttentionMetadataBuilder as Builder,
    )

    @dataclass(frozen=True)
    class BankDescriptor(BatchDescriptor):
        bank: int = 0

    original_descriptor = Dispatcher._create_padded_batch_descriptor

    def descriptor(self, *a, **kw):
        desc = original_descriptor(self, *a, **kw)
        return BankDescriptor(**vars(desc), bank=getattr(self, "_owned_bank", 0))

    original_initialize = Dispatcher.initialize_cudagraph_keys

    def initialize(self, *a, **kw):
        result = original_initialize(self, *a, **kw)
        keys = self.cudagraph_keys[CUDAGraphMode.FULL]
        keys.update(replace(k, bank=1) for k in tuple(keys))
        return result

    original_warmup = Runner._warmup_and_capture

    def warmup(self, desc, *a, **kw):
        self._owned_capture_bank = desc.bank
        try:
            return original_warmup(self, desc, *a, **kw)
        finally:
            self._owned_capture_bank = None

    original_determine = Runner._determine_batch_execution_and_padding

    def determine(self, *a, **kw):
        bank = getattr(self, "_owned_capture_bank", None)
        if bank is None:
            bank = getattr(self, "_owned_next_bank", 0)
        self._owned_bank = bank
        self.cudagraph_dispatcher._owned_bank = bank
        return original_determine(self, *a, **kw)

    original_attention = Runner._build_attention_metadata

    def attention(self, num_tokens, num_reqs, max_query_len, **kw):
        tokens = kw.get("num_tokens_padded") or num_tokens
        bank = getattr(self, "_owned_bank", 0)
        groups = {}
        for gid, attn_groups in enumerate(self.attn_groups):
            for aid, group in enumerate(attn_groups):
                builder = group.get_metadata_builder(0)
                if isinstance(builder, Builder):
                    if builder.vllm_config.cache_config.mamba_cache_mode not in (
                        "none",
                        "align",
                    ):
                        raise ValueError(
                            "Owned publication requires mamba_cache_mode=none or align"
                        )
                    groups[gid, aid] = builder
        if not hasattr(self, "_owned_frames"):
            self._owned_frames = {}
            self._owned_ingress = torch.npu.Stream()
        key = tokens, bank
        if key not in self._owned_frames:
            self._owned_frames[key] = Frame(
                tokens, groups, self.device, self._owned_ingress
            )
        frame = self._owned_frames[key]
        frame.acquire()
        self._owned_frame = frame
        for key, builder in groups.items():
            builder._owned_publication = (
                frame,
                key,
                self.input_batch.block_table[key[0]].get_cpu_tensor().numpy(),
            )
        try:
            result = original_attention(self, num_tokens, num_reqs, max_query_len, **kw)
        finally:
            for builder in groups.values():
                del builder._owned_publication
        frame.publish()
        if not getattr(self, "_elastic_dummy", False):
            self._owned_next_bank = 1 - bank
        return result

    original_forward = Runner._model_forward

    def forward(self, *a, **kw):
        ctx = get_forward_context()
        if ctx.attn_metadata is None:
            return original_forward(self, *a, **kw)
        frame = self._owned_frame
        with graph_resources(
            self, self._owned_bank, next(iter(frame.metas.values())).tokens
        ):
            result = original_forward(self, *a, **kw)
        frame.release()
        return result

    Dispatcher._create_padded_batch_descriptor = descriptor
    Dispatcher.initialize_cudagraph_keys = initialize
    Runner._warmup_and_capture = warmup
    Runner._determine_batch_execution_and_padding = determine
    Runner._build_attention_metadata = attention
    Runner._model_forward = forward
