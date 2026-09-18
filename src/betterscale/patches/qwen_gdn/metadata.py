"""Capacity-stable GDN metadata; packed active prefix and permanent empty sentinel."""

from types import SimpleNamespace


def chunk_rows(lengths, size, capacity, requests=8):
    if (
        not lengths
        or len(lengths) > requests
        or any(n <= 0 for n in lengths)
        or sum(lengths) > capacity
    ):
        raise ValueError("Invalid packed GDN lengths")
    count = (capacity + size - 1) // size + requests - 1
    rows = [
        (i, j) for i, n in enumerate(lengths) for j in range((n + size - 1) // size)
    ]
    return rows + [(requests, 0)] * (count - len(rows))


class Metadata:
    def __init__(self, tokens, decode, device):
        import torch

        self.tokens, self.decode = tokens, decode
        self.requests = tokens if decode else 9
        n = self.requests
        self.cu = torch.empty(n + 1, dtype=torch.int64, device=device)
        self.slots = torch.empty(n, dtype=torch.int64, device=device)
        self.state = torch.empty((n, 2), dtype=torch.int64, device=device)
        self.conv_cu = torch.empty(n + 1, dtype=torch.int32, device=device)
        self.conv_slots = torch.empty((n, 1), dtype=torch.int32, device=device)
        self.conv_initial = torch.empty(n, dtype=torch.bool, device=device)
        self.indices = (
            {}
            if decode
            else {
                size: torch.empty(
                    ((tokens + size - 1) // size + 7, 2),
                    dtype=torch.int64,
                    device=device,
                )
                for size in (64, 256, 1216)
            }
        )
        self.engine = None
        if not decode:
            import os
            from .runtime import Kernels

            # Device POD initialization includes H2D; do it before capture.
            self.engine = Kernels(
                os.environ["BETTERSCALE_GDN_LIBRARY"],
                tokens,
                n,
                len(self.indices[64]),
                state_pool=True,
            )

    def update(self, builder, m, lengths):
        import torch
        from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor

        n = len(lengths)
        if n > self.requests or sum(lengths) > self.tokens:
            raise ValueError("GDN capacity exceeded")
        ends = [0]
        for length in lengths:
            ends.append(ends[-1] + length)
        ends += [ends[-1]] * (self.requests - n)
        # Pageable source / blocking copy: no unsafe reuse of a pinned slab.
        self.cu.copy_(torch.tensor(ends, dtype=torch.int64))
        self.conv_cu.copy_(self.cu)
        table = mamba_get_block_table_tensor(
            m.block_table_tensor,
            m.seq_lens,
            builder.kv_cache_spec,
            builder.vllm_config.cache_config.mamba_cache_mode,
        )
        self.slots.fill_(-1)
        self.slots[:n].copy_(table[:n, 0])
        self.conv_slots[:, 0].copy_(self.slots)
        self.conv_initial.zero_()
        self.conv_initial[:n].copy_(m.compute_num_computed_tokens()[:n] > 0)
        self.state[:, 0].copy_(self.slots)
        self.state[:, 1].copy_(self.conv_initial)
        for size, dest in self.indices.items():
            dest.copy_(
                torch.tensor(chunk_rows(lengths, size, self.tokens), dtype=torch.int64)
            )


def install():
    from vllm_ascend.ops.gdn_attn_builder import (
        AscendGDNAttentionMetadataBuilder as Builder,
    )
    from vllm.v1.attention.backend import AttentionCGSupport

    def build(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
        m = common_attn_metadata
        raw = m.query_start_loc_cpu.diff().tolist()
        lengths = tuple(n for n in raw if n > 0)
        if raw[: len(lengths)] != list(lengths):
            raise ValueError(
                "GDN requires packed positive requests followed by empty suffix"
            )
        tokens = m.num_input_tokens
        decode = tokens <= 8 and all(n == 1 for n in lengths)
        publication = getattr(self, "_owned_publication", None)
        if publication is not None:
            frame, key, slots = publication
            meta = frame.fill(
                key,
                m,
                lengths,
                slots,
                aligned_block_size=(
                    self.kv_cache_spec.block_size
                    if self.vllm_config.cache_config.mamba_cache_mode == "align"
                    else None
                ),
            )
        else:
            cache = getattr(self, "_elastic_buffers", None)
            if cache is None:
                cache = self._elastic_buffers = {}
            key = tokens, decode
            if key not in cache:
                cache[key] = Metadata(tokens, decode, m.query_start_loc.device)
            meta = cache[key]
            meta.update(self, m, lengths)
        return SimpleNamespace(
            owned=meta,
            num_actual_tokens=tokens,
            num_prefills=0 if decode else len(lengths),
            num_decodes=len(lengths) if decode else 0,
            num_spec_decodes=0,
            spec_sequence_masks=None,
        )

    Builder.build = build
    Builder.build_for_cudagraph_capture = lambda self, m: self.build(0, m)
    Builder._cudagraph_support = AttentionCGSupport.ALWAYS
