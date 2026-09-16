"""Single-sequence padding for the bounded Qwen FULL prefill contract.

Keep causal-convolution endpoints real. Only chunk recurrence consumes a padded
sequence, with q/k/v/beta/g zeroed outside the real tensor endpoint. Mixed batches are excluded; qualification is recorded in this patch README.
BF16 shape changes need not preserve bitwise output versus unpadded execution.
"""

import torch


def normalize(builder, metadata, bucket):
    from vllm_ascend.ops.gdn_attn_builder import (
        _build_non_spec_chunked_prefill_metadata,
    )

    assert metadata.num_prefills == 1 and metadata.num_decodes == 0
    if not hasattr(builder, "_padded_chunks"):
        builder._padded_chunks = {}
    if bucket not in builder._padded_chunks:
        cpu = torch.tensor([0, bucket], dtype=torch.int32)
        chunk = _build_non_spec_chunked_prefill_metadata(
            builder, cpu, metadata.prefill_query_start_loc.device
        )
        chunk.padded_cu_seqlens = cpu.to(device=metadata.prefill_query_start_loc.device)
        builder._padded_chunks[bucket] = chunk
    chunk = builder._padded_chunks[bucket]
    metadata.non_spec_prefill_metadata.chunk = chunk
    metadata.chunk_indices = chunk.chunk_indices_chunk64
    metadata.chunk_offsets = chunk.chunk_offsets_chunk64
    # These belong to the alternative Triton causal-convolution implementation;
    # this pinned Ascend GDN core never reads them. Its native conv consumes the
    # real tensor endpoints in non_spec_prefill_metadata.causal_conv1d instead.
    metadata.nums_dict = None
    metadata.batch_ptr = None
    metadata.token_chunk_offset_ptr = None
    metadata.num_actual_tokens = bucket
    metadata.num_prefill_tokens = bucket
    return metadata


def install():
    import vllm_ascend.ops.gdn as gdn

    original = gdn.chunk_gated_delta_rule

    def masked(*args, **kwargs):
        meta = kwargs.get("prebuilt_meta")
        if meta is None or not hasattr(meta, "padded_cu_seqlens"):
            return original(*args, **kwargs)
        assert not args and not kwargs.get("head_first", False)
        q = kwargs["q"]
        assert q.shape[0] == 1
        valid = torch.arange(q.shape[1], device=q.device) < kwargs["cu_seqlens"][-1]
        for name in ["q", "k", "v", "g", "beta"]:
            value = kwargs[name]
            mask = valid.reshape(1, -1, *([1] * (value.ndim - 2)))
            kwargs[name] = torch.where(mask, value, 0)
        kwargs["cu_seqlens"] = meta.padded_cu_seqlens
        return original(**kwargs)

    gdn.chunk_gated_delta_rule = masked
