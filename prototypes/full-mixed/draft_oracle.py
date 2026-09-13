"""Shadow-only signed-zero equivalence, restricted to addressed BF16 KV rows.

Never reinterpret an entire aliased cache pool as BF16: other groups can keep
FP32 or quantized state in the same backing. Everything outside these rows must
remain byte-identical. This permits no nonzero numerical tolerance.
"""
import torch


def verify_rejection(rank):
    """Exercise the actual Ascend K5 kernel with arbitrary wrong proposals."""
    from vllm_ascend.ops.triton.reject_sample import rejection_greedy_sample_with_triton
    target = torch.tensor([11,12,13,14,15,21,22,23,24,25], device='npu', dtype=torch.int32)
    draft = target.clone()
    draft[6] = 999  # Second request must stop at the first mismatch.
    output = torch.full((2,6), -1, device='npu', dtype=torch.int32)
    rejection_greedy_sample_with_triton(output, [5,5],
        torch.tensor([5,10], device='npu', dtype=torch.int32), draft, target,
        torch.tensor([[16],[26]], device='npu', dtype=torch.int32), None, 5, 1, 2)
    actual = output.cpu().tolist()
    assert actual == [[11,12,13,14,15,16],[21,22,-1,-1,-1,-1]], actual
    return dict(rank=rank, kernel='Ascend rejection_greedy_sample_with_triton',
                accepted='target tokens only; first mismatch terminates proposal')


def addressed_rows(drafter, count, capacity):
    result = {}
    for index, layer in enumerate(drafter.model.model.layers.values()):
        cache = layer.self_attn.dsa_attn.swa_cache_layer.kv_cache
        while isinstance(cache, (tuple, list)) and len(cache) == 1:
            cache = cache[0]
        assert isinstance(cache, torch.Tensor) and cache.dtype == torch.bfloat16
        assert cache.is_contiguous() and cache.ndim >= 3
        row_bytes = cache[0, 0].numel() * cache.element_size()
        gid = drafter._layer_group_idx[index]
        context = drafter._context_slot_mapping_buffers[index][:capacity]
        query = drafter._per_group_query_slot_mapping_buffers[gid][:count * drafter.num_query_per_req]
        base = cache.storage_offset() * cache.element_size()
        rows = result.setdefault(cache.untyped_storage().data_ptr(), set())
        for slots in (context, query):
            assert slots.ndim == 1
            for slot in slots.cpu().tolist():
                if slot < 0:
                    continue
                assert slot < cache.shape[0] * cache.shape[1]
                rows.add((base + slot * row_bytes, row_bytes))
    return result


def signed_zero_only(a, b, start, rows):
    """CPU byte slabs; return admitted differing BF16 word count, or None."""
    assert a.dtype == b.dtype == torch.uint8 and start % 2 == 0
    lhs = a.reshape(-1, 2).to(torch.int32)
    rhs = b.reshape(-1, 2).to(torch.int32)
    lhs = lhs[:, 0] | (lhs[:, 1] << 8)
    rhs = rhs[:, 0] | (rhs[:, 1] << 8)
    changed = torch.nonzero(lhs != rhs).flatten()
    if not bool((((lhs[changed] & 0x7fff) == 0) & ((rhs[changed] & 0x7fff) == 0)).all()):
        return None
    # Slab membership is bounded by the wave's addressed rows, not pool size.
    allowed = torch.zeros(a.numel() // 2, dtype=torch.bool)
    end = start + a.numel()
    for base, length in rows:
        lo, hi = max(base, start), min(base + length, end)
        if lo < hi:
            allowed[(lo-start)//2:(hi-start)//2] = True
    if not bool(allowed[changed].all()):
        return None
    return changed.numel()


def equivalent(live, saved, rows):
    count = 0
    for start in range(0, live.numel(), 1024 * 1024):
        a, b = live[start:start+1024*1024], saved[start:start+1024*1024]
        if torch.equal(a, b):
            continue
        admitted = signed_zero_only(a.cpu(), b.cpu(), start, rows)
        if admitted is None:
            return None
        count += admitted
    return count


def diagnose_output(entry, kwargs, observed, expected, pools, before, after):
    """Failure-only controlled re-executions, never a relaxed acceptance gate."""
    d = entry.drafter
    result = dict(graph_tokens=observed.cpu().tolist(),
                  eager_tokens=expected.cpu().tolist(),
                  capacity=entry.context_capacity, requests=entry.request_count)
    addressed = addressed_rows(d, entry.request_count, entry.context_capacity)
    rows = []
    for index, (live, saved) in enumerate(zip(pools, after)):
        ranges = sorted(addressed.get(live.untyped_storage().data_ptr(), ()))
        if not ranges:
            continue
        a = torch.cat([live[start:start+length].view(torch.bfloat16) for start,length in ranges]).float()
        b = torch.cat([saved[start:start+length].view(torch.bfloat16) for start,length in ranges]).float()
        rows.append(dict(pool=index, addressed_values=a.numel(), unequal=int((a!=b).sum()),
                         max_diff=float((a-b).abs().max()), max_abs=float(a.abs().max())))
    result['addressed_kv_differences'] = rows

    def reset():
        for dst, src in zip(pools, before):
            dst.copy_(src)
    reset()
    repeated = entry.reference(**kwargs).clone()
    torch.npu.synchronize()
    result['repeated_unpadded_tokens'] = repeated.cpu().tolist()
    reset()
    padded = entry.original(**kwargs).clone()
    torch.npu.synchronize()
    result['padded_eager_tokens'] = padded.cpu().tolist()
    result['padded_eager_kv_matches_graph'] = all(torch.equal(a,b) for a,b in zip(pools,after))
    manager = getattr(entry.reference, '__self__', None)
    if manager is not None and hasattr(manager, 'actual_context'):
        result['actual_context'] = manager.actual_context
        reset()
        capacity = d._dflash_num_context
        try:
            d._dflash_num_context = manager.actual_context
            unpadded = entry.original(**kwargs).clone()
            torch.npu.synchronize()
            result['canonical_unpadded_tokens'] = unpadded.cpu().tolist()
        finally:
            d._dflash_num_context = capacity
    return result
