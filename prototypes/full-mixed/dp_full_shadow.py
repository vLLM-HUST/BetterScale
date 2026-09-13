"""Optional same-state oracle against the ORIGINAL split DSA target program.

Never enabled in performance runs. Snapshots physical cache bytes once per pool;
refreshes native metadata before reference, then restores candidate state/inputs.
"""
import json
import os
from pathlib import Path
import torch
from torch.utils._pytree import tree_flatten, tree_map
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context


def byte_pools(values):
    pools = {}
    for value in tree_flatten(values)[0]:
        if not isinstance(value, torch.Tensor):
            continue
        storage = value.untyped_storage()
        key = (str(value.device), storage.data_ptr(), storage.nbytes())
        if key not in pools:
            raw = torch.as_strided(value, (storage.nbytes() // value.element_size(),), (1,), storage_offset=0)
            pools[key] = raw.view(torch.uint8)
    return list(pools.values())


def addressed_swa_rows(runner, metadata):
    rows = {}
    for name, module in runner.compilation_config.static_forward_context.items():
        if name not in metadata or 'swa_cache' not in name:
            continue
        meta = metadata[name]
        for cache in tree_flatten(getattr(module, 'kv_cache', []))[0]:
            if not isinstance(cache, torch.Tensor):
                continue
            assert cache.dtype == torch.bfloat16 and cache.is_contiguous() and cache.ndim >= 3
            ranges = rows.setdefault(cache.untyped_storage().data_ptr(), set())
            row_bytes = cache[0, 0].numel() * cache.element_size()
            base = cache.storage_offset() * cache.element_size()
            for segment in (meta.prefill, meta.decode):
                if segment is None or segment.slot_mapping is None:
                    continue
                for block, offset in segment.slot_mapping.cpu().tolist():
                    if block < 0:
                        continue
                    slot = block * cache.shape[1] + offset
                    assert 0 <= slot < cache.shape[0] * cache.shape[1]
                    ranges.add((base + slot * row_bytes, row_bytes))
    return rows


def install(worker):
    runner = worker.model_runner
    p = runner.vllm_config.parallel_config
    runner._dp_full_rank = p.data_parallel_rank * p.tensor_parallel_size + worker.rank
    runner._dp_full_checks = []
    runner.model._dp_full_shadow_runner = runner
    return dict(rank=runner._dp_full_rank, shadow='original-native-split')


def reference_metadata(mode, runner, args, kwargs, prepared):
    from dp_full import native_reference, original_metadata
    assert mode in ('native', 'unified', 'padded')
    if mode == 'unified':
        # Dummy runs clear the SOURCE slot map after DSA copies it. Rebuilding
        # would change writes rather than reproduce the captured invocation.
        return prepared
    token = native_reference.set(mode)
    try:
        return original_metadata(runner, *args, **kwargs)[0]
    finally:
        native_reference.reset(token)


def check(wrapper, runner, replay_call, args, kwargs):
    from dp_full import native_reference, original_metadata
    ctx = get_forward_context()
    metadata_args, metadata_kwargs = runner._dp_full_metadata_call
    root = Path(os.environ['DONOR_DP_OUTPUT'])
    rank = runner._dp_full_rank
    torch.npu.synchronize()
    old_metadata, old_mode = ctx.attn_metadata, ctx.cudagraph_runtime_mode
    meta = next(iter(old_metadata.values()))
    valid = int(meta.query_start_loc[-1])
    pools = byte_pools(runner.kv_caches)
    addressed = addressed_swa_rows(runner, old_metadata)
    side = [m._mtp_hidden_buffer for m in runner.get_model().modules() if hasattr(m, '_mtp_hidden_buffer')]
    state = pools + side
    before = [x.clone() for x in state]
    result = replay_call(wrapper, *args, **kwargs)
    torch.npu.synchronize()
    clone = lambda x: x.clone() if isinstance(x, torch.Tensor) else x
    actual = tree_map(clone, result)
    after = [x.clone() for x in state]
    for target, source in zip(state, before):
        target.copy_(source)
    row = dict(descriptor=str(ctx.batch_descriptor), valid=valid, status='RUNNING', signed_zero_words=0,
               reference=os.environ.get('DP_FULL_REFERENCE', 'native'))
    bounds = None
    upper = None
    try:
        mode = os.environ.get('DP_FULL_REFERENCE', 'native')
        bounds = getattr(runner, '_cross_step_bounds', None)
        upper = bounds.reference_begin(ctx) if bounds is not None else None
        prepared = ctx.attn_metadata if upper is not None else old_metadata
        reference = reference_metadata(mode, runner, metadata_args, metadata_kwargs, prepared)
        ctx.attn_metadata = reference
        ctx.cudagraph_runtime_mode = CUDAGraphMode.NONE
        refmeta = next(iter(reference.values()))
        row.update(prefills=refmeta.num_prefills, decodes=refmeta.num_decodes, exact_metadata=upper is not None)
        expected = wrapper.runnable(*args, **kwargs)
        torch.npu.synchronize()
        max_diff = 0.0
        for left, right in zip(tree_flatten(actual)[0], tree_flatten(expected)[0]):
            if isinstance(left, torch.Tensor):
                left, right = left[:valid], right[:valid]
                torch.testing.assert_close(left, right, rtol=.01, atol=1e-6)
                if left.numel():
                    max_diff = max(max_diff, float((left.float()-right.float()).abs().max()))
        row['output_max_diff'] = max_diff
        for i, (left, right) in enumerate(zip(after, state)):
            if i < len(pools):
                # No floating reinterpretation of heterogeneous shared backing.
                if not torch.equal(left, right):
                    from draft_oracle import equivalent
                    admitted = equivalent(right, left, addressed.get(right.untyped_storage().data_ptr(), ()))
                    if admitted is not None:
                        row['signed_zero_words'] += admitted
                        continue
                    details = []
                    count = 0
                    for start in range(0, left.numel(), 4*1024*1024):
                        a, b = left[start:start+4*1024*1024], right[start:start+4*1024*1024]
                        bad = (a != b).nonzero().flatten()
                        count += bad.numel()
                        if bad.numel() and len(details) < 16:
                            for offset in bad[:16-len(details)].cpu().tolist():
                                word = (start + offset) // 2 * 2
                                details.append(dict(offset=start+offset, left=left[word:word+2].cpu().tolist(),
                                                    right=right[word:word+2].cpu().tolist()))
                    row['kv_difference'] = dict(pool=i, byte_count=count, samples=details)
                    row['cache_views'] = [dict(name=name, shape=list(t.shape), dtype=str(t.dtype),
                        offset=t.storage_offset()*t.element_size(),
                        block_table=[row[:12] for row in (reference[name].prefill or reference[name].decode).block_table.cpu().tolist()])
                        for name, module in runner.compilation_config.static_forward_context.items()
                        if name in reference for t in tree_flatten(getattr(module, 'kv_cache', []))[0]
                        if isinstance(t, torch.Tensor) and t.untyped_storage().data_ptr()==right.untyped_storage().data_ptr()]
                    raise AssertionError(f'KV backing {i} differs')
            else:
                torch.testing.assert_close(left[:valid], right[:valid], rtol=.01, atol=1e-6)
        row.update(status='PASS', output_max_diff=max_diff, kv_byte_equal=row['signed_zero_words'] == 0)
        runner._dp_full_checks.append(row)
        (root/f'dp-full-shadow-rank{rank}.json').write_text(json.dumps(runner._dp_full_checks, indent=2))
    except Exception as error:
        row.update(status='FAIL', error=str(error))
        (root/f'dp-full-shadow-failure-rank{rank}.json').write_text(json.dumps(row, indent=2))
        raise
    finally:
        if bounds is not None and upper is not None:
            bounds.reference_end(ctx, upper)
        # Native reference changes shared metadata buffers; re-establish the
        # candidate's values before returning control to native sampling/draft.
        if os.environ.get('DP_FULL_REFERENCE', 'native') != 'unified':
            original_metadata(runner, *metadata_args, **metadata_kwargs)
        ctx.attn_metadata, ctx.cudagraph_runtime_mode = old_metadata, old_mode
        for target, source in zip(state, after):
            target.copy_(source)
    return result
