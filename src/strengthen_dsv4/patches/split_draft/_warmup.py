"""Build native DSpark exemplars before READY, with all KV stores disabled.

Do not run fake requests through a serving scheduler. The native proposer builds
the same query inputs/metadata as a real invocation; its bounded runnable is
captured using invalid slot mappings, like native graph warmup. No acceptance
or sampled-token result is installed into the target runner's feedback State.
"""

import torch
from vllm.forward_context import get_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata


@torch.inference_mode()
def prepare(worker):
    r = worker.model_runner
    d = r.drafter
    manager = worker._exact_draft_graph
    assert r.input_batch.num_reqs == 0
    assert not manager.decode_graphs and not manager.query_graphs
    assert all(
        g.get_metadata_builder().compressor_ratio <= 1 for g in d.draft_attn_groups
    )
    original_runnable = d._runnable
    original_tables = dict(d._per_group_block_tables)
    original_slots = dict(d._per_group_slot_mappings)

    def capture_without_kv_writes(**kwargs):
        # These are data values, not substituted operations: the captured
        # program still contains the original store, and serving refreshes the
        # real slot maps. No warmup sample can write a request's KV page.
        for buffer in d._context_slot_mapping_buffers:
            if buffer is not None:
                buffer.fill_(-1)
        for metadata in get_forward_context().attn_metadata.values():
            req = metadata.req_metadata
            req.slot_mapping.fill_(-1)
            if req.dspark_swa_indices is not None:
                req.dspark_swa_indices.fill_(-1)
        return manager(**kwargs)

    manager.preparing = True
    d._runnable = capture_without_kv_writes
    try:
        desc = next(iter(r.model._decode_pair.packets[0]))
        width = d.hidden_size * len(d.model.model.target_layer_ids)
        for n in range(4, 0, -1):
            # Fused K5, query-only short extension, query-only prefill. Query
            # geometry is bounded by seats/K, not the target prefill budget.
            for context, prefill in ((6, False), (7, False), (7, True)):
                t = n * context
                qsl_cpu = torch.arange(n + 1, dtype=torch.int32) * context
                seq_cpu = torch.full((n,), 128 + context, dtype=torch.int32)
                positions = torch.arange(context, device=r.device).repeat(n) + 128
                tables, slots = {}, {}
                for gid, table in enumerate(r.input_batch.block_table.block_tables):
                    tables[gid] = torch.zeros_like(table.get_device_tensor()[:n])
                    slots[gid] = torch.full_like(table.slot_mapping.gpu[:t], -1)
                    d.set_per_group_attn_metadata(gid, tables[gid], slots[gid])
                gid = d.draft_attn_groups[0].kv_cache_group_id
                common = AscendCommonAttentionMetadata(
                    query_start_loc=qsl_cpu.to(r.device),
                    query_start_loc_cpu=qsl_cpu,
                    seq_lens=seq_cpu.to(r.device),
                    _seq_lens_cpu=seq_cpu,
                    seq_lens_cpu_upper_bound=seq_cpu,
                    num_reqs=n,
                    num_actual_tokens=t,
                    num_input_tokens=t,
                    max_query_len=context,
                    max_seq_len=128 + context,
                    block_table_tensor=tables[gid],
                    slot_mapping=slots[gid],
                    positions=positions,
                    attn_state=AscendAttentionState.ChunkedPrefill,
                    is_prefilling=torch.full((n,), prefill, dtype=torch.bool),
                    decode_token_per_req=6,
                )
                d._propose(
                    target_token_ids=torch.zeros(t, dtype=torch.int32, device=r.device),
                    target_positions=positions,
                    target_hidden_states=torch.zeros(
                        (t, width), dtype=r.dtype, device=r.device
                    ),
                    next_token_ids=torch.zeros(n, dtype=torch.int32, device=r.device),
                    token_indices_to_sample=None,
                    common_attn_metadata=common,
                    target_model_batch_desc=desc,
                    sampling_metadata=None,
                    num_prefill_reqs=n if prefill else 0,
                    num_decode_reqs=0 if prefill else n,
                )
        assert set(manager.decode_graphs) == {1, 2, 3, 4}
        assert len(manager.query_graphs) == 8
        assert all(
            e.graph is not None
            for e in (*manager.decode_graphs.values(), *manager.query_graphs.values())
        )
    finally:
        torch.npu.synchronize()
        d._runnable = original_runnable
        d._per_group_block_tables = original_tables
        d._per_group_slot_mappings = original_slots
        manager.preparing = False
        for entry in (*manager.decode_graphs.values(), *manager.query_graphs.values()):
            entry.allow_capture = False
        manager.context_calls = manager.context_rows = manager.context_max = 0
