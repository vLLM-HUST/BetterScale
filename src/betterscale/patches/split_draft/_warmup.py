"""Finite TP DSpark preparation through its native proposer, before READY.

No scheduler requests are admitted here. Seed only unused KV pages. The memory
controller retires trial State or clears scratch contents in final State without
changing graph-bound addresses. This module does not own KV allocation policy.
"""

import torch
from vllm.forward_context import BatchDescriptor
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.attention.attention_v1 import AscendAttentionState


@torch.inference_mode()
def prepare(worker):
    r = worker.model_runner
    d = r.drafter
    assert r.input_batch.num_reqs == 0
    assert r.parallel_config.tensor_parallel_size == 8
    assert r.parallel_config.data_parallel_size == 1
    assert d.num_speculative_tokens == 5
    assert hasattr(worker, "_exact_draft_graph")
    observations = []
    # Native DSpark builds its own query layout from the target context. Seed
    # both ordinary K5 context and a short-extend context, largest batch first.
    for n in (4, 3, 2, 1):
        for width in (6, 17, 1):
            nt = n * width
            position = torch.arange(width, device=d.device).repeat(n) + 128
            qcpu = torch.arange(n + 1, dtype=torch.int32) * width
            seqcpu = torch.full((n,), 128 + width, dtype=torch.int32)
            tables, slots = {}, {}
            for group in d.draft_attn_groups:
                gid = group.kv_cache_group_id
                block = group.kv_cache_spec.block_size
                native = r.input_batch.block_table[gid].get_device_tensor(n)
                # Preserve the native table shape/stride. Only a small prefix is
                # addressed; zero is the sentinel, pages1..32 are scratch here.
                table = torch.zeros_like(native)
                pages = (128 + width + d.num_query_per_req + block - 1) // block
                for request in range(n):
                    table[request, :pages] = torch.arange(
                        1 + request * pages,
                        1 + (request + 1) * pages,
                        device=d.device,
                        dtype=table.dtype,
                    )
                token_owner = torch.arange(n, device=d.device).repeat_interleave(width)
                slot = (
                    table[token_owner, position // block].to(torch.int64) * block
                    + position % block
                )
                tables[gid], slots[gid] = table, slot
                d.set_per_group_attn_metadata(gid, table, slot)
            cad = AscendCommonAttentionMetadata(
                query_start_loc=qcpu.to(d.device),
                query_start_loc_cpu=qcpu,
                seq_lens=seqcpu.to(d.device),
                seq_lens_cpu=seqcpu,
                _seq_lens_cpu=seqcpu.clone(),
                num_reqs=n,
                num_actual_tokens=nt,
                max_query_len=width,
                max_seq_len=128 + width,
                block_table_tensor=tables[d.kv_cache_gid],
                slot_mapping=slots[d.kv_cache_gid],
                num_computed_tokens_cpu=torch.full((n,), 128, dtype=torch.int32),
                positions=position,
                positions_cpu=position.cpu(),
                num_input_tokens=nt,
                decode_token_per_req=6,
                actual_seq_lengths_q=[width] * n,
                attn_state=(
                    AscendAttentionState.SpecDecoding
                    if width == 6
                    else AscendAttentionState.ChunkedPrefill
                ),
                is_prefilling=torch.full((n,), width > 6, device="cpu"),
            )
            # DSV4 consumes the three configured auxiliary target hidden lanes.
            hidden = torch.zeros(
                (nt, d.hidden_size * 3), device=d.device, dtype=d.dtype
            )
            worker.snapshot("draft_prepare_before", requests=n, context_width=width)
            result = d._propose(
                target_token_ids=torch.full(
                    (nt,), 17, device=d.device, dtype=torch.int64
                ),
                target_positions=position,
                target_hidden_states=hidden,
                next_token_ids=torch.full((n,), 17, device=d.device, dtype=torch.int64),
                token_indices_to_sample=None,
                common_attn_metadata=cad,
                target_model_batch_desc=BatchDescriptor(
                    num_tokens=nt, num_reqs=n, uniform=width == 6
                ),
                sampling_metadata=None,
            )
            torch.npu.synchronize()
            assert result.shape == (n, 5), result.shape
            manager = worker._exact_draft_graph
            observations.append(
                worker.snapshot(
                    "draft_prepare_after",
                    requests=n,
                    context_width=width,
                    decode_entries=len(manager.decode_graphs),
                    query_entries=len(manager.query_graphs),
                )
            )
    return observations
