# SPDX-License-Identifier: Apache-2.0
# Adapted from pinned vLLM-Ascend dsa_cp.py (Apache-2.0).
"""Use the already-created local CPU mirrors for QLI tiling maxima.

Verification mode compares exact GPU/CPU maxima; keep it off for timing.
"""
import torch
from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPMetadataBuilder
_original=AscendDSACPMetadataBuilder._build_qli_metadata
_enabled=False
_verify=False


def install(worker):
    """Install after native warmup; import alone never replaces a donor method."""
    global _enabled, _verify
    torch.npu.synchronize()
    _enabled=True;_verify=False
    AscendDSACPMetadataBuilder._build_qli_metadata = _cpu_qli_metadata
    return dict(rank=worker.rank,cpu_qli=True,verify=False)


def _cpu_qli_metadata(self, query_start_loc, seq_lens, seq_lens_q, num_reqs):
    if not _enabled or self.compressor_ratio != 4:
        return _original(self, query_start_loc, seq_lens, seq_lens_q, num_reqs)
    cache_key = "cp_qli"
    metadata = self.common_ratio_to_sas_metadata.get(cache_key)

    if metadata is None:
        cpu = self.common_ratio_to_sas_metadata.get('_cpu_local')
        if cpu is None:
            return _original(self, query_start_loc, seq_lens, seq_lens_q, num_reqs)
        qsl = cpu['qsl_cpu']
        qlens = qsl[1:num_reqs+1] - qsl[:num_reqs]
        max_seqlen_q = max(1, int(qlens.max().item()))
        max_seqlen_k = max(1, int(cpu['sl_cpu'].max().item()))
        if _verify:
            assert max_seqlen_q == max(1, int(seq_lens_q.max().item())), 'CPU Q maximum diverged'
            assert max_seqlen_k == max(1, int(seq_lens.max().item())), 'CPU KV maximum diverged'
        metadata = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata(
            actual_seq_lengths_query=query_start_loc[1:].clone(),
            actual_seq_lengths_key=seq_lens.clone(),
            num_heads_q=self.model_config.hf_config.index_n_heads,
            num_heads_k=1,
            head_dim=self.model_config.hf_config.index_head_dim,
            query_quant_mode=0,
            key_quant_mode=0,
            batch_size=num_reqs,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.model_config.hf_config.index_topk,
            sparse_mode=3,
            pre_tokens=(1 << 63) - 1,
            next_tokens=(1 << 63) - 1,
            cmp_ratio=4,
            device=str(self.seqused_q.device),
        )
    self.common_ratio_to_sas_metadata[cache_key] = metadata
    self.req_qli_metadata[:1024] = metadata
    return self.req_qli_metadata[:1024]
