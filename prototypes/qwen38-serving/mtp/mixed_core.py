"""Experimental mixed core: activation routing only, state always stays in pool.

This is an operator composition probe, not an installed runner/Worker. Static
capacity buffers hold two request lists. Device token maps select and restore
activations; no recurrent state is gathered, transposed or scattered.
"""
import os
from types import SimpleNamespace
from count_policy import WIDTH

import torch
from vllm_ascend.ops.triton.fla import chunk

from betterscale.patches.qwen_gdn.metadata import Metadata
from betterscale.patches.qwen_gdn.preprocess import preprocess
from betterscale.patches.qwen_gdn.chunk_wy import chunk_wy
from betterscale.patches.qwen_gdn.decode_kv import fused_recurrent_gated_delta_rule_fwd
from vllm.triton_utils import triton, tl


@triton.jit
def restore_kernel(P, V, MAP, OUT, T: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source = tl.load(MAP + token)
    spec = source >= T
    p = tl.load(P + (d // 128) * T * 128 + source * 128 + d % 128,
                (d < 3072) & ~spec, other=0)
    v = tl.load(V + (source - T) * 3072 + d, (d < 3072) & spec, other=0)
    tl.store(OUT + token * 3072 + d, tl.where(spec, v, p), d < 3072)


class MixedCore:
    def __init__(self, capacity, device='npu'):
        self.capacity = capacity
        self.layout_fusion = os.environ.get("MTP_GDN_LAYOUT_FUSION", "1") == "1"
        self.width = WIDTH
        self.prefill = Metadata(capacity, False, device)
        self.cu = torch.zeros(10, dtype=torch.int32, device=device)
        self.prefill_conv = torch.full((9, 1), -1, dtype=torch.int32, device=device)
        self.verify_conv = torch.full_like(self.prefill_conv, -1)
        self.initial = torch.zeros(9, dtype=torch.bool, device=device)
        self.accepted = torch.ones(9, dtype=torch.int32, device=device)
        self.verify = SimpleNamespace(
            cu=torch.zeros(10, dtype=torch.int32, device=device),
            slots=torch.full((9, WIDTH), -1, dtype=torch.int64, device=device),
            accepted=torch.ones(9, dtype=torch.int32, device=device))
        # Maps route activations only. Restore writes each output row once;
        # padded rows read valid row zero and are ignored by the model.
        self.prefill_map = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.verify_map = torch.zeros(8*WIDTH, dtype=torch.int64, device=device)
        self.restore = torch.zeros(capacity, dtype=torch.int64, device=device)

    def prepare(self, lengths, speculative, slots, accepted, initial):
        from betterscale.patches.qwen_gdn.metadata import chunk_rows

        assert len(lengths) <= 8 and sum(lengths) <= self.capacity
        assert all(n > 0 for n in lengths)
        assert all(n <= self.width for n, spec in zip(lengths, speculative) if spec)
        ends = [0]
        for n in lengths:
            ends.append(ends[-1] + n)
        self.cu.copy_(torch.tensor(ends + [ends[-1]] * (10-len(ends)), dtype=torch.int32))
        self.prefill_conv.fill_(-1); self.verify_conv.fill_(-1)
        self.initial.zero_(); self.accepted.fill_(1)
        pre_rows = [i for i, spec in enumerate(speculative) if not spec]
        ver_rows = [i for i, spec in enumerate(speculative) if spec]
        assert pre_rows, 'pure verification belongs to its small graph, not mixed'
        for i, spec in enumerate(speculative):
            (self.verify_conv if spec else self.prefill_conv)[i, 0] = slots[i][0]
            self.initial[i] = initial[i]
            self.accepted[i] = accepted[i]
        maps = []
        for rows, meta in ((pre_rows, self.prefill), (ver_rows, self.verify)):
            sub_lengths = [lengths[i] for i in rows]
            sub_ends = [0]
            for n in sub_lengths:
                sub_ends.append(sub_ends[-1] + n)
            meta.cu.copy_(torch.tensor(sub_ends + [sub_ends[-1]] * (10-len(sub_ends)), dtype=meta.cu.dtype))
            mapping = [j for i in rows for j in range(ends[i], ends[i+1])]
            maps.append(mapping)
        self.prefill.state.zero_()
        for j, i in enumerate(pre_rows):
            self.prefill.state[j, 0] = slots[i][0]
            self.prefill.state[j, 1] = initial[i]
        for size, dest in self.prefill.indices.items():
            dest.copy_(torch.tensor(chunk_rows([lengths[i] for i in pre_rows], size, self.capacity), dtype=torch.int64))
        self.verify.slots.fill_(-1); self.verify.accepted.fill_(1)
        for j, i in enumerate(ver_rows):
            self.verify.slots[j].copy_(torch.tensor(slots[i], dtype=torch.int64))
            self.verify.accepted[j] = accepted[i]
        # Padded maps read a valid token; their values are never restored.
        for dest, mapping in zip((self.prefill_map, self.verify_map), maps):
            dest.zero_()
            if mapping:
                dest[:len(mapping)].copy_(torch.tensor(mapping))
        restore = [0] * self.capacity
        for base, mapping in zip((0, self.capacity), maps):
            for packed, original in enumerate(mapping):
                restore[original] = base + packed
        self.restore.copy_(torch.tensor(restore))

    def __call__(self, x, a, b, weight, log, bias, conv, state):
        transformed = torch.empty_like(x)
        for spec, slots in ((False, self.prefill_conv), (True, self.verify_conv)):
            torch.ops._C_ascend.npu_causal_conv1d_custom(
                transformed, x, weight, conv_state=conv, bias_opt=None,
                query_start_loc_opt=self.cu, cache_indices_opt=slots,
                initial_state_mode_opt=None if spec else self.initial,
                num_accepted_tokens_opt=self.accepted if spec else None,
                activation_mode=1, pad_slot_id=-1, run_mode=1 if spec else 0)
        # Fuse activation routing into preprocessing; recurrent state never moves.
        tensors = []
        for i, mapping in enumerate((self.prefill_map, self.verify_map)):
            tensors.append(preprocess(transformed, a, b, log, bias, token_map=mapping,
                                      q_head_major=self.layout_fusion and i == 0))
        q, k, v, g, beta = tensors[0]
        meta = self.prefill
        if self.layout_fusion:
            from chunk_layout import chunk_wy as layout_wy
            w, u, gh = layout_wy(k, v, beta, g, meta)
        else:
            cumulative = chunk.chunk_local_cumsum(g, chunk_size=64, cu_seqlens=meta.cu, block_indices=meta.indices[256])
            w, u, gh = chunk_wy(k, v, beta, cumulative, meta)
        ph = meta.engine.pool_forward(*(z.transpose(1, 2).contiguous() for z in (q, k, w, u)),
                                      gh, state, meta.cu, meta.state, meta.indices[64]).transpose(1, 2)
        q, k, v, g, beta = tensors[1]
        vh, _ = fused_recurrent_gated_delta_rule_fwd(
            q, k, v, g, beta, 128 ** -.5, state, cu_seqlens=self.verify.cu,
            ssm_state_indices=self.verify.slots, num_accepted_tokens=self.verify.accepted)
        output = torch.empty((1, self.capacity, 24, 128), dtype=x.dtype, device=x.device)
        if self.layout_fusion:
            from mixed_layout import restore_tiled
            restore_tiled[(triton.cdiv(self.capacity, 16), 24)](ph, vh, self.restore, output, self.capacity, 16)
        else:
            restore_kernel[(self.capacity, 3)](ph, vh, self.restore, output, self.capacity, 1024)
        return output
