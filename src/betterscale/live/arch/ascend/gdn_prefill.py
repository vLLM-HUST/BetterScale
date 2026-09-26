# SPDX-License-Identifier: Apache-2.0
"""Chunked GDN numerical resource over the model's resident K-V State lanes.

Prefill follows one committed short step, so candidate zero and convolution
history zero are canonical. No native allocator, runner, or State gather is
involved. The model root owns these immutable tilings and index tensors.
"""

import os

import torch


class GDNPrefill:
    def __init__(self, tokens, geometry, device):
        from betterscale.patches.qwen_gdn.runtime import Kernels

        self.tokens = tokens
        self.cu = torch.tensor([0, tokens], dtype=torch.int64, device=device)
        self.initial = torch.ones(1, dtype=torch.bool, device=device)
        self.indices = {
            size: torch.tensor(
                [(0, j) for j in range((tokens + size - 1) // size)],
                dtype=torch.int64,
                device=device,
            )
            for size in (64, 256, 1216)
        }
        self.engine = Kernels(
            os.environ["BETTERSCALE_GDN_LIBRARY"],
            tokens,
            1,
            len(self.indices[64]),
            state_pool=True,
            key_heads=geometry.gdn_key_heads,
            value_heads=geometry.gdn_value_heads,
        )

    def __call__(self, q, k, v, g, beta, state, conv_slots):
        from vllm_ascend.ops.triton.fla.chunk import chunk_local_cumsum
        from vllm_ascend.ops.triton.fla.l2norm import l2norm_fwd

        from betterscale.patches.qwen_gdn.chunk_wy import chunk_wy

        q, k = l2norm_fwd(q), l2norm_fwd(k)
        g = g.reshape(1, self.tokens, -1)
        beta = beta.reshape_as(g)
        cumulative = chunk_local_cumsum(
            g, chunk_size=64, cu_seqlens=self.cu, block_indices=self.indices[256]
        )
        w, u, gh = chunk_wy(k, v, beta, cumulative, self)
        # H/O reads and writes candidate zero directly in the resident pool.
        state_meta = torch.cat(
            (conv_slots.to(torch.int64) * 3, self.initial[:, None]), dim=1
        )
        return (
            self.engine.pool_forward(
                *(x.transpose(1, 2).contiguous() for x in (q, k, w, u)),
                gh,
                state.recurrent.tensor,
                self.cu,
                state_meta,
                self.indices[64],
            )
            .transpose(1, 2)
            .contiguous()
        )

    def close(self):
        self.engine = None
        self.indices.clear()
        self.cu = self.initial = None
