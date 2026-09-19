# SPDX-License-Identifier: Apache-2.0
# Launch contracts adapted from the pinned vLLM-Ascend FLA wrappers.
"""One head-major gate layout shared by KKT, WY and the owned H/O kernel."""

import torch
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.triton.triton_utils import get_aicore_num
from vllm_ascend.ops.triton.fla.wy_fast import recompute_w_u_fwd_kernel
from .solve_tril import solve_tril


def chunk_wy(k, v, beta, cumulative_g, meta):
    batch, tokens, key_heads, dim = k.shape
    assert batch == 1 and key_heads == 8 and dim == 128
    assert v.shape == (1, tokens, 24, 128)
    heads, block = 24, 64
    indices = meta.indices[block]
    tasks = len(indices)
    # For B=1, [H,B,T] (KKT) and [B,H,T] (WY/H/O) share byte order.
    bh = beta.transpose(1, 2).contiguous()
    gh = cumulative_g.transpose(1, 2).contiguous()
    matrix = torch.empty(
        (1, tokens, heads, block), dtype=torch.float32, device=k.device
    )
    matrix = DeviceOperator.chunk_scaled_dot_kkt_fwd(
        num_core=get_aicore_num(),
        bh_step=heads,
        task_num=tasks * heads,
        k=k,
        beta=bh,
        g_cumsum=gh,
        A=matrix,
        cu_seqlens=meta.cu,
        chunk_indices=indices,
        T=tokens,
        B=1,
        H=heads,
        Hg=key_heads,
        K=dim,
        BT=block,
        BK=128,
    )
    matrix = solve_tril(
        matrix,
        cu_seqlens=meta.cu,
        chunk_indices_large_block=meta.indices[1216],
        chunk_indices_bt=indices,
        output_dtype=k.dtype,
    )
    w = torch.empty((1, tokens, heads, dim), dtype=k.dtype, device=k.device)
    u = torch.empty_like(v)
    recompute_w_u_fwd_kernel[(tasks, 1)](
        k=k,
        v=v,
        beta=bh,
        w=w,
        u=u,
        A=matrix,
        g=gh,
        cu_seqlens=meta.cu,
        chunk_indices=indices,
        T=tokens,
        H=heads,
        Hg=key_heads,
        K=dim,
        V=128,
        BT=block,
        BK=64,
        BV=64,
        num_warps=4,
        num_stages=3,
    )
    return w, u, gh
