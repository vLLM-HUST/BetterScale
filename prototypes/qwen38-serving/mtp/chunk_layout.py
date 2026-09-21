# SPDX-License-Identifier: Apache-2.0
# Launch contracts adapted from the pinned vLLM-Ascend FLA wrappers.
"""One head-major gate layout shared by KKT, WY and the owned H/O kernel."""

import os
import torch
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.triton.triton_utils import get_aicore_num
from wy_head_major import recompute_head_major_kernel as recompute_w_u_fwd_kernel
from betterscale.patches.qwen_gdn.solve_tril import solve_tril, solve_tril_16x16_kernel
from mixed_layout import cumulative_gates


def chunk_wy(k, v, beta, g, meta):
    batch, tokens, key_heads, dim = k.shape
    assert batch == 1 and key_heads == 8 and dim == 128
    assert v.shape == (1, tokens, 24, 128)
    heads, block = 24, 64
    indices = meta.indices[block]
    tasks = len(indices)
    # For B=1, [H,B,T] (KKT) and [B,H,T] (WY/H/O) share byte order.
    bh, gh = cumulative_gates(beta, g, meta)
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
    mode = os.environ.get("MTP_GDN_WY_MODE", "original")
    if mode not in ("original", "parallel", "fused", "merge"):
        raise ValueError(f"Unknown experimental WY mode: {mode}")
    diagonal = matrix
    if mode == "merge":
        diagonal = torch.empty((1, tokens, heads, 16), dtype=torch.float32, device=k.device)
        solve_tril_16x16_kernel[(len(meta.indices[1216]), heads)](
            matrix, diagonal, meta.cu, meta.indices[1216], tokens, heads,
            BT=64, LARGE_BLOCK_T=1216, EXTRACT_SLICE_STRIDE_1=38,
            num_warps=1, num_stages=4,
        )
    elif mode != "fused":
        matrix = solve_tril(
            matrix,
            cu_seqlens=meta.cu,
            chunk_indices_large_block=meta.indices[1216],
            chunk_indices_bt=indices,
            output_dtype=k.dtype,
        )
    w = torch.empty((1, heads, tokens, dim), dtype=k.dtype, device=k.device)
    u = torch.empty_like(w)
    kernel, grid, extra = recompute_w_u_fwd_kernel, (tasks, 1), {}
    if mode != "original":
        from solve_wy_fusion import solve_wy_kernel
        group = int(os.environ.get("MTP_GDN_WY_HEADS", "1"))
        if group not in (1, 3, 6):
            raise ValueError("Experimental WY grouping must be 1, 3 or 6")
        kernel, grid = solve_wy_kernel, (tasks, heads // group)
        extra = {"FUSE_SOLVE": mode == "fused", "FUSE_MERGE": mode == "merge", "Ad": diagonal, "HEADS_PER_TASK": group}
    kernel[grid](
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
        **extra,
    )
    return w.transpose(1, 2), u.transpose(1, 2), gh
