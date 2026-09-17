# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
# mypy: ignore-errors

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import extract_slice, insert_slice

from vllm_ascend.ops.triton.fla.solve_tril import (
    merge_16x16_to_64x64_inverse_kernel,
)


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T", "H"])
def solve_tril_16x16_kernel(
    A,
    Ad,
    cu_seqlens,
    chunk_indices,
    T,
    H,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    LARGE_BLOCK_T: tl.constexpr,
    EXTRACT_SLICE_STRIDE_1: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A = A + (bos * H + i_h) * BT
    Ad = Ad + (bos * H + i_h) * 16

    base_t = i_t * LARGE_BLOCK_T

    NTASKS: tl.constexpr = 2
    N_BLOCKS: tl.constexpr = LARGE_BLOCK_T // 16 // NTASKS

    # Capacity padding is a device sentinel, not real triangular work.
    if base_t < T:
        for taskid in range(0, NTASKS):
            base_t += taskid * (LARGE_BLOCK_T // NTASKS)

            # use make_block_ptr to reduce vector computation
            b_A = tl.zeros((N_BLOCKS, 16, 16), dtype=tl.float32)
            for blkid in range(0, N_BLOCKS):
                row_start_o = base_t + blkid * 16
                col_start_o = row_start_o % BT

                # 1 Create in-block offset
                offs_rows_in_block = tl.arange(0, 16)
                offs_cols_in_block = tl.arange(0, 16)

                # 2 Calculate the pointer of each element
                ptr_A_subrec16 = (
                    A
                    + row_start_o * H * BT
                    + col_start_o
                    + offs_rows_in_block[:, None] * H * BT
                    + offs_cols_in_block[None, :]
                )

                # 3 Create a mask to prevent out-of-bounds access
                global_rows = row_start_o + offs_rows_in_block[:, None]
                global_cols = col_start_o + offs_cols_in_block[None, :]
                load_mask = (global_rows < T) & (global_cols < BT)

                # 4 Use mask to safely load data
                b_A_subrec16 = tl.load(ptr_A_subrec16, mask=load_mask, other=0.0).to(
                    tl.float32
                )
                b_A = insert_slice(
                    ful=b_A,
                    sub=b_A_subrec16[None, :, :],  # (1, 16, 16)
                    offsets=[blkid, 0, 0],
                    sizes=[1, 16, 16],
                    strides=[1, 1, 1],
                )

            local_ori_A = tl.trans(b_A, (1, 0, 2))
            local_ori_A = tl.reshape(local_ori_A, (16, 16 * N_BLOCKS))

            # Convert mask into matrix multiplication to avoid for loops ub oom
            tmp = tl.arange(0, 16).to(tl.float32)
            rows = tmp[:, None]
            cols = tmp[None, :]
            is_lower = (rows > cols).to(b_A.dtype)
            b_A = -b_A * is_lower

            # for loop to update N_BLOCKS row vector
            for i in range(1, 16):
                nblks_vec16 = -extract_slice(
                    local_ori_A, (i, 0), (1, 16 * N_BLOCKS), (EXTRACT_SLICE_STRIDE_1, 1)
                )
                b_a = tl.reshape(nblks_vec16, (N_BLOCKS, 16))

                dot_tmp = tl.trans(b_a[:, :, None] * b_A, (1, 0, 2))
                dot_product = tl.sum(dot_tmp, 0)
                b_a = b_a + dot_product

                b_a_new_expanded = b_a[:, None, :]
                b_A = insert_slice(
                    ful=b_A,
                    sub=b_a_new_expanded,
                    offsets=[0, i, 0],
                    sizes=[N_BLOCKS, 1, 16],
                    strides=[1, 1, 1],
                )

            on_diagonal = rows == cols
            b_A = tl.where(on_diagonal, b_A + 1.0, b_A)

            b_A = tl.reshape(b_A, (N_BLOCKS * 16, 16))
            p_Ai = tl.make_block_ptr(
                Ad, (T, 16), (H * 16, 1), (base_t, 0), (N_BLOCKS * 16, 16), (1, 0)
            )

            # 1 Create in-block offset
            offs_rows_to_store = tl.arange(0, N_BLOCKS * 16)
            offs_cols_to_store = tl.arange(0, 16)

            # 2 Calculate the pointer of each element
            p_Ai = (
                Ad
                + base_t * H * 16
                + 0
                + offs_rows_to_store[:, None] * H * 16
                + offs_cols_to_store[None, :]
            )
            # 3 Create a mask to prevent out-of-bounds access, only check rows
            global_store_rows = base_t + offs_rows_to_store[:, None]
            store_mask = global_store_rows < T
            # 4 use mask to save data safely
            tl.store(
                p_Ai,
                b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
                mask=store_mask,
            )


def solve_tril(
    A, cu_seqlens, chunk_indices_large_block, chunk_indices_bt, output_dtype
):
    """Owned 910B/BT64 route; keep donor arithmetic and merge unchanged.

    Forked from vllm-ascend 9bf964cb. Empty large-block tasks skip the
    16x16 recurrence; real and partially filled tasks keep the original math.
    """
    B, T, H, BT = A.shape
    if BT != 64 or cu_seqlens is None:
        raise ValueError("Owned GDN solve requires BT64 variable-length metadata")
    Ad = torch.empty((B, T, H, 16), device=A.device, dtype=torch.float32)
    Ai = torch.empty_like(A, dtype=output_dtype)
    solve_tril_16x16_kernel[len(chunk_indices_large_block), B * H](
        A,
        Ad,
        cu_seqlens,
        chunk_indices_large_block,
        T,
        H,
        BT=BT,
        LARGE_BLOCK_T=1216,
        EXTRACT_SLICE_STRIDE_1=38,
        num_warps=1,
        num_stages=4,
    )
    merge_16x16_to_64x64_inverse_kernel[len(chunk_indices_bt), B * H](
        A,
        Ad,
        Ai,
        cu_seqlens,
        chunk_indices_bt,
        T,
        H,
        BT=BT,
        num_warps=4,
        num_stages=3,
    )
    return Ai
