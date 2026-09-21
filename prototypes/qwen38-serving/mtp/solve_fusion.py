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

"""Experimental BT64 solve fusion. Preserve donor FP32 math and BF16 boundary."""
import torch
from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import extract_slice, insert_slice


@triton.jit
def inverse64(A, bos, T, i_t, i_h, H):
    # Four diagonal inverses together: same row recurrence/reduction as donor.
    block = tl.arange(0, 4)
    row = tl.arange(0, 16)
    col = tl.arange(0, 16)
    rows = i_t * 64 + block[:, None, None] * 16 + row[None, :, None]
    cols = block[:, None, None] * 16 + col[None, None, :]
    # Legalize tail addresses before load: Ascend masks may lower after memory IO.
    safe_rows = tl.minimum(rows, T - 1)
    original = tl.load(A + (bos * H + i_h) * 64 + safe_rows * H * 64 + cols)
    original = tl.where(rows < T, original, 0).to(tl.float32)
    diagonal = -original * (row[:, None] > col[None, :])
    original_rows = tl.trans(original, (1, 0, 2)).reshape(16, 64)
    for i in range(1, 16):
        a = -extract_slice(original_rows, (i, 0), (1, 64), (4, 1)).reshape(4, 16)
        product = tl.sum(tl.trans(a[:, :, None] * diagonal, (1, 0, 2)), 0)
        updated = (a + product)[:, None, :]
        diagonal = insert_slice(diagonal, updated, (0, i, 0), (4, 1, 16), (1, 1, 1))
    diagonal = tl.where(row[:, None] == col[None, :], diagonal + 1., diagonal)
    # Padded Ad rows were zero in the donor merge loads, not identity rows.
    diagonal = tl.where(rows < T, diagonal, 0.)
    return merge64(A, diagonal, bos, T, i_t, i_h, H)


@triton.jit
def inverse_from_ad(A, Ad, bos, T, i_t, i_h, H):
    rows = i_t * 64 + tl.arange(0, 64)
    cols = tl.arange(0, 16)
    values = tl.load(Ad + ((bos + tl.minimum(rows[:, None], T - 1)) * H + i_h) * 16 + cols[None, :])
    diagonal = tl.where(rows[:, None] < T, values, 0).reshape(4, 16, 16)
    return merge64(A, diagonal, bos, T, i_t, i_h, H)


@triton.jit
def merge64(A, diagonal, bos, T, i_t, i_h, H):
    # Base pointers (already offset by batch and head)
    A += (bos * H + i_h) * 64

    # load Ai_22 (Ad block at row i_t * 64 + 16, col 0, 16 * 16)
    offs_m = i_t * 64 + 16 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    Ai_22 = extract_slice(diagonal, (1, 0, 0), (1, 16, 16), (1, 1, 1)).reshape(16, 16)

    # load A_21 (A block at row i_t * 64 + 16, col 0, 16 * 16)
    mask_A = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_A = A + tl.minimum(offs_m[:, None], T - 1) * (H * 64) + offs_n[None, :]
    A_21 = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)
    tmp = tl.dot(Ai_22, A_21, input_precision="ieee")

    # load Ai_11 (Ad block at row i_t * 64, col 0, 16 * 16)
    offs_m = i_t * 64 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    Ai_11 = extract_slice(diagonal, (0, 0, 0), (1, 16, 16), (1, 1, 1)).reshape(16, 16)

    Ai_21 = -tl.dot(tmp, Ai_11, input_precision="ieee")

    # load Ai_44 (Ad block at row i_t * 64 + 48, col 0, 16 * 16)
    offs_m = i_t * 64 + 48 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    Ai_44 = extract_slice(diagonal, (3, 0, 0), (1, 16, 16), (1, 1, 1)).reshape(16, 16)

    # load A_43 (Ad block at row i_t * 64 + 48, col 32, 16 * 16)
    offs_n = 32 + tl.arange(0, 16)
    mask_A = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_A = A + tl.minimum(offs_m[:, None], T - 1) * (H * 64) + offs_n[None, :]
    A_43 = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)
    tmp = tl.dot(Ai_44, A_43, input_precision="ieee")

    # load Ai_33 (Ad block at row i_t * 64 + 32, col 0, 16 * 16)
    offs_m = i_t * 64 + 32 + tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    Ai_33 = extract_slice(diagonal, (2, 0, 0), (1, 16, 16), (1, 1, 1)).reshape(16, 16)

    Ai_43 = -tl.dot(tmp, Ai_33, input_precision="ieee")

    # build Ai_22_32 (32 * 32)
    Ai_22_32 = tl.zeros((32, 32), tl.float32)
    Ai_22_32 = insert_slice(Ai_22_32, Ai_33, (0, 0), (16, 16), (1, 1))
    Ai_22_32 = insert_slice(Ai_22_32, Ai_44, (16, 16), (16, 16), (1, 1))
    Ai_22_32 = insert_slice(Ai_22_32, Ai_43, (16, 0), (16, 16), (1, 1))

    # load A_21_32 (A block at row i_t * 64 + 32, col 0, 32 * 32)
    offs_m = i_t * 64 + 32 + tl.arange(0, 32)
    offs_n = tl.arange(0, 32)
    mask_A = (offs_m[:, None] < T) & (offs_n[None, :] < 64)
    ptr_A = A + tl.minimum(offs_m[:, None], T - 1) * (H * 64) + offs_n[None, :]
    A_21_32 = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)
    tmp = tl.dot(Ai_22_32, A_21_32, input_precision="ieee")

    # build Ai_11_32 (32 * 32)
    Ai_11_32 = tl.zeros((32, 32), tl.float32)
    Ai_11_32 = insert_slice(Ai_11_32, Ai_11, (0, 0), (16, 16), (1, 1))
    Ai_11_32 = insert_slice(Ai_11_32, Ai_22, (16, 16), (16, 16), (1, 1))
    Ai_11_32 = insert_slice(Ai_11_32, Ai_21, (16, 0), (16, 16), (1, 1))

    Ai_21_32 = -tl.dot(tmp, Ai_11_32, input_precision="ieee")

    inverse = tl.zeros((64, 64), tl.float32)
    inverse = insert_slice(inverse, Ai_11_32, (0, 0), (32, 32), (1, 1))
    inverse = insert_slice(inverse, Ai_22_32, (32, 32), (32, 32), (1, 1))
    inverse = insert_slice(inverse, Ai_21_32, (32, 0), (32, 32), (1, 1))
    return inverse


@triton.jit
def solve_kernel(A, Ai, CU, INDICES, H: tl.constexpr):
    task, head = tl.program_id(0), tl.program_id(1)
    request = tl.load(INDICES + task * 2).to(tl.int32)
    chunk = tl.load(INDICES + task * 2 + 1).to(tl.int32)
    bos = tl.load(CU + request).to(tl.int32)
    length = tl.load(CU + request + 1).to(tl.int32) - bos
    if chunk * 64 < length:
        inverse = inverse64(A, bos, length, chunk, head, H)
        rows = chunk * 64 + tl.arange(0, 64)
        cols = tl.arange(0, 64)
        tl.store(Ai + ((bos + rows[:, None]) * H + head) * 64 + cols[None, :],
                 inverse.to(Ai.dtype.element_ty, fp_downcast_rounding="rtne"), rows[:, None] < length)


def solve(A, meta):
    out = torch.empty_like(A, dtype=torch.bfloat16)
    solve_kernel[(len(meta.indices[64]), 24)](A, out, meta.cu, meta.indices[64], 24)
    return out
