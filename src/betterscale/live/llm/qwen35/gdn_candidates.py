# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import torch

from vllm.triton_utils import tl, triton

exp = tl.exp
# BetterScale: pinned vLLM752a3a50 general FLA recurrence, K-V pool addresses.
# Isolated experimental MTP adapter from September24 capacity32 capsule,
# source Git blob 4c1518d498467947f8f8e2b42e71e299a0cf1db7.
# Geometry admission: Qwen0.8B TP1 H16/HV16 and Qwen35B TP2 H8/HV16.
# Do not replace the published non-speculative adapter without its own gate.


@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
        "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
    }
)
@triton.jit(do_not_specialize=["N", "T"])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    N: tl.int64,  # num of sequences
    T: tl.int64,  # num of tokens
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace
    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
    SINGLE_TOKEN: tl.constexpr,
    QUERY_CAPACITY: tl.constexpr,
    TOKEN_OFFSET: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    bos += TOKEN_OFFSET
    T = tl.maximum(T - TOKEN_OFFSET, 0)
    if T == 0:
        # no tokens to process for this sequence
        return

    if SINGLE_TOKEN:
        T = 1

    if USE_INITIAL_STATE and IS_CONTINUOUS_BATCHING:
        if tl.load(ssm_state_indices + i_n * stride_indices_seq) < 0:
            return
    for i_v in range((V + BV - 1) // BV):
        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)

        p_q = q + (bos * H + i_h) * K + o_k
        p_k = k + (bos * H + i_h) * K + o_k
        p_v = v + (bos * HV + i_hv) * V + o_v
        if IS_BETA_HEADWISE:
            p_beta = beta + (bos * HV + i_hv) * V + o_v
        else:
            p_beta = beta + bos * HV + i_hv

        if not IS_KDA:
            p_g = g + bos * HV + i_hv
        else:
            p_gk = g + (bos * HV + i_hv) * K + o_k

        p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_k[:, None] & mask_v[None, :]

        b_h = tl.zeros([BK, BV], dtype=tl.float32)
        if USE_INITIAL_STATE:
            if IS_CONTINUOUS_BATCHING:
                if IS_SPEC_DECODING:
                    if TOKEN_OFFSET:
                        i_t = TOKEN_OFFSET - 1
                    else:
                        i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
                else:
                    i_t = 0
                # Load state index and check for invalid entries
                state_idx = tl.load(
                    ssm_state_indices
                    + i_n * stride_indices_seq
                    + i_t * stride_indices_tok
                ).to(tl.int64)
                # Owned pool uses negative indices for invalid rows; slot zero is valid.
                p_h0 = h0 + state_idx * stride_init_state_token
            else:
                p_h0 = h0 + bos * HV * V * K
            p_h0 = p_h0 + i_hv * V * K + o_k[:, None] * V + o_v[None, :]
            b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

        # Static speculative expansion preserves intermediate snapshots on the pinned
        # Ascend compiler; its dynamic loop passed outputs but corrupted states.
        for i_t in tl.static_range(QUERY_CAPACITY):
            if i_t < T:
                b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
                b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
                b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

                if USE_QK_L2NORM_IN_KERNEL:
                    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
                    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
                b_q = b_q * scale
                # [BK, BV]
                if not IS_KDA:
                    b_g = tl.load(p_g).to(tl.float32)
                    b_h *= exp(b_g)
                else:
                    b_gk = tl.load(p_gk).to(tl.float32)
                    b_h *= exp(b_gk[:, None])
                # [BV]
                b_v -= tl.sum(b_h * b_k[:, None], 0)
                if IS_BETA_HEADWISE:
                    b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
                else:
                    b_beta = tl.load(p_beta).to(tl.float32)
                b_v *= b_beta
                # [BK, BV]
                b_h = tl.fma(b_k[:, None], b_v[None, :], b_h)
                # [BV]
                b_o = tl.sum(b_h * b_q[:, None], 0)
                tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

                # keep the states for multi-query tokens
                if INPLACE_FINAL_STATE:
                    # Load state index and check for invalid entries
                    final_state_idx = tl.load(
                        ssm_state_indices
                        + i_n * stride_indices_seq
                        + (i_t + TOKEN_OFFSET) * stride_indices_tok
                    ).to(tl.int64)
                    # Only store valid owned-pool slots.
                    if final_state_idx >= 0:
                        p_ht = ht + final_state_idx * stride_final_state_token
                        p_ht = p_ht + i_hv * V * K + o_k[:, None] * V + o_v[None, :]
                        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
                else:
                    p_ht = ht + (bos + i_t) * stride_final_state_token
                    p_ht = p_ht + i_hv * V * K + o_k[:, None] * V + o_v[None, :]
                    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

                p_q += H * K
                p_k += H * K
                p_o += HV * V
                p_v += HV * V
                if not IS_KDA:
                    p_g += HV
                else:
                    p_gk += HV * K
                p_beta += HV * (V if IS_BETA_HEADWISE else 1)


def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update the K-V pool directly, optionally retaining speculative prefixes.

    Speculative rows own disjoint candidate slots. The previous wave's accepted
    count includes the anchor (1..row width), selecting the initial state; this
    wave writes one candidate per input token. Row width remains fixed even if
    the current query is shorter. The caller owns these device-value invariants
    and must not reuse slots until their preceding wave has completed.
    """
    assert inplace_final_state
    assert cu_seqlens is not None and ssm_state_indices is not None
    speculative = num_accepted_tokens is not None
    if speculative:
        assert ssm_state_indices.ndim == 2 and 2 <= ssm_state_indices.shape[1] <= 5
        assert num_accepted_tokens.ndim == 1 and num_accepted_tokens.is_contiguous()
        assert num_accepted_tokens.dtype in (torch.int32, torch.int64)
        assert len(num_accepted_tokens) == len(cu_seqlens) - 1
    else:
        assert ssm_state_indices.ndim == 1
    assert len(ssm_state_indices) == len(cu_seqlens) - 1
    B, T, H, K, V = *k.shape, v.shape[-1]
    assert B == 1 and H in (8, 16) and K == V == 128 and v.shape[2] == 16
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32 if speculative else 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 1
    num_warps = 1

    o = q.new_empty(NK, *v.shape)
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = q.new_empty(T, HV, V, K, dtype=initial_state.dtype)

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid = (NK, 1, N * HV)
    # More than three statically expanded tokens causes pathological compiler
    # work on the pinned Ascend backend. Short graph-captured stages preserve
    # all candidate slots and continue directly from the preceding pool column.
    width = ssm_state_indices.shape[1] if speculative else 1
    for token_offset in range(0, width, 3):
        fused_recurrent_gated_delta_rule_fwd_kernel[grid](
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            o=o,
            h0=initial_state,
            ht=final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            scale=scale,
            N=N,
            T=T,
            B=B,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            stride_init_state_token=stride_init_state_token,
            stride_final_state_token=stride_final_state_token,
            stride_indices_seq=stride_indices_seq,
            stride_indices_tok=stride_indices_tok,
            IS_BETA_HEADWISE=beta.ndim == v.ndim,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            INPLACE_FINAL_STATE=inplace_final_state,
            IS_KDA=False,
            SINGLE_TOKEN=not speculative,
            QUERY_CAPACITY=min(3, width - token_offset),
            TOKEN_OFFSET=token_offset,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    o = o.squeeze(0)
    return o, final_state
