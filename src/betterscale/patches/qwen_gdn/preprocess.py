# SPDX-License-Identifier: Apache-2.0
# Normalization/gating arithmetic follows the pinned vLLM FLA and Ascend donor.
"""Packed conv output -> Q/K norm, V and gates without intermediate cat/copies.

Keep the donor BF16 Q/K and beta rounding boundaries; recurrence is unchanged.
Explicit token/head axes avoid Ascend scalar gather lowering for head % 24.
The opt-in Q layout retains logical [1,T,8,128] but is physically head-major;
K stays token-major for the pinned KKT/WY consumers.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def preprocess_kernel(
    X,
    A,
    B,
    LOG,
    BIAS,
    Q,
    K,
    V,
    G,
    BETA,
    TOKEN_MAP,
    MAPPED: tl.constexpr,
    Q_HEAD_MAJOR: tl.constexpr,
    T: tl.constexpr,
    XS: tl.constexpr,
    AS: tl.constexpr,
    BS: tl.constexpr,
    R: tl.constexpr,
):
    token = tl.program_id(0) * R + tl.arange(0, R)
    source = tl.load(TOKEN_MAP + token, token < T, other=0) if MAPPED else token
    d = tl.arange(0, 128)[None, None, :]
    head = tl.arange(0, 8)[None, :, None]
    off = source[:, None, None] * XS + head * 128 + d
    mask = token[:, None, None] < T
    qx = tl.load(X + off, mask, other=0).to(tl.float32)
    kx = tl.load(X + off + 1024, mask, other=0).to(tl.float32)
    qy = qx * tl.rsqrt(tl.sum(qx * qx, 2) + 1.0e-6)[:, :, None]
    ky = kx * tl.rsqrt(tl.sum(kx * kx, 2) + 1.0e-6)[:, :, None]
    dest = token[:, None, None] * 1024 + head * 128 + d
    qdest = (head * T + token[:, None, None]) * 128 + d if Q_HEAD_MAJOR else dest
    tl.store(Q + qdest, qy, mask)
    tl.store(K + dest, ky, mask)
    vd = tl.arange(0, 4096)[None, :]
    vm = (token[:, None] < T) & (vd < 3072)
    value = tl.load(X + source[:, None] * XS + 2048 + vd, vm, other=0)
    tl.store(V + token[:, None] * 3072 + vd, value, vm)
    gh = tl.arange(0, 32)[None, :]
    gm = (token[:, None] < T) & (gh < 24)
    a = tl.load(A + source[:, None] * AS + gh, gm, other=0).to(tl.float32)
    b = tl.load(B + source[:, None] * BS + gh, gm, other=0).to(tl.float32)
    bias = tl.load(BIAS + gh, gh < 24, other=0).to(tl.float32)
    log = tl.load(LOG + gh, gh < 24, other=0).to(tl.float32)
    gate_input = a + bias
    softplus = tl.where(
        gate_input <= 20.0, tl.log(1.0 + tl.exp(gate_input)), gate_input
    )
    tl.store(G + token[:, None] * 24 + gh, -tl.exp(log) * softplus, gm)
    tl.store(BETA + token[:, None] * 24 + gh, tl.sigmoid(b), gm)


def preprocess(x, a, b, log, bias, token_map=None, *, q_head_major=False):
    source_tokens = x.shape[0]
    t = source_tokens if token_map is None else token_map.numel()
    assert x.shape == (source_tokens, 5120) and 0 < t <= 2048
    assert a.shape == b.shape == (source_tokens, 24)
    if token_map is not None:
        assert token_map.ndim == 1 and token_map.is_contiguous()
        assert token_map.device == x.device and token_map.dtype in (torch.int32, torch.int64)
    assert log.shape == bias.shape == (24,)
    assert all(v.dtype == torch.bfloat16 for v in (x, a, b))
    assert all(v.device == x.device and v.stride(-1) == 1 for v in (a, b, log, bias))
    assert x.stride(-1) == 1
    q_shape = (1, 8, t, 128) if q_head_major else (1, t, 8, 128)
    q = torch.empty(q_shape, dtype=x.dtype, device=x.device)
    if q_head_major:
        q = q.transpose(1, 2)
    k = torch.empty((1, t, 8, 128), dtype=x.dtype, device=x.device)
    v = torch.empty((1, t, 24, 128), dtype=x.dtype, device=x.device)
    g = torch.empty((1, t, 24), dtype=torch.float32, device=x.device)
    beta = torch.empty_like(g, dtype=b.dtype)
    rows = 1 if t <= 16 else 4
    preprocess_kernel[(triton.cdiv(t, rows),)](
        x,
        a,
        b,
        log,
        bias,
        q,
        k,
        v,
        g,
        beta,
        token_map,
        token_map is not None,
        q_head_major,
        t,
        x.stride(0),
        a.stride(0),
        b.stride(0),
        rows,
        num_warps=4,
    )
    return q, k, v, g, beta
