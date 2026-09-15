"""Bounded BF16 Cube/Vector fusion probe, not a serving operator.

Use Triton-Ascend's tiled dot + epilogue mechanism to test the arithmetic and
compiler-managed relay before owning an AscendC ring-buffer implementation.
No claim that a LocalTensor or fused launch eliminates physical HBM traffic.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def gate_up_swiglu(
    X,
    W,
    Y,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    ntiles = tl.cdiv(I, BN)
    mtiles = tl.cdiv(M, BM)
    # One reusable relay allocation per physical Cube/Vector program, not tile.
    for pid in range(tl.program_id(0), tl.cdiv(M, BM) * ntiles, 24):
        rows = (pid % mtiles) * BM + tl.arange(0, BM)
        cols = (pid // mtiles) * (2 * BN) + tl.arange(0, 2 * BN)
        kk = tl.arange(0, BK)
        acc = tl.full((BM, 2 * BN), 0, tl.float32)
        for block in range(tl.cdiv(H, BK)):
            k = block * BK + kk
            x = tl.load(
                X + rows[:, None] * H + k[None, :],
                (rows[:, None] < M) & (k[None, :] < H),
                other=0,
            )
            w = tl.load(
                W + k[:, None] * (2 * I) + cols[None, :],
                (k[:, None] < H) & (cols[None, :] < 2 * I),
                other=0,
            )
            acc = tl.dot(x, w, acc)
        # Preserve the native BF16 gate/up materialization boundary, even fused.
        rounded = acc.to(tl.bfloat16).to(tl.float32)
        gate, up = tl.split(rounded.reshape(BM, BN, 2))
        activated = gate / (1.0 + tl.exp(-gate)) * up
        out_cols = (pid // mtiles) * BN + tl.arange(0, BN)
        tl.store(
            Y + rows[:, None] * I + out_cols[None, :],
            activated,
            (rows[:, None] < M) & (out_cols[None, :] < I),
        )


def pack(weight):
    """Native [gate half; up half] rows -> [H,I,gate/up] once, outside graph."""
    assert weight.ndim == 2 and weight.dtype == torch.bfloat16
    assert weight.shape[0] % 2 == 0
    i, h = weight.shape[0] // 2, weight.shape[1]
    return weight.reshape(2, i, h).permute(2, 1, 0).contiguous().reshape(h, 2 * i)


def fused(x, weight, output, tile=(64, 64, 512)):
    m, h = x.shape
    i = weight.shape[1] // 2
    assert x.dtype == weight.dtype == output.dtype == torch.bfloat16
    assert x.is_contiguous() and weight.is_contiguous() and output.is_contiguous()
    assert x.device == weight.device == output.device
    assert weight.shape[0] == h and output.shape == (m, i)
    assert 1 <= m <= 4096 and 1 <= h <= 5120 and 1 <= i <= 13824
    bm, bn, bk = tile
    gate_up_swiglu[(24,)](x, weight, output, m, h, i, bm, bn, bk)
    return output
