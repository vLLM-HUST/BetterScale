"""Layout-only fusion for packed mixed GDN; state pool and math stay unchanged."""
import torch
from vllm.triton_utils import triton, tl


@triton.jit
def cumulative_kernel(G, BETA, GH, BH, CU, INDICES, T: tl.constexpr):
    task = tl.program_id(0)
    seq = tl.load(INDICES + task * 2)
    chunk = tl.load(INDICES + task * 2 + 1)
    begin = tl.load(CU + seq)
    end = tl.load(CU + seq + 1)
    # Same 4x64 scan order as the pinned 256-token cumsum wrapper.
    token = begin + chunk * 256 + tl.arange(0, 256)
    head = tl.arange(0, 32)
    mask = (token[:, None] < end) & (head[None, :] < 24)
    g = tl.load(G + token[:, None] * 24 + head[None, :], mask, other=0)
    x = tl.trans(tl.reshape(g, (4, 64, 32)), (1, 0, 2))
    y = tl.cumsum(x, 0)
    y = tl.reshape(tl.trans(y, (1, 0, 2)), (256, 32))
    beta = tl.load(BETA + token[:, None] * 24 + head[None, :], mask, other=0)
    dest = head[None, :] * T + token[:, None]
    tl.store(GH + dest, y, mask)
    tl.store(BH + dest, beta, mask)


def cumulative_gates(beta, g, meta):
    t = g.shape[1]
    gh = torch.empty((1, 24, t), dtype=g.dtype, device=g.device)
    bh = torch.empty_like(gh, dtype=beta.dtype)
    cumulative_kernel[(len(meta.indices[256]),)](
        g, beta, gh, bh, meta.cu, meta.indices[256], t, num_warps=8)
    return bh, gh


@triton.jit
def restore_tiled(P, V, MAP, OUT, T: tl.constexpr, R: tl.constexpr):
    token = tl.program_id(0) * R + tl.arange(0, R)
    head = tl.program_id(1)
    d = tl.arange(0, 128)
    source = tl.load(MAP + token, token < T, other=0)
    spec = source >= T
    # Ascend may lower a vector gather mask to a post-load select. Keep BOTH
    # branches in bounds before masking; false lanes must not form negative V
    # offsets or beyond-capacity P addresses. Standalone allocator slack hid this.
    pre_source = tl.where(spec, 0, source)
    ver_source = tl.where(spec, source - T, 0)
    p = tl.load(P + (head * T + pre_source[:, None]) * 128 + d[None, :],
                (token[:, None] < T) & ~spec[:, None], other=0)
    v = tl.load(V + ver_source[:, None] * 3072 + head * 128 + d[None, :],
                (token[:, None] < T) & spec[:, None], other=0)
    tl.store(OUT + token[:, None] * 3072 + head * 128 + d[None, :],
             tl.where(spec[:, None], v, p), token[:, None] < T)
