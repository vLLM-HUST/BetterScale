"""Resident and candidate address publication; no model arithmetic."""

from vllm.triton_utils import triton, tl


@triton.jit(do_not_specialize=["N", "NP", "NV"])
def publish(
    SEATS,
    SELECTION,
    SEQ,
    CU,
    PRE_IDS,
    VER_IDS,
    INITIAL,
    PRE_CONV,
    VER_CONV,
    PRE_STATE,
    VER_SLOTS,
    VER_ACCEPTED,
    N,
    NP,
    NV,
    DECODE: tl.constexpr,
):
    row = tl.arange(0, 16)
    candidate = tl.arange(0, 4)
    if not DECODE:
        seq = tl.load(SEQ + row, row < N, other=0)
        begin = tl.load(CU + row, row < N, other=0)
        end = tl.load(CU + row + 1, row < N, other=0)
        tl.store(INITIAL + row, seq > end - begin, row < N)
        pre = tl.load(PRE_IDS + row, row < NP, other=0)
        seat = tl.load(SEATS + pre, row < NP, other=0)
        selected = tl.load(SELECTION + seat, row < NP, other=1) - 1
        seq = tl.load(SEQ + pre, row < NP, other=0)
        begin = tl.load(CU + pre, row < NP, other=0)
        end = tl.load(CU + pre + 1, row < NP, other=0)
        tl.store(PRE_CONV + pre, seat, row < NP)
        tl.store(PRE_STATE + 2 * row, seat * 3 + selected, row < NP)
        tl.store(PRE_STATE + 2 * row + 1, seq > end - begin, row < NP)
        ver = tl.load(VER_IDS + row, row < NV, other=0)
    else:
        ver = row
    seat = tl.load(SEATS + ver, row < NV, other=0)
    selected = tl.load(SELECTION + seat, row < NV, other=1)
    tl.store(VER_CONV + ver, seat, row < NV)
    tl.store(VER_ACCEPTED + row, selected, row < NV)
    tl.store(
        VER_SLOTS + row[:, None] * 3 + candidate[None, :],
        seat[:, None] * 3 + candidate[None, :],
        (row[:, None] < NV) & (candidate[None, :] < 3),
    )
