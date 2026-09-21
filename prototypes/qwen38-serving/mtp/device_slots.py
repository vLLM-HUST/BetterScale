"""One device-authoritative slot publication; no acceptance/length readback."""
from vllm.triton_utils import triton, tl


@triton.jit(do_not_specialize=["N", "NP", "NV"])
def slots_kernel(TABLE, SEQ, CU, PRE_IDS, VER_IDS, INITIAL, PRE_CONV,
                 VER_CONV, PRE_STATE, VER_SLOTS, N,
                 NP, NV, WIDTH: tl.constexpr,
                 STRIDE: tl.constexpr, BLOCK: tl.constexpr,
                 DECODE: tl.constexpr, W: tl.constexpr):
    row = tl.arange(0, 16)
    col = tl.arange(0, W)
    if DECODE:
        seq = tl.load(SEQ + row, row < N, other=1)
        block_col = tl.maximum(seq - 1, 0) // BLOCK
        slots = tl.load(TABLE + row[:, None] * STRIDE + block_col[:, None] + col[None, :],
                        (row[:, None] < N) & (col[None, :] < WIDTH), other=-1)
        tl.store(VER_SLOTS + row[:, None] * WIDTH + col[None, :], slots,
                 (row[:, None] < N) & (col[None, :] < WIDTH))
        first = tl.load(TABLE + row * STRIDE + block_col, row < N, other=-1)
        tl.store(VER_CONV + row, first, row < N)
    else:
        seq = tl.load(SEQ + row, row < N, other=0)
        start = tl.load(CU + row, row < N, other=0)
        end = tl.load(CU + row + 1, row < N, other=0)
        tl.store(INITIAL + row, seq > end - start, row < N)
        pre = tl.load(PRE_IDS + row, row < NP, other=0)
        ps = tl.load(SEQ + pre, row < NP, other=1)
        pc = tl.maximum(ps - 1, 0) // BLOCK
        slot = tl.load(TABLE + pre * STRIDE + pc, row < NP, other=-1)
        begin = tl.load(CU + pre, row < NP, other=0)
        finish = tl.load(CU + pre + 1, row < NP, other=0)
        tl.store(PRE_CONV + pre, slot, row < NP)
        tl.store(PRE_STATE + row * 2, slot, row < NP)
        tl.store(PRE_STATE + row * 2 + 1, ps > finish - begin, row < NP)
        ver = tl.load(VER_IDS + row, row < NV, other=0)
        vs = tl.load(SEQ + ver, row < NV, other=1)
        vc = tl.maximum(vs - 1, 0) // BLOCK
        slots = tl.load(TABLE + ver[:, None] * STRIDE + vc[:, None] + col[None, :],
                        (row[:, None] < NV) & (col[None, :] < WIDTH), other=-1)
        tl.store(VER_SLOTS + row[:, None] * WIDTH + col[None, :], slots,
                 (row[:, None] < NV) & (col[None, :] < WIDTH))
        first = tl.load(TABLE + ver * STRIDE + vc, row < NV, other=-1)
        tl.store(VER_CONV + ver, first, row < NV)


def publish_slots(meta):
    table, seq, block, pre, verify = meta.device_slot_source
    assert seq.device.type == 'npu' and seq.is_contiguous()
    assert table.stride(1) == 1 and 0 < meta.live <= 8
    # Unused arguments are valid pointers; constexpr removes that entire branch.
    dummy = meta.verify_conv
    slots_kernel[(1,)](
        table, seq, getattr(meta, 'cu', dummy), getattr(meta, 'prefill_ids', dummy),
        getattr(meta, 'verify_ids', dummy), getattr(meta, 'initial', dummy),
        getattr(meta, 'prefill_conv', dummy), meta.verify_conv,
        meta.prefill.state if not meta.decode else dummy, meta.verify.slots,
        meta.live, len(pre), len(verify), meta.width, table.stride(0), block,
        meta.decode, triton.next_power_of_2(meta.width), num_warps=4)
