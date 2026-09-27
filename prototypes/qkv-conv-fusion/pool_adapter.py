"""Conv-only pool bridge for the qualified full-backing QKV producer.

Recurrent state never moves. The three Conv history rows are snapshotted before
any consumer writes. Inactive/sentinel rows cannot dereference or update a slot.
Speculative rows retain the native accepted-token update, not the prefill update.
"""
import torch
import qkv_fused  # registers the installed native binary
import triton
import triton.language as tl


@triton.jit
def history_read(POOL, SLOTS, INITIAL, CU, HISTORY,
                 N: tl.constexpr, STATE: tl.constexpr, BLOCK: tl.constexpr):
    seq = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(SLOTS + seq)
    live = tl.load(CU + seq + 1) > tl.load(CU + seq)
    initial = tl.load(INITIAL + seq) != 0
    value = tl.load(POOL + slot * STATE * N + d,
                    (d < 3 * N) & (slot >= 0) & live & initial, other=0)
    tl.store(HISTORY + seq * 3 * N + d, value, d < 3 * N)


@triton.jit
def history_write(TAIL, SLOTS, CU, POOL,
                  N: tl.constexpr, STATE: tl.constexpr, BLOCK: tl.constexpr):
    seq = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(SLOTS + seq)
    live = tl.load(CU + seq + 1) > tl.load(CU + seq)
    active = (d < 3 * N) & (slot >= 0) & live
    value = tl.load(TAIL + seq * 3 * N + d, active, other=0)
    tl.store(POOL + slot * STATE * N + d, value, active)


def eligible(tokens, hidden, channels):
    return (hidden, channels) == (2048, 4096) and tokens in (128,256,512,1024,2048,4096)


def project_conv(x, w, cw, pool, cu, prefill_slots, initial, verify_slots, accepted):
    """All allocations are graph-owned. Metadata values remain device-dynamic."""
    m, n, requests = x.shape[0], w.shape[0], cu.numel()-1
    assert eligible(m, x.shape[1], n), ("unsupported projection", tuple(x.shape), tuple(w.shape), tuple(pool.shape))
    assert pool.ndim == 3 and pool.shape[1] >= 3 and pool.shape[2] == n
    assert pool.is_contiguous() and prefill_slots.numel() == requests
    h = torch.empty((requests,3,n), dtype=x.dtype, device=x.device)
    s = torch.empty_like(h)
    z = torch.empty((m,n), dtype=x.dtype, device=x.device)
    y = torch.empty_like(z)
    grid = (requests, triton.cdiv(3*n,1024))
    history_read[grid](pool,prefill_slots,initial,cu,h,n,pool.shape[1],1024)
    torch.ops.qkv_fused._mixed_control(x,w,cw,h,z,y,s,cu,10)
    history_write[grid](s,prefill_slots,cu,pool,n,pool.shape[1],1024)
    # Prefill results are left intact: only verify_slots' active rows are written.
    torch.ops._C_ascend.npu_causal_conv1d_custom(
        y,z,cw,conv_state=pool,bias_opt=None,query_start_loc_opt=cu,
        cache_indices_opt=verify_slots,initial_state_mode_opt=None,
        num_accepted_tokens_opt=accepted,activation_mode=1,pad_slot_id=-1,run_mode=1)
    return y,z
