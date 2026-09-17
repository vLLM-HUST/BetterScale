"""All owned routes interpret the same K-V state pool, including uncaptured execution."""

import torch


def forward_core(self, mixed_qkv, b, a, core_attn_out):
    from vllm.forward_context import get_forward_context
    from vllm_ascend.device.device_op import DeviceOperator
    from vllm_ascend.ops.gdn import l2norm_fwd

    context = get_forward_context()
    if context.attn_metadata is None:
        return
    meta = context.attn_metadata[self.prefix].owned
    t = meta.tokens
    mixed_qkv, a, b = mixed_qkv[:t], a[:t], b[:t]
    conv, bank = self.kv_cache
    weights = self.conv1d.weight.view(
        self.conv1d.weight.size(0), self.conv1d.weight.size(2)
    ).T
    transformed = torch.empty_like(mixed_qkv)
    torch.ops._C_ascend.npu_causal_conv1d_custom(
        transformed,
        mixed_qkv,
        weights,
        conv_state=conv,
        bias_opt=self.conv1d.bias,
        query_start_loc_opt=meta.conv_cu,
        cache_indices_opt=meta.conv_slots,
        initial_state_mode_opt=None if meta.decode else meta.conv_initial,
        num_accepted_tokens_opt=None,
        activation_mode=1 if self.activation else 0,
        pad_slot_id=-1,
        run_mode=1 if meta.decode else 0,
    )
    q, k, v = self.rearrange_mixed_qkv(transformed)
    q, k = l2norm_fwd(q), l2norm_fwd(k)
    g, beta = DeviceOperator.fused_gdn_gating(self.A_log, a, b, self.dt_bias)
    if meta.decode:
        from .decode_kv import fused_recurrent_gated_delta_rule_fwd

        output, _ = fused_recurrent_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=128**-0.5,
            initial_state=bank,
            inplace_final_state=True,
            cu_seqlens=meta.cu,
            ssm_state_indices=meta.slots,
        )
    else:
        from vllm_ascend.ops.triton.fla import chunk

        g = chunk.chunk_local_cumsum(
            g, chunk_size=64, cu_seqlens=meta.cu, block_indices=meta.indices[256]
        )
        matrix = chunk.chunk_scaled_dot_kkt_fwd(
            k=k,
            beta=beta,
            g_cumsum=g,
            cu_seqlens=meta.cu,
            chunk_indices=meta.indices[64],
            output_dtype=torch.float32,
        )
        from .solve_tril import solve_tril

        matrix = solve_tril(
            matrix,
            cu_seqlens=meta.cu,
            chunk_indices_large_block=meta.indices[1216],
            chunk_indices_bt=meta.indices[64],
            output_dtype=k.dtype,
        )
        w, u = chunk.recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=matrix,
            g_cumsum=g,
            cu_seqlens=meta.cu,
            chunk_indices=meta.indices[64],
        )
        qh, kh, wh, uh, gh = [x.transpose(1, 2).contiguous() for x in (q, k, w, u, g)]
        output = (
            meta.engine.pool_forward(
                qh, kh, wh, uh, gh, bank, meta.cu, meta.state, meta.indices[64]
            )
            .transpose(1, 2)
            .contiguous()
        )
    core_attn_out[:t] = output.squeeze(0)


def install():
    # Import the donor patch first so it cannot subsequently overwrite this core.
    from vllm_ascend.patch.worker.patch_qwen3_5 import _GDN_PATCH_TARGET

    _GDN_PATCH_TARGET._forward_core = forward_core
