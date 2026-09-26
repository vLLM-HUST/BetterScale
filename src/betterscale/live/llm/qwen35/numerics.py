# SPDX-License-Identifier: Apache-2.0
# Projection/decoder ordering follows pinned vLLM Qwen3.5 (752a3a50) and
# vllm-ascend GDN (9bf964cb); State addressing is owned by BetterScale.
"""Numerical leaves borrowing owned State, never the native attention runner."""

import torch


def attention(module, state, hidden, positions, write_slots, read_slots):
    """Bounded eager oracle path over explicit physical token addresses.

    read_slots covers only this request's valid prefix plus this wave. This
    deliberately uses plain attention before adding a graph-compatible FIA port.
    It is not a high-throughput or maximum-context implementation.
    """
    qkv, _ = module.qkv_proj(hidden)
    q, k, v, gate = module._project_qkv_gate(qkv, positions)
    keys = state.key.tensor.flatten(0, 1)
    values = state.value.tensor.flatten(0, 1)
    keys.index_copy_(
        0, write_slots, k.reshape(-1, module.num_kv_heads, module.head_dim)
    )
    values.index_copy_(
        0, write_slots, v.reshape(-1, module.num_kv_heads, module.head_dim)
    )
    # Flattened token projections, but separate prefix axes: requests must
    # never attend across the batch. Single-request callers retain the same math.
    reads = read_slots.reshape(1, -1) if read_slots.ndim == 1 else read_slots
    batch, columns = reads.shape
    rows = hidden.shape[0]
    width = rows // batch
    shape = (batch, columns, module.num_kv_heads, module.head_dim)
    k = keys.index_select(0, reads.flatten()).reshape(shape).transpose(1, 2)
    v = values.index_select(0, reads.flatten()).reshape(shape).transpose(1, 2)
    q = q.reshape(batch, width, module.num_heads, module.head_dim).transpose(1, 2)
    repeat = module.num_heads // module.num_kv_heads
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    score = torch.matmul(q.float(), k.float().transpose(-1, -2)) * module.scaling
    pos = positions[0] if positions.ndim == 2 else positions
    mask = torch.arange(columns, device=hidden.device)[None, None, :] > (
        pos.reshape(batch, width, 1)
    )
    score.masked_fill_(mask[:, None], float("-inf"))
    output = torch.matmul(score.softmax(-1), v.float()).to(hidden.dtype)
    output = output.transpose(1, 2).reshape(rows, -1)
    if gate is not None:
        output = output * torch.sigmoid(gate)
    return module.o_proj(output)[0]


def gdn(module, state, hidden, geometry, seat, accepted, candidate_metadata=None):
    from vllm_ascend.device.device_op import DeviceOperator

    from .gdn_candidates import fused_recurrent_gated_delta_rule_fwd

    mixed, _ = module.in_proj_qkvz(hidden)
    qkv, z = mixed.split(
        [geometry.conv_channels, geometry.gdn_value_heads * geometry.gdn_value_dim], -1
    )
    z = z.reshape(hidden.shape[0], geometry.gdn_value_heads, geometry.gdn_value_dim)
    ba, _ = module.in_proj_ba(hidden)
    b, a = module.split_ba(ba)
    g, beta = DeviceOperator.fused_gdn_gating(
        module.A_log, a.contiguous(), b.contiguous(), module.dt_bias
    )
    y = torch.empty_like(qkv)
    if candidate_metadata is None:
        cu = torch.tensor([0, hidden.shape[0]], dtype=torch.int32, device=hidden.device)
        conv_slots = torch.tensor([[seat]], dtype=torch.int32, device=hidden.device)
        slots = torch.tensor(
            [[seat * 3 + i for i in range(3)]], dtype=torch.int64, device=hidden.device
        )
        selected = torch.tensor([accepted], dtype=torch.int32, device=hidden.device)
    else:
        cu, conv_slots, slots, selected = candidate_metadata
    weight = module.conv1d.weight.reshape(
        geometry.conv_channels, geometry.conv_kernel
    ).T
    torch.ops._C_ascend.npu_causal_conv1d_custom(
        y,
        qkv.contiguous(),
        weight,
        conv_state=state.conv.tensor,
        bias_opt=module.conv1d.bias,
        query_start_loc_opt=cu,
        cache_indices_opt=conv_slots,
        initial_state_mode_opt=None,
        num_accepted_tokens_opt=selected,
        activation_mode=1,
        pad_slot_id=-1,
        run_mode=1,
    )
    q, k, v = y.split(
        [geometry.gdn_key_heads * geometry.gdn_key_dim] * 2
        + [geometry.gdn_value_heads * geometry.gdn_value_dim],
        -1,
    )
    q = q.reshape(1, -1, geometry.gdn_key_heads, geometry.gdn_key_dim).contiguous()
    k = k.reshape_as(q).contiguous()
    v = v.reshape(1, -1, geometry.gdn_value_heads, geometry.gdn_value_dim).contiguous()
    out, _ = fused_recurrent_gated_delta_rule_fwd(
        q,
        k,
        v,
        g,
        beta,
        geometry.gdn_key_dim**-0.5,
        state.recurrent.tensor,
        cu_seqlens=cu,
        ssm_state_indices=slots,
        num_accepted_tokens=selected,
        use_qk_l2norm_in_kernel=True,
    )
    output = torch.empty_like(hidden)
    module._output_projection(out, z, output, hidden.shape[0])
    return output


def decoder(
    layer,
    state,
    hidden,
    residual,
    positions,
    write_slots,
    read_slots,
    geometry,
    seat,
    accepted,
    candidate_metadata=None,
):
    if layer.layer_scale:
        raise ValueError("layer scaling is outside this Qwen35 execution envelope")
    if residual is None:
        residual = hidden
        hidden = layer.input_layernorm(hidden)
    else:
        hidden, residual = layer.input_layernorm(hidden, residual)
    if layer.layer_type == "linear_attention":
        hidden = gdn(
            layer.linear_attn,
            state,
            hidden,
            geometry,
            seat,
            accepted,
            candidate_metadata,
        )
    else:
        hidden = attention(
            layer.self_attn, state, hidden, positions, write_slots, read_slots
        )
    hidden, residual = layer.post_attention_layernorm(hidden, residual)
    return layer.mlp(hidden), residual
