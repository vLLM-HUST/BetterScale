"""Keep immutable Conv1d weights in the native consumer's contiguous layout."""


def pack_conv_weights(model):
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention

    count = 0
    for layer in model.modules():
        if not isinstance(layer, QwenGatedDeltaNetAttention):
            continue
        # Ascend patches methods onto the donor Qwen class; it does not replace
        # instances with its implementation subclass.
        if (
            layer._forward_core.__func__
            is not AscendGatedDeltaNetAttention._forward_core
        ):
            raise ValueError("unqualified GDN consumer")
        weight = layer.conv1d.weight
        if weight.ndim != 3 or weight.shape[1] != 1 or weight.shape[2] != 4:
            raise ValueError(
                f"unqualified GDN convolution shape: {tuple(weight.shape)}"
            )
        # Preserve Parameter identity, logical values and public shape. The native
        # consumer's view(C, W).T is now contiguous without a per-step device copy.
        weight.data = weight.data.transpose(0, 2).contiguous().transpose(0, 2)
        if not weight.view(weight.shape[0], weight.shape[2]).T.is_contiguous():
            raise RuntimeError("native convolution weight view is not contiguous")
        count += 1
    if count != 48:
        raise ValueError(f"expected48 Qwen27 GDN layers, packed{count}")
    return count
