"""Keep immutable Conv1d weights in the native consumer's contiguous layout."""


def pack_conv_weights(model):
    from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention

    count = 0
    for layer in model.modules():
        if not isinstance(layer, AscendGatedDeltaNetAttention):
            continue
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
