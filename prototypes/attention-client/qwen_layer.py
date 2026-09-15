"""The native Qwen layer cut; no global monkey-patch or EP implementation.

The returned normalized input and residual must stay alive across remote work.
Native sparse MLP owns router normalization and quantization in the reference
path. A real transport must preserve those contracts, not use an ad-hoc topk.
"""

from dataclasses import dataclass


@dataclass
class LayerInput:
    normalized: object
    residual: object


def attention_half(layer, positions, hidden, residual):
    if residual is None:
        residual = hidden
        hidden = layer.input_layernorm(hidden)
    else:
        hidden, residual = layer.input_layernorm(hidden, residual)
    hidden = layer.self_attn(positions=positions, hidden_states=hidden)
    hidden, residual = layer.post_attention_layernorm(hidden, residual)
    return LayerInput(hidden, residual)


def reference_expert(layer, pending):
    # Includes the model's actual router and optional shared expert exactly once.
    return layer.mlp(pending.normalized), pending.residual


def split_forward(
    model, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs
):
    if intermediate_tensors is not None or kwargs:
        raise NotImplementedError("first prototype: TP1/PP1 ordinary Qwen only")
    if model.start_layer != 0 or model.end_layer != len(model.layers):
        raise NotImplementedError("pipeline parallel model is not supported")
    hidden = (
        inputs_embeds if inputs_embeds is not None else model.embed_input_ids(input_ids)
    )
    residual = None
    for layer in model.layers:
        pending = attention_half(layer, positions, hidden, residual)
        hidden, residual = reference_expert(layer, pending)
    hidden, _ = model.norm(hidden, residual)
    return hidden
