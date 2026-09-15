"""Fixed-context FULL segment capture oracle; not a dynamic metadata catalog.

Native eager reference constructs the metadata. This probe holds those exact
values/addresses while replaying every segment three times. It explicitly does
NOT qualify changing context lengths, requests or block tables across replays.
"""
import torch
from qwen_layer import attention_half


def capture_segments(model, input_ids, positions, intermediate_tensors=None,
                     inputs_embeds=None, **kwargs):
    if intermediate_tensors is not None or kwargs:
        raise NotImplementedError('ordinary TP1/PP1 only')
    initial = inputs_embeds if inputs_embeds is not None else model.embed_input_ids(input_ids)
    hidden, residual = initial, None
    segments = []
    for layer in model.layers:
        # Bank-private stable input addresses. Do not alias another lane's IO.
        x = hidden.clone()
        r = residual.clone() if residual is not None else None
        # Warm original operator route, then capture its entire attention half.
        attention_half(layer, positions, x, r)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        # RMSNorm may mutate residual; restore all bank inputs after warmup.
        x.copy_(hidden)
        if r is not None: r.copy_(residual)
        with torch.npu.graph(graph):
            pending = attention_half(layer, positions, x, r)
        x.copy_(hidden)
        if r is not None: r.copy_(residual)
        graph.replay()
        # Reference expert retained locally; server transport will replace this.
        hidden = layer.mlp(pending.normalized)
        residual = pending.residual
        segments.append((layer, x, r, graph, pending))
    return initial, segments


def replay_segments(model, initial, segments):
    hidden, residual = initial, None
    for layer, x, r, graph, pending in segments:
        x.copy_(hidden)
        if r is not None: r.copy_(residual)
        graph.replay()
        hidden = layer.mlp(pending.normalized)
        residual = pending.residual
    hidden, _ = model.norm(hidden, residual)
    return hidden
