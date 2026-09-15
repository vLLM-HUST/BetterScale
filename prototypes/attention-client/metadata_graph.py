"""Bank-private native attention task-update handles, TP1 prototype only.

Uses native FIA/PA metadata update, not frozen Python sequence-length lists.
Host access is serialized; swapping the native graph-parameter registry is NOT
thread-safe. Each bank owns its capture IO, events and workspace. Requests may
only reuse a bank after the prior invocation has completed.
"""

import copy
from contextlib import contextmanager, nullcontext
from unittest.mock import patch
import torch
from vllm.forward_context import get_forward_context, override_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention import attention_v1
from vllm_ascend.compilation import acl_graph
from qwen_layer import attention_half


def clone_cache(cache):
    # Native cache is a tuple of typed views over a raw Char allocation.
    # deepcopy follows the typed-storage wrapper and fails on this alias;
    # clone each logical tensor instead (this fixture has separate K/V views).
    if isinstance(cache, torch.Tensor):
        return cache.clone()
    if isinstance(cache, tuple):
        return tuple(clone_cache(v) for v in cache)
    if isinstance(cache, list):
        return [clone_cache(v) for v in cache]
    raise TypeError(f"unexpected KV container: {type(cache)}")


class AttentionBank:
    def __init__(self, runner, layer, key, positions, hidden, residual):
        self.runner, self.layer, self.key = runner, layer, key
        self.tokens = hidden.shape[0]
        self.update_stream = torch.npu.Stream()
        self.params = acl_graph.GraphParams(
            {self.tokens: []}, {self.tokens: None}, {self.tokens: []}, {self.tokens: []}
        )
        self.positions = positions.clone()
        self.hidden = hidden.clone()
        self.residual = None if residual is None else residual.clone()
        self.graph = torch.npu.NPUGraph()
        self.private_cache = None
        if getattr(runner, "attention_private_kv", False):
            self.private_cache = clone_cache(layer.self_attn.attn.kv_cache)
        # Warm the original attention route outside capture.
        with self.cache_scope():
            attention_half(layer, self.positions, self.hidden, self.residual)
        torch.npu.synchronize()
        self.copy_inputs(positions, hidden, residual)
        context = copy.copy(get_forward_context())
        self.metadata = copy.deepcopy(context.attn_metadata[key])
        context.attn_metadata = {key: self.metadata}
        old_capture = context.capturing
        try:
            with self.registry(), self.cache_scope(), override_forward_context(context):
                context.capturing = True
                with torch.npu.graph(self.graph):
                    self.pending = attention_half(
                        layer, self.positions, self.hidden, self.residual
                    )
        finally:
            context.capturing = old_capture
        assert (
            len(self.params.handles[self.tokens]) == 1
        ), "one native attention task expected"

    @contextmanager
    def cache_scope(self):
        if self.private_cache is None:
            yield
            return
        attn = self.layer.self_attn.attn
        with (
            patch.object(attn, "kv_cache", self.private_cache),
            patch.object(attn.impl, "key_cache", self.private_cache[0]),
            patch.object(attn.impl, "value_cache", self.private_cache[1]),
        ):
            yield

    @contextmanager
    def registry(self):
        previous = acl_graph._graph_params
        previous_keys = attention_v1._ATTN_KEYS_BUFFER
        attention_v1._ATTN_KEYS_BUFFER = [self.key]
        acl_graph._graph_params = self.params
        try:
            yield
        finally:
            acl_graph._graph_params = previous
            attention_v1._ATTN_KEYS_BUFFER = previous_keys

    def copy_inputs(self, positions, hidden, residual):
        self.positions.copy_(positions)
        self.hidden.copy_(hidden)
        if self.residual is not None:
            self.residual.copy_(residual)

    def replay(self, positions, hidden, residual):
        self.copy_inputs(positions, hidden, residual)
        context = copy.copy(get_forward_context())
        incoming = context.attn_metadata[self.key]
        for name, value in vars(incoming).items():
            retained = getattr(self.metadata, name)
            if isinstance(value, torch.Tensor):
                if (
                    not isinstance(retained, torch.Tensor)
                    or retained.shape != value.shape
                ):
                    raise ValueError(f"metadata bank shape changed: {name}")
                retained.copy_(value)
            else:
                setattr(self.metadata, name, copy.deepcopy(value))
        context.attn_metadata = {self.key: self.metadata}
        # Metadata copies precede updater consumption even on a distinct stream.
        ready = torch.npu.Event()
        ready.record()
        self.update_stream.wait_event(ready)
        with self.registry(), override_forward_context(context):
            # Retain native retiling and data-before-event publication protocol.
            acl_graph.update_full_graph_params(
                self.runner.attn_backend,
                self.update_stream,
                context,
                self.tokens,
                self.runner.vllm_config,
                self.runner.speculative_config,
            )
        self.graph.replay()
        return self.pending


def _metadata_forward(
    runner,
    banks,
    model,
    input_ids,
    positions,
    intermediate_tensors=None,
    inputs_embeds=None,
    **kwargs,
):
    if intermediate_tensors is not None or kwargs:
        raise NotImplementedError("TP1 PP1 only")
    ctx = get_forward_context()
    hidden = (
        inputs_embeds if inputs_embeds is not None else model.embed_input_ids(input_ids)
    )
    residual = None
    keys = list(ctx.attn_metadata)
    assert len(keys) == len(model.layers)
    from scheduler import Lane, LayerScheduler, StreamReferenceTransport

    reused = 0
    layer_index = {id(layer): i for i, layer in enumerate(model.layers)}

    def attention(layer, positions, hidden, residual):
        nonlocal reused
        i = layer_index[id(layer)]
        metadata = ctx.attn_metadata[keys[i]]
        signature = (
            i,
            hidden.shape[0],
            tuple(positions.shape),
            str(metadata.attn_state),
            metadata.num_prefills,
            metadata.num_decodes,
        )
        if signature not in banks:
            banks[signature] = AttentionBank(
                runner, layer, keys[i], positions, hidden, residual
            )
        else:
            reused += 1
        return banks[signature].replay(positions, hidden, residual)

    transport = StreamReferenceTransport()
    scheduler = LayerScheduler(transport, capacity=1)
    lane = Lane(
        "native-batch",
        model,
        positions,
        hidden,
        lambda: override_forward_context(ctx),
        attention=attention,
    )
    scheduler.admit(lane)
    while scheduler.lanes:
        if not scheduler.tick():
            transport.wait_for_progress()
    _, output = scheduler.completed.popleft()
    return output, reused


def metadata_forward(runner, banks, model, *args, **kwargs):
    original = get_forward_context()
    context = copy.copy(original)
    context.attn_metadata = {}
    for key, value in original.attn_metadata.items():
        value = copy.copy(value)
        if value.attn_state == AscendAttentionState.PrefillNoCache:
            # Native FULL uses paged cache inputs; the direct no-cache route
            # supplies None block_table, which native capture cannot weak-ref.
            # This changes the attention execution path and needs the ORIGINAL
            # eager oracle, not merely a same-normalized-program comparison.
            value.attn_state = AscendAttentionState.ChunkedPrefill
        context.attn_metadata[key] = value
    try:
        with override_forward_context(context):
            return _metadata_forward(runner, banks, model, *args, **kwargs)
    finally:
        original.moe_layer_index = context.moe_layer_index
