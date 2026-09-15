"""Two native graph lanes with private KV and an intentionally late completion.

This is a same-device reference-MLP experiment, not the remote INT32 server.
Frozen snapshots have disjoint KV backing, metadata and graph IO. One is prefill,
the other decode. Suppressing one completion tests progress, not real latency.
"""

import copy
import inspect
import torch
from vllm.forward_context import get_forward_context, override_forward_context
from metadata_graph import metadata_forward, clone_cache
from qwen_layer import split_forward
from scheduler import Lane, LayerScheduler, StreamReferenceTransport


def snapshot(runner, model, expected, args, kwargs):
    context = copy.copy(get_forward_context())
    context.attn_metadata = copy.deepcopy(context.attn_metadata)
    if hasattr(context, "slot_mapping"):
        context.slot_mapping = copy.deepcopy(context.slot_mapping)
    context.moe_layer_index = 0
    args = copy.deepcopy(args)
    kwargs = copy.deepcopy(kwargs)
    banks = {}
    runner.attention_private_kv = True
    try:
        with override_forward_context(context):
            actual, _ = metadata_forward(runner, banks, model, *args, **kwargs)
        assert torch.equal(actual, expected), "private KV snapshot differs"
    finally:
        torch.npu.synchronize()
        runner.attention_private_kv = False
    private = [b.private_cache for b in banks.values()]
    return dict(
        context=context,
        args=args,
        kwargs=kwargs,
        banks=banks,
        expected=expected.clone(),
        private=private,
        reference=clone_cache(private),
    )


def run_pair(model, snapshots):
    order = []

    class Delayed(StreamReferenceTransport):
        def __init__(self):
            super().__init__()
            self.owners = {}
            self.fast_done = False

        def submit(self, identity, *args):
            h = super().submit(identity, *args)
            self.owners[h] = identity
            order.append(("submit", identity, args[0]))
            return h

        def poll(self, handle):
            if self.owners[handle] == "prefill" and not self.fast_done:
                return None
            result = super().poll(handle)
            if result is not None:
                order.append(("retire", self.owners.pop(handle)))
            return result

    transport = Delayed()
    scheduler = LayerScheduler(transport, capacity=2)
    for identity, snap in snapshots.items():
        context = snap["context"]
        context.moe_layer_index = 0
        inputs = inspect.signature(split_forward).bind(
            model, *snap["args"], **snap["kwargs"]
        )
        inputs.apply_defaults()
        positions = inputs.arguments["positions"]
        hidden = inputs.arguments["inputs_embeds"]
        if hidden is None:
            hidden = model.embed_input_ids(inputs.arguments["input_ids"])
        by_layer = {b.key: b for b in snap["banks"].values()}

        def attention(layer, pos, x, residual, banks=by_layer):
            matches = [b for b in banks.values() if b.layer is layer]
            assert len(matches) == 1
            return matches[0].replay(pos, x, residual)

        scheduler.admit(
            Lane(
                identity,
                model,
                positions,
                hidden,
                lambda ctx=context: override_forward_context(ctx),
                attention=attention,
            )
        )
    outputs = {}
    while scheduler.lanes:
        if not scheduler.tick():
            transport.wait_for_progress()
        while scheduler.completed:
            identity, output = scheduler.completed.popleft()
            outputs[identity] = output.clone()
            order.append(("finished", identity))
            if identity == "decode":
                assert scheduler.lanes["prefill"].layer == 0
                transport.fast_done = True
    torch.npu.synchronize()
    for identity, snap in snapshots.items():
        assert torch.equal(outputs[identity], snap["expected"]), identity
        for current, reference in zip(snap["private"], snap["reference"]):
            assert all(
                torch.equal(k, r) for k, r in zip(current, reference)
            ), "private KV differs"
    return dict(
        order=order,
        output_exact=True,
        private_kv_exact=True,
        late_prefill_did_not_block_decode=True,
    )
