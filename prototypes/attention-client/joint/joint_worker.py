"""Native Qwen oracle around prebuilt FULL attention and remote BF16 experts."""

import copy
import inspect
import json
import time
from pathlib import Path
import torch
from vllm.forward_context import get_forward_context, override_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.worker import NPUWorker
from metadata_graph import AttentionBank
from qwen_layer import split_forward
from scheduler import Lane, LayerScheduler
from common import CAP
import common
from transport import RemoteExperts


class JointWorker(NPUWorker):
    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        self.remote = None
        self.records = []
        self.events = []
        self.banks = {}
        self.preparing = True
        self.model_body = self.model_runner.model.model
        original = self.model_body.forward

        def forward(*args, **kwargs):
            if self.remote is None:
                return original(*args, **kwargs)
            ctx = get_forward_context()
            entry = ctx.moe_layer_index
            before = [
                (tensor, tensor.clone())
                for layer in self.model_body.layers
                for tensor in layer.self_attn.attn.kv_cache
            ]
            expected = original(*args, **kwargs).clone()
            exit_index = ctx.moe_layer_index
            caches = []
            for layer in self.model_body.layers:
                caches.extend((t, t.clone()) for t in layer.self_attn.attn.kv_cache)
            # Compare from the SAME pre-forward state. Leaving the native writes
            # installed could hide a missing KV update in the remote candidate.
            for tensor, snapshot in before:
                tensor.copy_(snapshot)
            del before
            context = copy.copy(ctx)
            context.moe_layer_index = entry
            context.attn_metadata = {
                key: copy.copy(value) for key, value in ctx.attn_metadata.items()
            }
            for metadata in context.attn_metadata.values():
                if metadata.attn_state == AscendAttentionState.PrefillNoCache:
                    metadata.attn_state = AscendAttentionState.ChunkedPrefill
            inputs = inspect.signature(split_forward).bind(
                self.model_body, *args, **kwargs
            )
            inputs.apply_defaults()
            positions = inputs.arguments["positions"]
            hidden = self.model_body.embed_input_ids(inputs.arguments["input_ids"])
            assert hidden.shape[0] <= CAP
            keys = list(context.attn_metadata)
            indices = {id(layer): i for i, layer in enumerate(self.model_body.layers)}
            replays = 0

            def attention(layer, pos, x, residual):
                nonlocal replays
                i = indices[id(layer)]
                md = context.attn_metadata[keys[i]]
                signature = (
                    i,
                    x.shape[0],
                    str(md.attn_state),
                    md.num_prefills,
                    md.num_decodes,
                )
                if signature not in self.banks:
                    if not self.preparing:
                        raise RuntimeError(f"unprepared attention bank: {signature}")
                    self.banks[signature] = AttentionBank(
                        self.model_runner, layer, keys[i], pos, x, residual
                    )
                replays += 1
                return self.banks[signature].replay(pos, x, residual)

            scheduler = LayerScheduler(self.remote, capacity=1)
            scheduler.admit(
                Lane(
                    "request-batch",
                    self.model_body,
                    positions,
                    hidden,
                    lambda: override_forward_context(context),
                    attention=attention,
                )
            )
            try:
                while scheduler.lanes:
                    if not scheduler.tick():
                        time.sleep(0.001)
                actual = scheduler.completed.popleft()[1]
                torch.npu.synchronize()
                # Different grouped-GEMM/reduction paths can differ by BF16 rounding.
                # Report exactness and normalized errors, not just an absolute pass.
                diff = (actual.float() - expected.float()).abs()
                rel_l2 = (
                    torch.linalg.vector_norm(diff)
                    / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
                ).item()
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=1e-6)
                assert rel_l2 < 0.01, rel_l2
                kv_exact = all(torch.equal(t, ref) for t, ref in caches)
                kv_errors = [
                    (t.float() - ref.float()).abs().max().item() for t, ref in caches
                ]
                for t, ref in caches:
                    torch.testing.assert_close(t, ref, rtol=0.02, atol=1e-6)
                self.records.append(
                    dict(
                        rows=hidden.shape[0],
                        preparing=self.preparing,
                        attention_replays=replays,
                        output_exact=torch.equal(actual, expected),
                        max_abs=diff.max().item(),
                        relative_l2=rel_l2,
                        kv_exact=kv_exact,
                        kv_max_abs=max(kv_errors),
                    )
                )
                Path(common.OUTPUT).write_text(
                    json.dumps(dict(checks=self.records, events=self.events), indent=2)
                )
                return actual
            finally:
                ctx.moe_layer_index = exit_index

        self.model_body.forward = forward
        return result

    def attach(self):
        assert common.LINKS is not None, "fixture requires in-process native engine"
        self.remote = RemoteExperts(self.model_body, common.LINKS, self.events)

    def seal(self):
        self.preparing = False
        return len(self.banks)

    def detach(self):
        self.remote.close()
        self.remote = None
