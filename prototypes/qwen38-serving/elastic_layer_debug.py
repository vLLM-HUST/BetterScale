"""Two-layer graph/eager boundary witness. Diagnostic copies are not shipped."""

import json
import os
from pathlib import Path
import torch


def install():
    from vllm.forward_context import get_forward_context
    from vllm_ascend.patch.worker.patch_qwen3_5 import _GDN_PATCH_TARGET

    original = _GDN_PATCH_TARGET._forward_core

    def forward(self, mixed_qkv, b, a, core_attn_out):
        out = core_attn_out
        ctx = get_forward_context()
        if ctx.attn_metadata is None or not any(
            f".layers.{i}." in self.prefix for i in (0, 4)
        ):
            return original(self, mixed_qkv, b, a, out)
        meta = ctx.attn_metadata[self.prefix].owned
        initial = self.kv_cache[1][meta.slots]
        values = [mixed_qkv, b, a, initial]
        capturing = ctx.capturing
        debug = getattr(self, "_owned_debug", {})
        if capturing:
            debug[meta.tokens] = [v.clone() for v in values]
            self._owned_debug = debug
        elif str(ctx.cudagraph_runtime_mode) == "NONE" and meta.tokens in debug:
            valid = int(meta.cu[-1].item())
            count = int((meta.cu.diff() > 0).sum().item())
            row = dict(
                layer=self.prefix,
                capacity=meta.tokens,
                tokens=valid,
                lengths=meta.cu.diff().cpu().tolist(),
                inputs=[],
            )
            for i, (x, y) in enumerate(zip(debug[meta.tokens], values)):
                n = count if i == 3 else valid
                row["inputs"].append(
                    dict(
                        max_abs=float(
                            (x[:n].float() - y[:n].float()).abs().nan_to_num().max()
                        ),
                        close=bool(
                            torch.allclose(
                                x[:n], y[:n], atol=0.01, rtol=0.01, equal_nan=True
                            )
                        ),
                    )
                )
        original(self, mixed_qkv, b, a, out)
        if capturing:
            debug[meta.tokens].append(out.clone())
        elif str(ctx.cudagraph_runtime_mode) == "NONE" and meta.tokens in debug:
            saved = debug[meta.tokens][-1][:valid]
            actual = out[:valid]
            row["output"] = dict(
                max_abs=float(
                    (saved.float() - actual.float()).abs().nan_to_num().max()
                ),
                close=bool(
                    torch.allclose(saved, actual, atol=0.01, rtol=0.01, equal_nan=True)
                ),
            )
            with (
                Path(os.environ["CAPSULE"])
                / f'layer-debug-rank{os.environ.get("RANK","unknown")}-pid{os.getpid()}.jsonl'
            ).open("a") as f:
                f.write(json.dumps(row) + "\n")

    _GDN_PATCH_TARGET._forward_core = forward
