"""One-shot graph/NONE state shadow; synchronization here is diagnostic only."""

import json
import os
from pathlib import Path
import torch
from bucket_full_worker import Worker as BaseWorker


def install_shadow():
    from vllm.forward_context import get_forward_context
    from vllm.config import CUDAGraphMode
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    original = NPUModelRunner._model_forward

    def forward(self, *args, **kwargs):
        if not getattr(self, "_shadow_remaining", 0):
            return original(self, *args, **kwargs)
        self._shadow_remaining -= 1
        ctx = get_forward_context()
        assert ctx.cudagraph_runtime_mode == CUDAGraphMode.FULL
        tensors = []

        def walk(x, name):
            if isinstance(x, torch.Tensor):
                tensors.append((name, x))
            elif isinstance(x, dict):
                for k, v in x.items():
                    walk(v, name + "." + str(k))
            elif isinstance(x, (list, tuple)):
                for i, v in enumerate(x):
                    walk(v, name + "." + str(i))

        walk(self.kv_caches, "cache")
        meta = []
        seen = set()
        for name, m in ctx.attn_metadata.items():
            if type(m).__name__ in seen:
                continue
            seen.add(type(m).__name__)
            fields = {}
            for k, v in vars(m).items():
                if isinstance(v, torch.Tensor) and v.numel() <= 16:
                    fields[k] = v.cpu().tolist()
                elif isinstance(v, (int, str, bool, float)) or v is None:
                    fields[k] = v
            meta.append(dict(layer=name, type=type(m).__name__, fields=fields))
        torch.npu.synchronize()
        before = [t.clone() for _, t in tensors]
        graph = original(self, *args, **kwargs).clone()
        torch.npu.synchronize()
        after = [t.clone() for _, t in tensors]
        for (_, t), b in zip(tensors, before):
            t.copy_(b)
        mode = ctx.cudagraph_runtime_mode
        checks = []
        try:
            ctx.cudagraph_runtime_mode = CUDAGraphMode.NONE
            eager = original(self, *args, **kwargs)
            torch.npu.synchronize()
            actual = self._shadow_actual_tokens
            pairs = [("valid_hidden", graph[:actual], eager[:actual])]
            pairs += [(name, a, t) for a, (name, t) in zip(after, tensors)]
            for name, a, b in pairs:
                delta = (a.float() - b.float()).abs()
                checks.append(
                    dict(
                        name=name,
                        shape=list(a.shape),
                        close=bool(
                            torch.allclose(a, b, atol=0.01, rtol=0.01, equal_nan=True)
                        ),
                        max_abs=float(delta.nan_to_num().max().item()),
                    )
                )
            result = dict(
                rank=self.rank,
                actual_tokens=actual,
                metadata=meta,
                checks=checks,
                passed=all(x["close"] for x in checks),
            )
            path = Path(os.environ["CAPSULE"]) / f"shadow-rank{self.rank}.json"
            path.write_text(json.dumps(result, indent=2))
            self._shadow_result = dict(
                rank=self.rank, passed=result["passed"], artifact=str(path)
            )
        finally:
            ctx.cudagraph_runtime_mode = mode
            for (_, t), a in zip(tensors, after):
                t.copy_(a)
            torch.npu.synchronize()
        return graph

    NPUModelRunner._model_forward = forward


class Worker(BaseWorker):
    def __init__(self, *args, **kwargs):
        install_shadow()
        super().__init__(*args, **kwargs)

    def arm_shadow(self, actual_tokens):
        self.model_runner._shadow_remaining = 1
        self.model_runner._shadow_actual_tokens = actual_tokens
        return dict(rank=self.rank, armed=True)

    def shadow_result(self):
        return self.model_runner._shadow_result
