"""Owned attention seam: native arithmetic, bootstrap-only plan, GM metadata.

Only the pinned BF16 paged causal non-FD FIA binary is admitted. The scoped
forward_impl override preserves native projection, RoPE and reshape/cache writes.
Plans/operands survive captures; one workspace is compute-stream serialized.
"""

import ctypes
import os
from contextlib import contextmanager
from functools import partial

import torch
import torch_npu


class StaticAttention:
    def __init__(self, root):
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

        self.root = root
        self.lib = ctypes.CDLL(os.environ["FIA_PLAN_LIBRARY"])
        self.lib.plan_begin.argtypes = [ctypes.c_uint64]
        self.lib.plan_workspace.argtypes = [ctypes.c_int]
        self.lib.plan_workspace.restype = ctypes.c_uint64
        self.lib.plan_bind.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
        self.lib.plan_launch.argtypes = [ctypes.c_int, ctypes.c_void_p]
        for name in (
            "plan_finish",
            "plan_clone",
            "plan_release",
            "plan_bind",
            "plan_launch",
            "plan_begin",
        ):
            getattr(self.lib, name).restype = ctypes.c_int
        self.lib.plan_clone.argtypes = self.lib.plan_release.argtypes = [ctypes.c_int]
        self.impls = [
            root.bundle.config.compilation_config.static_forward_context[name].impl
            for name in root.bundle.layer_names
        ]
        if not all(isinstance(impl, AscendAttentionBackendImpl) for impl in self.impls):
            raise RuntimeError(
                "static FIA requires the pinned Ascend attention implementation"
            )
        self.in_scope = False
        self.templates = {}
        self.plans = {}
        # Bounded scratch shared only by sequential layers/banks on compute.
        self.workspace = torch.zeros(
            128 * 1024**2, dtype=torch.uint8, device=root.bundle.device
        )
        self.bootstrap_calls = 0
        self.launch_calls = 0

    @staticmethod
    def check(rc, operation):
        if rc != 0:
            raise RuntimeError(f"native FIA {operation} failed: {rc}")

    @contextmanager
    def scope(self, key):
        if self.in_scope:
            raise RuntimeError("overlapping owned attention scopes")
        self.in_scope = True
        saved = []
        try:
            # Patch only donated instances, never the global backend class.
            for impl in self.impls:
                saved.append((impl, impl.__dict__.get("forward_impl")))
                impl.forward_impl = partial(self.forward, key, impl)
            yield
        finally:
            for impl, original in reversed(saved):
                if original is None:
                    del impl.forward_impl
                else:
                    impl.forward_impl = original
            self.in_scope = False

    def close(self):
        # Owner closes LiveModule graphs before releasing their native plans.
        for plan in [*self.plans.values(), *self.templates.values()]:
            self.check(self.lib.plan_release(plan), "release")
        self.plans.clear()
        self.templates.clear()

    def forward(self, frame_key, impl, query, key, value, kv_cache, m, output):
        from livemodule.runtime.phase import LivePhase, current_live_phase

        if impl.sinks is not None or impl.sliding_window is not None or not m.causal:
            raise RuntimeError(
                "static FIA supports causal attention without sinks/SWA only"
            )
        if query.dtype != torch.bfloat16 or impl.head_size != 128:
            raise RuntimeError("static FIA requires BF16/head128")
        key, value, block_size, table, lengths = impl._get_fia_params(
            key, value, m, kv_cache
        )
        if block_size != 128 or table is None:
            raise RuntimeError("static FIA requires paged128 KV")
        f = self.root.frames[frame_key]
        geometry = (
            frame_key,
            impl.num_heads,
            impl.num_kv_heads,
            tuple(key.shape),
            impl.scale,
        )
        if geometry not in self.templates:
            if current_live_phase() == LivePhase.CAPTURE:
                raise RuntimeError("plan missing before capture")
            torch.npu.synchronize()
            self.check(self.lib.plan_begin(query.data_ptr()), "bootstrap begin")
            bootstrap = torch_npu.npu_fused_infer_attention_score(
                query,
                key,
                value,
                atten_mask=m.attn_mask,
                block_table=table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=m.actual_seq_lengths_q,
                actual_seq_lengths_kv=lengths,
                num_heads=impl.num_heads,
                num_key_value_heads=impl.num_kv_heads,
                scale=impl.scale,
                sparse_mode=3,
                next_tokens=0,
            )
            torch.npu.synchronize()
            template = self.lib.plan_finish()
            if template < 0:
                raise RuntimeError(f"unqualified FIA launch: {template}, {geometry}")
            if not 0 < self.lib.plan_workspace(template) <= self.workspace.numel():
                self.lib.plan_release(template)
                raise RuntimeError("static FIA workspace exceeds owned capacity")
            self.templates[geometry] = template
            self.bootstrap_calls += 1
        slot = (frame_key, id(impl))
        if slot not in self.plans:
            if current_live_phase() == LivePhase.CAPTURE:
                raise RuntimeError("layer binding missing before capture")
            plan = self.lib.plan_clone(self.templates[geometry])
            if plan < 0:
                raise RuntimeError(f"FIA clone failed: {plan}")
            self.plans[slot] = plan
        plan = self.plans[slot]
        pointers = (ctypes.c_uint64 * 9)(
            *[
                t.data_ptr()
                for t in (
                    query,
                    m.attn_mask,
                    f["device_q_lengths"],
                    f["device_kv_lengths"],
                    table,
                    output,
                    self.workspace,
                    key,
                    value,
                )
            ]
        )
        self.check(self.lib.plan_bind(plan, pointers), "bind")
        self.check(
            self.lib.plan_launch(plan, torch.npu.current_stream().npu_stream), "launch"
        )
        self.launch_calls += 1
        return output
