"""Native host attention planning once per wave, shared by every captured layer.

FIA numerical launch is suppressed during planning. The installed native planner
owns FD selection, splits and reduction layout. Separate FD/non-FD graph entries
use banked GM tiling; empty extra blocks give both variants fixed24-block launches.
"""

import ctypes
from collections import Counter

import torch
from static_attention import StaticAttention


class HostAttention(StaticAttention):
    def __init__(self, root):
        super().__init__(root)
        u64 = ctypes.c_uint64
        self.lib.plan_native_queries.argtypes = [
            ctypes.POINTER(u64),
            *[ctypes.c_int] * 6,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_void_p,
        ]
        self.lib.plan_metadata.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.plan_bind_metadata.argtypes = [ctypes.c_int, u64]
        for name in (
            "plan_native_queries",
            "plan_metadata",
            "plan_pad_blocks",
            "plan_blocks",
            "plan_is_fd",
            "plan_bind_metadata",
        ):
            getattr(self.lib, name).restype = ctypes.c_int
        self.fixtures = {}
        self.warm_plans = {}
        self.wave_plans = 0
        self.dispatches = Counter()
        geometries = {
            (i.num_heads, i.num_kv_heads, i.head_size, i.scale) for i in self.impls
        }
        if len(geometries) != 1:
            raise RuntimeError(
                "wave-shared FIA requires uniform layer attention geometry"
            )

    def fd_eligible(self, kind, count):
        impl = self.impls[0]
        width = 1 if kind == "d" else count
        rows = count if kind == "d" else 1
        return (
            self.root.max_length >= 2048
            and width <= 16
            and width * impl.num_heads // impl.num_kv_heads <= 128
            and rows * impl.num_kv_heads <= 0.8 * 24
        )

    def native(self, base, lengths, query_tokens=None):
        query, key, value, mask, table, output, impl, rows, width = self.fixtures[base]
        if query_tokens is not None and (rows != 1 or not 1 <= query_tokens <= width):
            raise ValueError("invalid actual prefill query length")
        offsets = (
            [query_tokens]
            if query_tokens is not None
            else [(i + 1) * width for i in range(rows)]
        )
        ptrs = (ctypes.c_uint64 * 7)(
            *[
                t.data_ptr()
                for t in (query, key, value, mask, table, output, self.workspace)
            ]
        )
        plan = self.lib.plan_native_queries(
            ptrs,
            rows,
            width,
            impl.num_heads,
            impl.num_kv_heads,
            key.shape[0],
            table.shape[1],
            impl.scale,
            (ctypes.c_int64 * rows)(*lengths),
            (ctypes.c_int64 * rows)(*offsets),
            torch.npu.current_stream().npu_stream,
        )
        if plan < 0:
            raise RuntimeError(
                f"native wave FIA planner failed: {plan}, {base}, {lengths}"
            )
        try:
            if not 0 < self.lib.plan_workspace(plan) <= self.workspace.numel():
                raise RuntimeError("native wave FIA workspace exceeds owned capacity")
            blocks = self.lib.plan_blocks(plan)
            self.check(self.lib.plan_pad_blocks(plan), "pad empty blocks")
            return plan, blocks
        except BaseException:
            self.lib.plan_release(plan)
            raise

    def export(self, plan):
        host = torch.empty(2528, dtype=torch.uint8, pin_memory=True)
        size = self.lib.plan_metadata(plan, host.data_ptr(), host.numel())
        if size != host.numel():
            raise RuntimeError(f"unqualified native tiling payload: {size}")
        return host

    def prepare(self, base, lengths, query_tokens=None):
        plan, blocks = self.native(base, lengths, query_tokens)
        try:
            fd = self.lib.plan_is_fd(plan)
            key = base + ("fd" if fd else "")
            if key not in self.root.frames:
                raise RuntimeError(f"native FIA selected uncaptured variant: {key}")
            host = self.export(plan)
        finally:
            self.check(self.lib.plan_release(plan), "release wave plan")
        self.wave_plans += 1
        self.dispatches[f"{fd}:{blocks}"] += 1
        return key, host

    def forward(self, frame_key, impl, query, key, value, kv_cache, m, output):
        from livemodule.runtime.phase import LivePhase, current_live_phase

        if impl.sinks is not None or impl.sliding_window is not None or not m.causal:
            raise RuntimeError("host-planned FIA supports causal no-sinks/no-SWA only")
        if query.dtype != torch.bfloat16 or impl.head_size != 128:
            raise RuntimeError("host-planned FIA requires BF16/head128")
        key, value, block_size, table, lengths = impl._get_fia_params(
            key, value, m, kv_cache
        )
        if (
            block_size != 128
            or table is None
            or not key.is_contiguous()
            or not value.is_contiguous()
        ):
            raise RuntimeError("host-planned FIA requires contiguous paged128 KV")
        f = self.root.frames[frame_key]
        base = frame_key.removesuffix("fd")
        capture = current_live_phase() == LivePhase.CAPTURE
        if base not in self.fixtures:
            if capture:
                raise RuntimeError("missing host planner fixture before capture")
            rows = self.root.residents if f["kind"] == "d" else 1
            width = 1 if f["kind"] == "d" else f["count"]
            self.fixtures[base] = (
                query,
                key,
                value,
                m.attn_mask,
                table,
                output,
                impl,
                rows,
                width,
            )
        if frame_key not in self.templates:
            if capture:
                raise RuntimeError("missing native plan before capture")
            rows, width = self.fixtures[base][-2:]
            seed_lengths = (
                [min(8192, self.root.max_length)] * rows if f["fd"] else [width] * rows
            )
            template, _ = self.native(base, seed_lengths)
            if bool(self.lib.plan_is_fd(template)) != f["fd"]:
                self.lib.plan_release(template)
                raise RuntimeError(f"native bootstrap variant mismatch: {frame_key}")
            self.templates[frame_key] = template
            f["tiling"].copy_(self.export(template))
            # Warmup does not represent a live long-context request. Use the
            # original short non-FD plan there; only CAPTURE binds the FD kernel.
            warm, _ = self.native(base, [width] * rows)
            self.warm_plans[frame_key] = warm
            self.bootstrap_calls += 1
        slot = (frame_key, id(impl))
        if slot not in self.plans:
            if capture:
                raise RuntimeError("missing layer binding before capture")
            plan = self.lib.plan_clone(self.templates[frame_key])
            if plan < 0:
                raise RuntimeError(f"native clone failed: {plan}")
            self.plans[slot] = plan
        plan = self.plans[slot] if capture else self.warm_plans[frame_key]
        ptrs = (ctypes.c_uint64 * 9)(
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
        self.check(self.lib.plan_bind(plan, ptrs), "bind")
        if capture:
            self.check(
                self.lib.plan_bind_metadata(plan, f["tiling"].data_ptr()),
                "bind GM tiling",
            )
        self.check(
            self.lib.plan_launch(plan, torch.npu.current_stream().npu_stream), "launch"
        )
        self.launch_calls += 1
        if f["kind"] == "p":
            # FIA writes only the live prefix. Never feed uninitialized padding
            # into downstream projection/router/GMM, even though KV ignores it.
            output.masked_fill_(
                f["padding_mask"].view(-1, *([1] * (output.ndim - 1))), 0
            )
        return output

    def close(self):
        super().close()
        for plan in self.warm_plans.values():
            self.check(self.lib.plan_release(plan), "release warmup")
        self.warm_plans.clear()
        self.fixtures.clear()
