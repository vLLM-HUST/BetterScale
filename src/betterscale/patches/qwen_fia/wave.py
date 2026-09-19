"""Wave-shared native FIA plans and banked publication for Qwen27 mixed FULL.

One serial runner owns the planner. Graphs own invocation-local scratch through
the framework pool; frames own only persistent metadata and its reuse fences.
"""

import ctypes
import torch


class Planner:
    def __init__(self, library):
        self.lib = library
        u64 = ctypes.c_uint64
        i64 = ctypes.c_int64
        self.lib.plan_native_queries.argtypes = [
            ctypes.POINTER(u64),
            *[ctypes.c_int] * 6,
            ctypes.c_double,
            ctypes.POINTER(i64),
            ctypes.POINTER(i64),
            ctypes.c_void_p,
        ]
        self.lib.plan_bind.argtypes = [ctypes.c_int, ctypes.POINTER(u64)]
        self.lib.plan_bind_metadata.argtypes = [ctypes.c_int, u64]
        self.lib.plan_launch.argtypes = [ctypes.c_int, ctypes.c_void_p]
        self.lib.plan_metadata.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.plan_workspace.argtypes = [ctypes.c_int]
        self.lib.plan_workspace.restype = u64
        self.lib.plan_replace.argtypes = [ctypes.c_int, ctypes.c_int]
        self.fixtures = None
        self.calls = 0

    def check(self, rc):
        if rc < 0:
            raise RuntimeError(f"FIA plan error {rc}")
        return rc

    def status(self, rc):
        if rc != 0:
            raise RuntimeError(f"FIA launch/status error {rc}")

    def native(self, frame, m):
        q, k, v, out, scale = self.fixtures
        offsets = m.actual_seq_lengths_q
        lengths = m.seq_lens_list
        if (
            not 0 < len(offsets) == len(lengths) <= 9
            or not 0 < offsets[-1] <= frame.tokens
        ):
            raise ValueError("FIA request/query capacity exceeded")
        # Suppressed planner launch never reads this temporary framework allocation.
        scratch = torch.empty(128 << 20, dtype=torch.uint8, device=q.device)
        ptrs = (ctypes.c_uint64 * 7)(
            *[t.data_ptr() for t in (q, k, v, m.attn_mask, frame.table, out, scratch)]
        )
        plan = self.lib.plan_native_queries(
            ptrs,
            len(offsets),
            frame.tokens,
            12,
            2,
            k.shape[0],
            frame.columns,
            scale,
            (ctypes.c_int64 * len(lengths))(*lengths),
            (ctypes.c_int64 * len(offsets))(*offsets),
            torch.npu.current_stream().npu_stream,
        )
        if plan < 0:
            raise RuntimeError(
                f"FIA plan {plan}: cap={frame.tokens}, q={offsets}, kv={lengths}"
            )
        try:
            if self.lib.plan_is_fd(plan) != 0 or self.lib.plan_blocks(plan) != 24:
                raise RuntimeError("Unqualified Qwen256 FIA variant/grid")
            if self.lib.plan_metadata(plan, frame.h_tiling.data_ptr(), 2528) != 2528:
                raise RuntimeError("Unqualified FIA tiling layout")
            workspace = self.lib.plan_workspace(plan)
            if frame.plan is not None and workspace != frame.workspace:
                raise RuntimeError("FIA workspace changed after graph capture")
            frame.workspace = workspace
            if not 0 < frame.workspace <= 128 << 20:
                raise RuntimeError("FIA workspace exceeds admitted planner capacity")
            if frame.plan is None:
                frame.plan = plan
            else:
                self.status(self.lib.plan_replace(frame.plan, plan))
        except BaseException:
            self.lib.plan_release(plan)
            raise
        self.calls += 1


class Frame:
    def __init__(self, tokens, columns, device, stream):
        self.tokens = tokens
        self.columns = columns
        self.stream = stream
        size = 2528 + 9 * 16 + 9 * columns * 4
        self.host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        self.device = torch.empty(size, dtype=torch.uint8, device=device)

        def views(t):
            return (
                t[:2528],
                t[2528:2600].view(torch.int64),
                t[2600:2672].view(torch.int64),
                t[2672:].view(torch.int32).view(9, columns),
            )

        self.h_tiling, self.h_q, self.h_kv, self.h_table = views(self.host)
        self.tiling, self.q, self.kv, self.table = views(self.device)
        self.uploaded = torch.npu.Event()
        self.consumed = torch.npu.Event()
        self.has_upload = self.has_consumer = False
        self.plan = None
        self.stream.wait_stream(torch.npu.current_stream())

    def prepare(self, planner, m, cpu_table, num_reqs):
        if self.has_upload and not self.uploaded.query():
            self.uploaded.synchronize()
        self.h_q.zero_()
        self.h_kv.zero_()
        self.h_table.zero_()
        n = len(m.actual_seq_lengths_q)
        self.h_q.numpy()[:n] = m.actual_seq_lengths_q
        self.h_kv.numpy()[:n] = m.seq_lens_list
        self.h_table[:num_reqs].copy_(cpu_table[:num_reqs])
        planner.native(self, m)
        with torch.npu.stream(self.stream):
            if self.has_consumer:
                self.stream.wait_event(self.consumed)
            self.device.copy_(self.host, non_blocking=True)
            self.uploaded.record(self.stream)
        self.has_upload = True
        torch.npu.current_stream().wait_event(self.uploaded)

    def release(self):
        self.consumed.record(torch.npu.current_stream())
        self.has_consumer = True


def install(library):
    from vllm.forward_context import get_forward_context
    from vllm.config import CUDAGraphMode
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner as Runner
    from vllm_ascend.attention.attention_v1 import (
        AscendAttentionBackendImpl as Impl,
        AscendMetadata,
    )

    if getattr(Runner, "_betterscale_wave_fia", False):
        return
    original_forward = Runner._model_forward
    original_fia = Impl.forward_fused_infer_attention
    original_update = Runner._update_full_graph_params_if_needed
    active = {}

    def forward(self, *a, **kw):
        ctx = get_forward_context()
        if ctx.attn_metadata is None or (
            ctx.cudagraph_runtime_mode != CUDAGraphMode.FULL
            and getattr(self, "_owned_capture_bank", None) is None
        ):
            return original_forward(self, *a, **kw)
        metas = [m for m in ctx.attn_metadata.values() if isinstance(m, AscendMetadata)]
        if not metas:
            return original_forward(self, *a, **kw)
        m = metas[0]
        if not hasattr(self, "_fia_planner"):
            self._fia_planner = Planner(library)
            self._fia_frames = {}
        # The single full-attention KV group owns these canonical CPU block rows.
        layer = next(k for k, v in ctx.attn_metadata.items() if v is m)
        gid = next(
            i
            for i, g in enumerate(self.kv_cache_config.kv_cache_groups)
            if layer in g.layer_names
        )
        cpu = self.input_batch.block_table[gid].get_cpu_tensor()
        tokens = next(iter(self._owned_frame.metas.values())).tokens
        key = (tokens, self._owned_bank)
        if key not in self._fia_frames:
            self._fia_frames[key] = Frame(
                tokens, cpu.shape[1], self.device, self._owned_ingress
            )
        frame = self._fia_frames[key]
        planner = self._fia_planner
        if planner.fixtures is not None and not ctx.capturing:
            frame.prepare(
                planner,
                m,
                cpu,
                (
                    self.input_batch.num_reqs
                    if getattr(self, "_owned_capture_bank", None) is None
                    else min(len(m.actual_seq_lengths_q), cpu.shape[0])
                ),
            )
        if active:
            raise RuntimeError("Wave FIA requires a serial, non-reentrant runner")
        active.update(frame=frame, planner=planner, metadata=m, cpu=cpu, runner=self)
        self._fia_wave_active = True
        try:
            return original_forward(self, *a, **kw)
        finally:
            frame.release()
            active.clear()
            self._fia_wave_active = False

    def fia(self, query, key, value, m, output, kv_cache=None):
        if not active or (
            get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.NONE
            and getattr(active["runner"], "_owned_capture_bank", None) is None
        ):
            return original_fia(self, query, key, value, m, output, kv_cache)
        if (
            (self.num_heads, self.num_kv_heads, self.head_size) != (12, 2, 256)
            or self.sinks is not None
            or self.sliding_window is not None
            or not m.causal
        ):
            raise ValueError("Unqualified wave FIA attention geometry")
        key, value, block_size, _, _ = self._get_fia_params(key, value, m, kv_cache)
        if block_size != 128 or any(
            t.dtype != torch.bfloat16 or not t.is_contiguous()
            for t in (query, key, value, output)
        ):
            raise ValueError("Wave FIA requires contiguous BF16 paged-128 tensors")
        planner = active["planner"]
        frame = active["frame"]
        if planner.fixtures is None:
            if get_forward_context().capturing:
                raise RuntimeError("Warm the wave FIA planner before graph capture")
            # One independent query/output fixture, never retain graph intermediates.
            q = torch.empty((2048, 12, 256), device=query.device, dtype=query.dtype)
            planner.fixtures = (q, key, value, torch.empty_like(q), self.scale)
            frame.prepare(
                planner,
                m,
                active["cpu"],
                min(len(m.actual_seq_lengths_q), active["cpu"].shape[0]),
            )
        if self.scale != planner.fixtures[-1] or key.shape != planner.fixtures[1].shape:
            raise ValueError("Wave FIA layers must share scale and KV geometry")
        plan = planner.check(planner.lib.plan_clone(frame.plan))
        try:
            # Invocation-local allocator workspace; ACLGraph pool owns capture reuse.
            scratch = torch.empty(
                frame.workspace, dtype=torch.uint8, device=query.device
            )
            ptrs = (ctypes.c_uint64 * 9)(
                *[
                    t.data_ptr()
                    for t in (
                        query,
                        m.attn_mask,
                        frame.q,
                        frame.kv,
                        frame.table,
                        output,
                        scratch,
                        key,
                        value,
                    )
                ]
            )
            planner.status(planner.lib.plan_bind(plan, ptrs))
            planner.status(
                planner.lib.plan_bind_metadata(plan, frame.tiling.data_ptr())
            )
            planner.status(
                planner.lib.plan_launch(plan, torch.npu.current_stream().npu_stream)
            )
        finally:
            planner.status(planner.lib.plan_release(plan))
        return output

    def update(self, *a, **kw):
        if getattr(self, "_fia_wave_active", False):
            return
        return original_update(self, *a, **kw)

    Runner._model_forward = forward
    Impl.forward_fused_infer_attention = fia
    Runner._update_full_graph_params_if_needed = update

    def prime_startup():
        runner = active["runner"]
        if getattr(runner, "_owned_capture_bank", None) is None:
            return False
        if runner.input_batch.num_reqs:
            raise RuntimeError(
                "FIA graph priming requires an empty startup request pool"
            )
        return True

    install_replay_ordering(lambda: bool(active), prime_startup)
    Runner._betterscale_wave_fia = True


def install_replay_ordering(owned, prime_startup=None):
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context

    original = ACLGraphWrapper.__call__

    def call(self, *args, **kwargs):
        ordered = (
            owned()
            and self.runtime_mode == CUDAGraphMode.FULL
            and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
        )
        prime = False
        if ordered and prime_startup is not None and prime_startup():
            descriptor = get_forward_context().batch_descriptor
            entry = self.concrete_aclgraph_entries.get(descriptor)
            prime = entry is None or entry.aclgraph is None
        saved = self.enable_enpu
        if ordered:
            # In this pinned wrapper ONLY, this field selects caller-owned
            # replay ordering. No ENPU runner path is enabled. Old task updates
            # are gone; bank fences and the serial compute stream order reuse.
            self.enable_enpu = True
        try:
            result = original(self, *args, **kwargs)
            if prime:
                # Pay each bank's first-replay runtime setup before accepting
                # requests. Only disposable startup state exists here; ordinary
                # cold-request initialization still owns real cache/state rows.
                self.concrete_aclgraph_entries[descriptor].aclgraph.replay()
                torch.npu.current_stream().synchronize()
            return result
        finally:
            self.enable_enpu = saved

    ACLGraphWrapper.__call__ = call
