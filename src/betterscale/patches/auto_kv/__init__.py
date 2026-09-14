"""Physical KV budgeting, using the same pool for trial and final graphs.

Imported without touching native runtime state. Worker composes this mixin;
manual KV byte budgets still use the native allocator. No launch environment,
scheduler, cache layout or public artifact directory is installed here.
"""

import dataclasses
import gc
import logging

log = logging.getLogger("vllm.betterscale.memory")

GiB = 1 << 30


def clear_trial_graphs(wrappers, retained_graphs=None):
    import torch
    from vllm_ascend.compilation import acl_graph

    # The target trial catalogs live in these wrappers; draft retirement is
    # handled first by _draft.retire_trial. DP auxiliary programs are installed
    # by Worker after final native warmup, not in the trial.
    torch.npu.synchronize()
    for wrapper in wrappers:
        # HCCL AIV recapture can still depend on graph-owned resources. Retain
        # only graph handles, not old State/input/output packets. These handles
        # must NEVER replay after the trial addresses have been retired.
        if retained_graphs is not None:
            catalogs = [wrapper.concrete_aclgraph_entries]
            pair = wrapper.__dict__.get("_decode_pair")
            if pair is not None:
                catalogs.extend(pair.catalogs)
            for catalog in catalogs:
                for entry in catalog.values():
                    if entry.aclgraph is not None and all(
                        entry.aclgraph is not graph for graph in retained_graphs
                    ):
                        retained_graphs.append(entry.aclgraph)
        pair = wrapper.__dict__.pop("_decode_pair", None)
        if pair is not None:
            for catalog in pair.catalogs:
                catalog.clear()
            for packets in pair.packets:
                packets.clear()
        wrapper.concrete_aclgraph_entries.clear()
    for name in ("_graph_params", "_draft_graph_params", "_draft_graph_prefill_params"):
        params = getattr(acl_graph, name, None)
        if params is not None:
            for field in dataclasses.fields(params):
                entries = getattr(params, field.name)
                for key in entries:
                    entries[key] = None if field.name == "workspaces" else []
            # Native attention-backend reinitialization calls set_*_graph_params.
            # Retire the singleton too, not just its tensor-bearing contents.
            setattr(acl_graph, name, None)


class PhysicalMemoryMixin:
    def _init_device(self):
        if self.cache_config.kv_cache_memory_bytes is not None:
            return super()._init_device()
        # The pinned native init uses this fraction ONLY to reject startup if
        # free < total*fraction. Disable that provisional rejection locally;
        # the actual free snapshot, model loading and positive physical-budget
        # check decide feasibility. Restore the user's config even on failure.
        # This is not a public zero-utilization setting or a zero-byte budget.
        fraction = self.cache_config.gpu_memory_utilization
        self.cache_config.gpu_memory_utilization = 0.0
        try:
            device = super()._init_device()
        finally:
            self.cache_config.gpu_memory_utilization = fraction
        self.requested_memory = self.init_snapshot.free_memory
        return device

    def capture_trial_program(self):
        from ._draft import capture_trial

        return capture_trial(self)

    def retire_trial_program(self):
        from ._draft import retire_trial

        retire_trial(self)

    def prepare_final_program(self):
        if not self._native_dp:
            from ._draft import prepare_final

            prepare_final(self)

    def snapshot(self, phase, **extra):
        import torch

        torch.npu.synchronize()
        free, total = torch.npu.mem_get_info()
        row = dict(
            phase=phase,
            rank=self.rank,
            free=free,
            total=total,
            allocated=torch.npu.memory_allocated(),
            reserved=torch.npu.memory_reserved(),
            **extra,
        )
        if phase == "physical_budget_after_trial_release":
            log.info(
                "BetterScale physical KV rank=%s: budget=%.3f GiB, "
                "complete graph=%.3f GiB, safety=%.3f GiB",
                self.rank,
                extra["kv_budget"] / GiB,
                extra["measured_target_graph"] / GiB,
                extra["safety"] / GiB,
            )
        elif phase == "ready_after_capture":
            log.info(
                "BetterScale ready rank=%s: free=%.3f GiB, reserved=%.3f GiB, "
                "context ceiling=%s tokens, active seats=%s",
                self.rank,
                free / GiB,
                row["reserved"] / GiB,
                self.model_config.max_model_len,
                self.vllm_config.scheduler_config.max_num_seqs,
            )
        else:
            log.debug("Physical KV accounting: %s", row)
        return row

    def determine_available_memory(self):
        import torch
        from vllm.config import set_current_vllm_config
        from vllm.v1.core.kv_cache_utils import (
            get_kv_cache_groups,
            get_kv_cache_config_from_groups,
        )
        from vllm.compilation.counter import compilation_counter
        from vllm_ascend.compilation import acl_graph

        if not self._native_dp:
            from ._communication import prepare

            prepare(self)
        if self.cache_config.kv_cache_memory_bytes is not None:
            return super().determine_available_memory()
        # Budget only memory actually available to this worker at startup.
        self.requested_memory = self.init_snapshot.free_memory
        super().determine_available_memory()
        runner = self.model_runner
        specs = runner.get_kv_cache_spec()
        groups = get_kv_cache_groups(self.vllm_config, specs)
        previous_override = self.cache_config.num_gpu_blocks_override
        self.cache_config.num_gpu_blocks_override = max(
            256, self.vllm_config.compilation_config.max_cudagraph_capture_size
        )
        try:
            trial = get_kv_cache_config_from_groups(self.vllm_config, groups, 0)
        finally:
            self.cache_config.num_gpu_blocks_override = previous_override
        torch.npu.synchronize()
        torch.npu.empty_cache()
        before_trial = torch.npu.memory_allocated()
        wrappers = list(acl_graph._acl_graph_wrappers)
        assert all(
            not w.concrete_aclgraph_entries and "_decode_pair" not in w.__dict__
            for w in wrappers
        )
        original_pools = [w.graph_pool for w in wrappers]
        # Reuse the final pool: retaining graph handles must not reserve a
        # separate full-model activation arena. No trial graph will replay.
        assert original_pools and all(p == original_pools[0] for p in original_pools)
        trial_pool = original_pools[0]
        self._preflight_retired_graphs = []
        captured_before = compilation_counter.num_cudagraph_captured
        try:
            with set_current_vllm_config(self.vllm_config):
                runner.initialize_kv_cache(trial)
            self.cache_config.num_gpu_blocks = trial.num_blocks
            for wrapper in wrappers:
                wrapper.graph_pool = trial_pool
            self.snapshot(
                "trial_before_capture",
                trial_kv_bytes=sum(t.size for t in trial.kv_cache_tensors),
            )
            graph_bytes = self.capture_trial_program()
            assert graph_bytes > 0
            self.snapshot("trial_after_capture", trial_graph_bytes=graph_bytes)
        finally:
            self.retire_trial_program()
            clear_trial_graphs(wrappers, self._preflight_retired_graphs)
            for wrapper, pool in zip(wrappers, original_pools):
                wrapper.graph_pool = pool
            runner._cleanup_profiling_kv_cache()
            compilation_counter.num_cudagraph_captured = captured_before
            del trial_pool
            gc.collect()
            torch.npu.empty_cache()
            torch.npu.synchronize()
        persistent_growth = max(0, torch.npu.memory_allocated() - before_trial)
        # Fail rather than hiding a retained trial cache behind another budget.
        if persistent_growth >= 256 * (1 << 20):
            raise RuntimeError(
                f"Trial KV retirement retained {persistent_growth} bytes"
            )
        safety = GiB
        budget = int(
            self.init_snapshot.free_memory
            - runner.model_memory_usage
            - self.peak_activation_memory
            - self.non_torch_memory
            - graph_bytes
            - persistent_growth
            - safety
        )
        if budget <= 0:
            raise RuntimeError(f"No physical memory remains for KV: {budget} bytes")
        self.available_kv_cache_memory_bytes = budget
        self.snapshot(
            "physical_budget_after_trial_release",
            kv_budget=budget,
            measured_target_graph=graph_bytes,
            persistent_growth=persistent_growth,
            safety=safety,
        )
        return budget
