"""Experimental TP8 physical fit with complete, single-pool draft catalog.

Requires the frozen experimental split_draft implementation: shared platform
pool and actual eager priming before every startup draft capture. Not a public
Worker, nor permission to silently capture new shapes on live requests.
"""

import torch
from betterscale.patches import split_draft
from communication_prewarm import prepare as prepare_communication
from draft_warmup import prepare as prepare_draft
from preflight_worker import PreflightWorker


class TPPhysicalWorker(PreflightWorker):
    def determine_available_memory(self):
        prepare_communication(self)
        return super().determine_available_memory()

    def _prepare_draft(self):
        self._preparing_draft_graphs = True
        try:
            prepare_draft(self)
        finally:
            self._preparing_draft_graphs = False

    def capture_trial_program(self):
        target_bytes = super().capture_trial_program()
        self._trial_draft_bindings = {}
        d = self.model_runner.drafter
        for name in (
            "_per_group_block_tables",
            "_per_group_slot_mappings",
            "_per_group_block_table_buffers",
            "_context_slot_mapping_buffers",
        ):
            value = getattr(d, name, None)
            self._trial_draft_bindings[name] = (
                hasattr(d, name),
                dict(value) if isinstance(value, dict) else value,
            )
        split_draft.install(self)
        torch.npu.synchronize()
        before = torch.npu.mem_get_info()[0]
        self._prepare_draft()
        torch.npu.synchronize()
        extra = max(0, before - torch.npu.mem_get_info()[0])
        self.snapshot(
            "trial_complete_program",
            target_graph_bytes=target_bytes,
            draft_extra_bytes=extra,
        )
        return target_bytes + extra

    def retire_trial_program(self):
        manager = getattr(self, "_exact_draft_graph", None)
        if manager is None:
            return
        torch.npu.synchronize()
        for catalog in (manager.decode_graphs, manager.query_graphs):
            for entry in catalog.values():
                if entry.graph is not None:
                    self._preflight_retired_graphs.append(entry.graph)
            catalog.clear()
        d = self.model_runner.drafter
        d._runnable = manager.original
        del self._exact_draft_graph
        for name, (present, value) in self._trial_draft_bindings.items():
            if present:
                setattr(d, name, value)
            elif hasattr(d, name):
                delattr(d, name)
        del self._trial_draft_bindings

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        self._prepare_draft()
        assert self.model_runner.input_batch.num_reqs == 0
        # Preparation writes scratch KV before admission. Retire those contents,
        # not their addresses; graph banks keep binding the same final State.
        seen = set()

        def clear(value):
            if isinstance(value, torch.Tensor) and value.numel():
                key = (value.data_ptr(), value.numel(), value.dtype)
                if key not in seen:
                    seen.add(key)
                    value.zero_()
            elif isinstance(value, (list, tuple)):
                for item in value:
                    clear(item)

        for layer in self.compilation_config.static_forward_context.values():
            if hasattr(layer, "kv_cache"):
                clear(layer.kv_cache)
        torch.npu.synchronize()
        self._execution_baseline = self.snapshot(
            "complete_program_before_admission", state_views_cleared=len(seen)
        )
        torch.npu.reset_peak_memory_stats()
        return result

    def capacity_snapshot(self, phase="after_cohort"):
        manager = self._exact_draft_graph
        record = self.snapshot(
            phase,
            draft_catalog={
                kind: [
                    dict(key=str(key), replays=entry.replays, fallbacks=entry.fallbacks)
                    for key, entry in catalog.items()
                ]
                for kind, catalog in (
                    ("decode", manager.decode_graphs),
                    ("query", manager.query_graphs),
                )
            },
        )
        record["baseline"] = self._execution_baseline
        return record
