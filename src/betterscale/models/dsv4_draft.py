"""Own the finite TP draft catalog through trial/final State replacement."""

import torch

from ..patches.auto_kv import snapshot


def prepare(self):
    from ..patches.split_draft._warmup import prepare as prepare_draft

    self._preparing_draft_graphs = True
    try:
        prepare_draft(self, snapshot=snapshot)
    finally:
        self._preparing_draft_graphs = False


def capture_trial(self):
    target_bytes = self.model_runner.capture_model()
    if self._native_dp:
        return target_bytes
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
    from ..patches.split_draft import install

    install(self)
    torch.npu.synchronize()
    before = torch.npu.mem_get_info()[0]
    prepare(self)
    torch.npu.synchronize()
    extra = max(0, before - torch.npu.mem_get_info()[0])
    snapshot(
        self,
        "trial_complete_program",
        target_graph_bytes=target_bytes,
        draft_extra_bytes=extra,
    )
    return target_bytes + extra


def retire_trial(self):
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


def prepare_final(self):
    prepare(self)
    assert self.model_runner.input_batch.num_reqs == 0
    # Preparation writes scratch KV before admission. Retire those contents,
    # not their addresses; graph banks keep binding the same final State.
    from ..patches.auto_kv._state import zero_kv_backings

    # The admitted native allocator owns these entire backing allocations.
    # Typed/page-strided aliases share them; clear each backing once, including
    # padding, without materializing a dense copy of every logical view.
    count = zero_kv_backings(self.compilation_config.static_forward_context)
    torch.npu.synchronize()
    snapshot(self, "complete_program_before_admission", state_backings_cleared=count)
