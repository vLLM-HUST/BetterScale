"""Bounded original-vs-split oracle inside native eager Qwen execution."""

import json
import os
from pathlib import Path
import torch
from vllm.forward_context import get_forward_context
from vllm_ascend.worker.worker import NPUWorker
from qwen_layer import split_forward


class AttentionProbeWorker(NPUWorker):
    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        self.armed = False
        self.checks = []
        self.graph_shapes = set()
        self.metadata_banks = {}
        self.lane_snapshots = {}
        self.lane_pair_done = False
        model = self.model_runner.model.model
        original = model.forward

        def forward(*args, **kwargs):
            ctx = get_forward_context()
            entry_moe_index = ctx.moe_layer_index
            expected = original(*args, **kwargs)
            if not self.armed:
                return expected
            expected = expected.clone()
            unique = {}

            def visit(obj):
                if isinstance(obj, torch.Tensor):
                    storage = obj.untyped_storage()
                    key = storage.data_ptr(), storage.nbytes()
                    if key not in unique:
                        view = torch.empty(
                            0, dtype=torch.uint8, device=obj.device
                        ).set_(storage, 0, (storage.nbytes(),), (1,))
                        unique[key] = view, view.clone()
                elif isinstance(obj, dict):
                    for v in obj.values():
                        visit(v)
                elif isinstance(obj, (list, tuple)):
                    for v in obj:
                        visit(v)

            visit(self.model_runner.kv_caches)
            assert unique
            exit_moe_index = ctx.moe_layer_index
            ctx.moe_layer_index = entry_moe_index
            try:
                got = split_forward(model, *args, **kwargs)
                assert ctx.moe_layer_index == exit_moe_index
            finally:
                ctx.moe_layer_index = exit_moe_index
            exact = torch.equal(expected, got)
            error = (expected - got).abs().max().item()
            assert exact, f"output mismatch: {error}"
            assert all(torch.equal(v, ref) for v, ref in unique.values()), "KV differs"
            metadata_reused = 0
            if os.environ.get("ATTENTION_METADATA_PROBE") == "1":
                from metadata_graph import metadata_forward

                ctx.moe_layer_index = entry_moe_index
                try:
                    actual, metadata_reused = metadata_forward(
                        self.model_runner, self.metadata_banks, model, *args, **kwargs
                    )
                    assert torch.equal(
                        expected, actual
                    ), f"metadata graph output differs {(expected-actual).abs().max().item()}"
                    assert all(
                        torch.equal(v, ref) for v, ref in unique.values()
                    ), "metadata graph KV differs"
                finally:
                    torch.npu.synchronize()
                    ctx.moe_layer_index = exit_moe_index
            if (
                os.environ.get("ATTENTION_LANES_PROBE") == "1"
                and not self.lane_pair_done
            ):
                from independent_lanes import snapshot, run_pair

                identity = {16: "prefill", 1: "decode"}.get(got.shape[0])
                if identity and identity not in self.lane_snapshots:
                    ctx.moe_layer_index = entry_moe_index
                    try:
                        self.lane_snapshots[identity] = snapshot(
                            self.model_runner, model, expected, args, kwargs
                        )
                    finally:
                        ctx.moe_layer_index = exit_moe_index
                if len(self.lane_snapshots) == 2:
                    try:
                        pair = run_pair(model, self.lane_snapshots)
                        Path(
                            os.environ["ATTENTION_PROBE_OUT"] + ".lanes.json"
                        ).write_text(json.dumps(pair, indent=2))
                        self.lane_pair_done = True
                    finally:
                        ctx.moe_layer_index = exit_moe_index
            graph_replays = 0
            if (
                os.environ.get("ATTENTION_GRAPH_PROBE") == "1"
                and got.shape[0] not in self.graph_shapes
            ):
                from graph_probe import capture_segments, replay_segments

                self.graph_shapes.add(got.shape[0])
                ctx.moe_layer_index = entry_moe_index
                try:
                    initial, segments = capture_segments(model, *args, **kwargs)
                    for _ in range(3):
                        ctx.moe_layer_index = entry_moe_index
                        actual = replay_segments(model, initial, segments)
                        assert torch.equal(expected, actual), "graph output differs"
                        assert all(
                            torch.equal(v, ref) for v, ref in unique.values()
                        ), "graph KV differs"
                        graph_replays += 1
                finally:
                    torch.npu.synchronize()
                    ctx.moe_layer_index = exit_moe_index
            self.checks.append(
                dict(
                    rows=got.shape[0],
                    graph_replays=graph_replays,
                    metadata_reused=metadata_reused,
                    output_exact=exact,
                    kv_exact=True,
                    max_abs_error=error,
                )
            )
            Path(os.environ["ATTENTION_PROBE_OUT"]).write_text(
                json.dumps(self.checks, indent=2)
            )
            return got

        model.forward = forward
        return result

    def arm(self):
        self.armed = True
