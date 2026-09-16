"""Native local expert FULL-graph control, at actual Qwen layer inputs."""

import json
import os
from pathlib import Path
import statistics
import torch
import torch_npu
from vllm.forward_context import get_forward_context
from vllm_ascend.worker.worker import NPUWorker


class LocalExpertWorker(NPUWorker):
    def arm(self):
        self.measurements = []
        self.seen = set()
        for index, layer in enumerate(self.model_runner.model.model.layers):
            original = layer.mlp.forward

            def forward(hidden, original=original, index=index):
                context = get_forward_context()
                entry = context.moe_layer_index
                reference = original(hidden)
                exit_index = context.moe_layer_index
                key = index, hidden.shape[0]
                if key in self.seen:
                    return reference
                self.seen.add(key)
                expected = reference.clone()
                x = hidden.clone()

                def body():
                    context.moe_layer_index = entry
                    return original(x)

                try:
                    for _ in range(3):
                        body()
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        output = body()
                    for _ in range(3):
                        graph.replay()
                    torch.npu.synchronize()
                    torch.testing.assert_close(output, expected, rtol=0.02, atol=1e-6)
                    exact = torch.equal(output, expected)
                    # External timing events: captured timing events are unsupported.
                    times = []
                    for _ in range(5):
                        start = torch.npu.Event(enable_timing=True)
                        end = torch.npu.Event(enable_timing=True)
                        start.record()
                        for _ in range(64):
                            graph.replay()
                        end.record()
                        end.synchronize()
                        times.append(start.elapsed_time(end) * 1000 / 64)
                    self.measurements.append(
                        dict(
                            layer=index,
                            rows=key[1],
                            exact=exact,
                            replay_us=times,
                            median_us=statistics.median(times),
                            scope="64 prequeued local MLP FULL replays, external event timing",
                        )
                    )
                    Path(os.environ["LOCAL_EXPERT_RESULT"]).write_text(
                        json.dumps(self.measurements, indent=2)
                    )
                    graph.reset()
                finally:
                    context.moe_layer_index = exit_index
                return reference

            layer.mlp.forward = forward
