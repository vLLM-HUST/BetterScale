"""Opt-in one-wave stage timings; diagnostic receipts are never throughput gates."""

import time
import torch


class LayerTimers:
    def __init__(self, root):
        self.handles = []
        self.events = []
        for i, layer in enumerate(root.model.language_model.layers):
            self.add(layer, f"layer{i}")
            self.add(layer.mlp, f"moe{i}")
            self.add(layer.mlp.shared_expert, f"shared{i}")

    def add(self, module, name):
        def before(module, args):
            event = torch.npu.Event(enable_timing=True)
            event.record()
            self.events.append(dict(name=name, start=event, cpu_start=time.monotonic()))

        def after(module, args, output):
            event = torch.npu.Event(enable_timing=True)
            event.record()
            record = next(
                r for r in reversed(self.events) if r["name"] == name and "end" not in r
            )
            record.update(end=event, cpu_end=time.monotonic())

        self.handles.extend(
            (
                module.register_forward_pre_hook(before),
                module.register_forward_hook(after),
            )
        )

    def result(self):
        torch.npu.synchronize()
        return [
            dict(
                name=r["name"],
                device_ms=r["start"].elapsed_time(r["end"]),
                cpu_ms=1000 * (r["cpu_end"] - r["cpu_start"]),
            )
            for r in self.events
        ]

    def close(self):
        for handle in self.handles:
            handle.remove()
