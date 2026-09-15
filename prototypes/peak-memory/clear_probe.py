"""One-card strided-view versus dense-backing clear: no model/weights."""

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch_npu
from betterscale.patches.auto_kv._state import zero_kv_backings

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
torch.npu.set_device(0)
rows = []
for mode in ("strided", "backing"):
    raw = torch.ones(2**30, dtype=torch.uint8, device="npu")
    view = raw.as_strided((8192, 65536), (131072, 1))
    context = {"layer": SimpleNamespace(kv_cache=[view, view.view(torch.int32)])}

    def clear():
        if mode == "strided":
            for v in context["layer"].kv_cache:
                v.zero_()
        else:
            zero_kv_backings(context)

    clear()
    torch.npu.synchronize()
    raw.fill_(13)
    torch.npu.synchronize()
    torch.npu.empty_cache()
    base = torch.npu.memory_allocated()
    reserved = torch.npu.memory_reserved()
    torch.npu.reset_peak_memory_stats()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    clear()
    end.record()
    end.synchronize()
    peak = torch.npu.max_memory_allocated()
    peak_reserved = torch.npu.max_memory_reserved()
    assert torch.count_nonzero(view).item() == 0
    assert view.data_ptr() == raw.data_ptr()
    # A graph captured against these addresses sees clearing and later updates.
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = view[:2, :8].clone()
    for val in (7, 21):
        raw.fill_(val)
        graph.replay()
        torch.npu.synchronize()
        assert torch.all(out == val).item()
        clear()
        graph.replay()
        torch.npu.synchronize()
        assert torch.count_nonzero(out).item() == 0
    rows.append(
        dict(
            mode=mode,
            base=base,
            base_reserved=reserved,
            peak_extra=peak - base,
            peak_reserved_extra=peak_reserved - reserved,
            clear_ms=start.elapsed_time(end),
            logical_bytes=view.numel(),
            backing_bytes=raw.numel(),
            graph_updates_exact=True,
        )
    )
    del graph, out, view, raw, context, start, end
    gc.collect()
    torch.npu.empty_cache()
a.output.write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows))
