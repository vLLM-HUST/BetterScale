"""Native HC-pre workspace and residual-alias leaf gates, not a model gate."""

import argparse
import gc
import json
import os
from pathlib import Path

import torch
import torch_npu
from vllm_ascend.utils import enable_custom_op

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--reference", type=Path)
p.add_argument("--vendor", type=Path)
p.add_argument("--rows", type=int, nargs="+", default=[1, 24, 255, 256, 257, 516, 4128])
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
torch_npu.npu.config.allow_internal_format = True
enable_custom_op()
if a.vendor:
    # Selected-op diagnostic only: use ONE effective OPP vendor. The native
    # extension retains its original ABI-compatible API library, but this
    # process exercises only HcPre, not HC-post or a complete model.
    # Workspace reduction must prove selection, not just library loading.
    assert a.vendor.is_dir()
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(a.vendor)
torch.npu.set_device(0)


def pre(x, w, scale, bias):
    return torch.ops._C_ascend.npu_hc_pre_v2(x, w, scale, bias, 4, 20, 1e-6, 1e-6)


def post(y, residual, p, c):
    return torch.ops._C_ascend.npu_hc_post(
        y.unsqueeze(0), residual.unsqueeze(0), p.unsqueeze(0), c.unsqueeze(0)
    ).squeeze(0)


@torch.inference_mode()
def check(rows):
    # Generate on CPU so separate native-library processes receive identical bits.
    torch.manual_seed(1024 + rows)
    cpu = [
        torch.randn(rows, 4, 4096, dtype=torch.bfloat16),
        torch.randn(24, 4 * 4096) * 0.01,
        torch.randn(3) * 0.01,
        torch.randn(24) * 0.01,
    ]
    x, w, s, b = [t.to("npu") for t in cpu]
    warm = pre(x, w, s, b)
    torch.npu.synchronize()
    del warm
    gc.collect()
    torch.npu.empty_cache()
    base = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    outputs = pre(x, w, s, b)
    torch.npu.synchronize()
    extra = torch.npu.max_memory_allocated() - base
    expected = [t.cpu() for t in outputs]
    assert torch.equal(x.cpu(), cpu[0]), "HC-pre modified residual input"
    if a.reference:
        ref = torch.load(a.reference / f"{rows}.pt", weights_only=True)
        for got, old in zip(expected, ref, strict=True):
            assert torch.equal(got, old), f"workspace-only change differs at {rows}"
    torch.save(expected, a.output / f"{rows}.pt")
    del outputs

    if a.vendor:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = pre(x, w, s, b)
        for value in (0.5, -0.75):
            x.fill_(value)
            reference = tuple(t.cpu() for t in pre(x, w, s, b))
            graph.replay()
            torch.npu.synchronize()
            for got, old in zip(result, reference, strict=True):
                assert torch.equal(got.cpu(), old), "HcPre FULL differs"
            assert torch.all(x == value).item()
        return dict(
            rows=rows,
            eager_extra_bytes=extra,
            baseline_exact=bool(a.reference),
            changing_input_full_exact=True,
        )

    # Same native HC-pre/post numerics, two consecutive residual lifetimes.
    # Identity inner transforms intentionally isolate the alias contract; this
    # does not replace the real attention/MLP forward integration gate.
    def block(clone):
        residual = x.clone() if clone else x
        y, p, c = pre(x, w, s, b)
        h = post(y, residual, p, c)
        residual = h.clone() if clone else h
        y, p, c = pre(h, w, s, b)
        return post(y, residual, p, c), residual

    reference = tuple(t.cpu() for t in block(True))
    result = block(False)
    for got, old in zip(result, reference, strict=True):
        assert torch.equal(got.cpu(), old), "eager alias differs"
    del result
    graph_memory = {}
    for clone in (True, False):
        gc.collect()
        torch.npu.empty_cache()
        allocated = torch.npu.memory_allocated()
        reserved = torch.npu.memory_reserved()
        torch.npu.reset_peak_memory_stats()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            result = block(clone)
        torch.npu.synchronize()
        graph_memory["clone" if clone else "alias"] = dict(
            allocated_extra=torch.npu.max_memory_allocated() - allocated,
            reserved_extra=torch.npu.max_memory_reserved() - reserved,
        )
        # Graph inputs change after capture; both HC lifetimes are replayed.
        for value in (0.5, -0.75):
            x.fill_(value)
            reference = tuple(t.cpu() for t in block(True))
            graph.replay()
            torch.npu.synchronize()
            for got, old in zip(result, reference, strict=True):
                assert torch.equal(got.cpu(), old), "FULL alias differs"
            assert torch.all(x == value).item(), "HC-post modified residual input"
        del graph, result, got
    return dict(
        rows=rows,
        eager_extra_bytes=extra,
        baseline_exact=bool(a.reference),
        hc_pair_alias_exact=True,
        changing_input_full_exact=True,
        graph_memory=graph_memory,
    )


receipts = []
for rows in a.rows:
    receipts.append(check(rows))
    gc.collect()
    torch.npu.empty_cache()
    (a.output / "result.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(json.dumps(receipts[-1]), flush=True)
# Local artifact only; preserve dynamic loader evidence without dumping environment.
libs = sorted(
    {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if ("optiling" in line or "opapi" in line) and "/" in line
    }
)
(a.output / "loaded-libraries.json").write_text(json.dumps(libs, indent=2) + "\n")
