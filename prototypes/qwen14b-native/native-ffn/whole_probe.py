"""Whole-FFN fusion: matched controls, guards and changed-input FULL replay."""

import argparse, ctypes, gc, json, statistics
from pathlib import Path
import torch, torch_npu
from bench import capture, elapsed
from probe import metric, H, I

p = argparse.ArgumentParser()
p.add_argument("--library", required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--performance-only", action="store_true")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
torch.npu.set_device(0)
torch.npu.config.allow_internal_format = True
torch.manual_seed(1905)
w = (torch.randn(2 * I, H, device="npu") / H**0.5).bfloat16()
wnz = torch_npu.npu_format_cast(w.t().contiguous(), 29)
d = (torch.randn(H, I, device="npu") / I**0.5).bfloat16()
dnz = torch_npu.npu_format_cast(d.t().contiguous(), 29)
lib = ctypes.CDLL(a.library)
whole = lib.launch_whole_ffn
whole.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_uint32] * 2 + [ctypes.c_void_p]
whole.restype = ctypes.c_int
leaf = lib.launch_native_ffn
leaf.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint32] * 3 + [ctypes.c_void_p]
leaf.restype = ctypes.c_int
reports = []
for m in ([512, 4096] if a.performance_only else [128, 257, 769, 1025, 512, 4096]):
    x = torch.randn(m, H, device="npu", dtype=torch.bfloat16)
    buffers = [
        torch.full((n + 256,), 19, device="npu", dtype=torch.bfloat16)
        for n in [m * I, 2 * 256 * 2 * I, m * H]
    ]
    z, scratch, y = [b[128:-128] for b in buffers]
    z = z.view(m, I)
    y = y.view(m, H)

    def native_nd():
        return torch.mm(torch_npu.npu_swiglu(torch.mm(x, w.t())), d.t())

    def native_nz():
        return torch.mm(torch_npu.npu_swiglu(torch.mm(x, wnz)), dnz)

    def current(use_nz=False):
        assert (
            leaf(
                x.data_ptr(),
                wnz.data_ptr(),
                z.data_ptr(),
                scratch.data_ptr(),
                m,
                256,
                256,
                torch.npu.current_stream().npu_stream,
            )
            == 0
        )
        return torch.mm(z, dnz if use_nz else d.t())

    # Current accepted path intentionally keeps original native ND down.
    ga, oa, _ = capture(native_nd)
    gb, ob, _ = capture(native_nz)
    gcg, oc, _ = capture(current)
    gd, od, _ = capture(lambda: current(True))
    for chunk in [512, 1024, 4096]:

        def candidate():
            assert (
                whole(
                    x.data_ptr(),
                    wnz.data_ptr(),
                    dnz.data_ptr(),
                    z.data_ptr(),
                    scratch.data_ptr(),
                    y.data_ptr(),
                    m,
                    chunk,
                    torch.npu.current_stream().npu_stream,
                )
                == 0
            )
            return y

        gg, og, _ = capture(candidate)
        errors = []
        for _ in range(3):
            x.mul_(0.875)
            for g in [ga, gb, gcg, gd, gg]:
                g.replay()
            torch.npu.synchronize()
            for out in [ob, oc, od, og]:
                e = metric(out, oa)
                assert e["max_abs"] <= 0.0625 and e["rms"] <= 0.005, e
            errors.append(e)
            assert all(
                bool(torch.all(b[:128] == 19)) and bool(torch.all(b[-128:] == 19))
                for b in buffers
            )
        row = dict(rows=m, chunk=chunk, errors=errors)
        if m in [512, 4096]:
            samples = [[], [], [], [], []]
            for trial in range(6):
                for j in ([0, 1, 2, 3, 4] if trial % 2 == 0 else [4, 3, 2, 1, 0]):
                    samples[j].append(elapsed([ga, gb, gcg, gd, gg][j]))
            row.update(
                samples_ms=samples,
                native_nd_ms=statistics.median(samples[0]),
                native_both_nz_ms=statistics.median(samples[1]),
                accepted_panel_ms=statistics.median(samples[2]),
                panel_nz_down_ms=statistics.median(samples[3]),
                whole_ms=statistics.median(samples[4]),
            )
        reports.append(row)
        (a.output / "results.json").write_text(json.dumps(reports, indent=2))
        print(json.dumps(row), flush=True)
        del gg, og
        gc.collect()
        torch.npu.empty_cache()
    del ga, gb, gcg, gd, oa, ob, oc, od, x, z, scratch, y, buffers
    gc.collect()
    torch.npu.empty_cache()
(a.output / "complete.json").write_text('{"status":"PASS"}')
