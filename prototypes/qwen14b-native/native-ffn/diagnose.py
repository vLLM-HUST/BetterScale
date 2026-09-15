"""Isolate native Cube pipeline from Vector arithmetic; not a pure Cube timer."""

import argparse
import ctypes
import json
import statistics
from pathlib import Path
import torch
import torch_npu
from bench import capture, elapsed
from probe import H, I, metric

p = argparse.ArgumentParser()
p.add_argument("--base", required=True)
p.add_argument("--m256", required=True)
p.add_argument("--old-base")
p.add_argument("--old-m256")
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
torch.npu.set_device(0)
torch.npu.config.allow_internal_format = True
torch.manual_seed(1903)
x = torch.randn(4096, H, device="npu", dtype=torch.bfloat16)
w = (torch.randn(2 * I, H, device="npu") / H**0.5).bfloat16()
b = torch_npu.npu_format_cast(w.t().contiguous(), 29)
reports = []
variants = [("base", a.base), ("m256", a.m256)]
if a.old_base and a.old_m256:
    variants = [("k128-base", a.old_base), ("k128-m256", a.old_m256)] + variants
for name, path in variants:
    lib = ctypes.CDLL(path)
    launch = lib.launch_native_ffn
    launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint32] * 3 + [ctypes.c_void_p]
    launch.restype = ctypes.c_int
    for slab in [256, 1024]:
        y = torch.empty(4096, I, device="npu", dtype=torch.bfloat16)
        guard = torch.full(
            (2 * slab * 2 * I + 256,), 19, device="npu", dtype=torch.bfloat16
        )
        scratch = guard[128:-128]

        def run(vc):
            assert (
                launch(
                    x.data_ptr(),
                    b.data_ptr(),
                    y.data_ptr(),
                    scratch.data_ptr(),
                    4096,
                    slab,
                    vc,
                    torch.npu.current_stream().npu_stream,
                )
                == 0
            )
            return y

        gn, on, _ = capture(lambda: torch.mm(x, b))
        gf, of, _ = capture(lambda: run(512))
        err = metric(of, torch_npu.npu_swiglu(on))
        assert err["max_abs"] <= 0.0625 and err["rms"] <= 0.005, err
        gc, _, _ = capture(lambda: run(0))
        last = scratch.view(2, slab, 2 * I)[1]
        ce = metric(last, on[-slab:])
        assert ce["max_abs"] <= 0.0625 and ce["rms"] <= 0.005, ce
        x.mul_(0.875)
        gn.replay()
        gf.replay()
        torch.npu.synchronize()
        err = metric(of, torch_npu.npu_swiglu(on))
        assert err["max_abs"] <= 0.0625 and err["rms"] <= 0.005, err
        samples = [[], [], []]
        for trial in range(6):
            for j in ([0, 1, 2] if trial % 2 == 0 else [2, 1, 0]):
                samples[j].append(elapsed([gn, gf, gc][j]))
        assert torch.all(guard[:128] == 19) and torch.all(guard[-128:] == 19)
        row = dict(
            variant=name,
            slab=slab,
            errors=err,
            cube_errors=ce,
            native_nz_gemm_ms=statistics.median(samples[0]),
            fused_gateup_swiglu_ms=statistics.median(samples[1]),
            cube_protocol_ms=statistics.median(samples[2]),
            samples_ms=samples,
        )
        reports.append(row)
        (a.output / "results.json").write_text(json.dumps(reports, indent=2))
        print(json.dumps(row), flush=True)
        del gn, gf, gc, on, of, last, scratch, guard, y
(a.output / "complete.json").write_text('{"status":"PASS"}')
