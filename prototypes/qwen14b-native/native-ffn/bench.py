"""Crossed FULL-graph leaf comparison; no model/serving claim."""

import argparse
import ctypes
import gc
import json
from pathlib import Path
import statistics
import torch
import torch_npu
from probe import metric, H, I


def capture(fn):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    base = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = fn()
    graph.replay()
    torch.npu.synchronize()
    return graph, out, torch.npu.max_memory_allocated() - base


def elapsed(g):
    a, b = [torch.npu.Event(enable_timing=True) for _ in range(2)]
    a.record()
    for _ in range(10):
        g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument(
        "--full-buffer",
        action="store_true",
        help="allocate unique256-row paired tiles; required for FULL_BUFFER binary",
    )
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    lib = ctypes.CDLL(args.library)
    launch = lib.launch_native_ffn
    launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint32] * 3 + [ctypes.c_void_p]
    launch.restype = ctypes.c_int
    torch.manual_seed(1902)
    w = (torch.randn(2 * I, H, device="npu") / H**0.5).bfloat16()
    wnz = torch_npu.npu_format_cast(w.t().contiguous(), 29)
    down = (torch.randn(H, I, device="npu") / I**0.5).bfloat16()
    reports = []
    for m in [512, 4096]:
        x = torch.randn(m, H, device="npu", dtype=torch.bfloat16)
        y = torch.empty(m, I, device="npu", dtype=torch.bfloat16)

        def native():
            return torch.mm(torch_npu.npu_swiglu(torch.mm(x, w.t())), down.t())

        def native_nz():
            return torch.mm(torch_npu.npu_swiglu(torch.mm(x, wnz)), down.t())

        ga, oa, ma = capture(native)
        gn, on, mn = capture(native_nz)
        ne = metric(on, oa)
        assert ne["max_abs"] <= 0.0625 and ne["rms"] <= 0.005, ne
        for slab in ([256] if args.full_buffer else [256, 1024, 2048]):
            sb = torch.full(
                (
                    (
                        (m + 255) // 256 * 256 * 2 * I
                        if args.full_buffer
                        else 2 * slab * 2 * I
                    )
                    + 256,
                ),
                19,
                device="npu",
                dtype=torch.bfloat16,
            )
            scratch = sb[128:-128]
            for vc in ([256] if args.full_buffer else [256, 512]):

                def candidate():
                    ret = launch(
                        x.data_ptr(),
                        wnz.data_ptr(),
                        y.data_ptr(),
                        scratch.data_ptr(),
                        m,
                        slab,
                        vc,
                        torch.npu.current_stream().npu_stream,
                    )
                    assert ret == 0, ret
                    return torch.mm(y, down.t())

                gb, ob, mb = capture(candidate)
                e = metric(ob, oa)
                assert e["max_abs"] <= 0.0625 and e["rms"] <= 0.005, e
                x.mul_(0.875)
                ga.replay()
                gn.replay()
                gb.replay()
                torch.npu.synchronize()
                e = metric(ob, oa)
                assert e["max_abs"] <= 0.0625 and e["rms"] <= 0.005, e
                samples = [[], [], []]
                graphs = [ga, gn, gb]
                for trial in range(6):
                    for idx in ([0, 1, 2] if trial % 2 == 0 else [2, 1, 0]):
                        samples[idx].append(elapsed(graphs[idx]))
                assert torch.all(sb[:128] == 19) and torch.all(sb[-128:] == 19)
                row = dict(
                    rows=m,
                    slab=slab,
                    vc=vc,
                    errors=e,
                    samples_ms=samples,
                    native_nd_ms=statistics.median(samples[0]),
                    native_nz_ms=statistics.median(samples[1]),
                    fused_ms=statistics.median(samples[2]),
                    native_capture_bytes=ma,
                    native_nz_capture_bytes=mn,
                    fused_capture_bytes=mb,
                    fused_external_bytes=scratch.numel() * 2 + y.numel() * 2,
                )
                reports.append(row)
                (args.output / "results.json").write_text(json.dumps(reports, indent=2))
                print(json.dumps(row), flush=True)
                del gb, ob
                gc.collect()
                torch.npu.empty_cache()
            del sb, scratch
        del ga, gn, oa, on, x, y
        gc.collect()
        torch.npu.empty_cache()
    (args.output / "complete.json").write_text('{"status":"PASS"}')


if __name__ == "__main__":
    main()
