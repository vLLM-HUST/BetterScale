import argparse
import ctypes
import json
from pathlib import Path
import torch
import torch_npu

H, I = 5120, 13824


def metric(a, b):
    delta = (a.float() - b.float()).abs()
    return dict(
        max_abs=delta.max().item(),
        rms=delta.square().mean().sqrt().item(),
        ref_rms=b.float().square().mean().sqrt().item(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    lib = ctypes.CDLL(args.library)
    launch = lib.launch_native_ffn
    launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint32] * 3 + [ctypes.c_void_p]
    launch.restype = ctypes.c_int
    torch.manual_seed(1901)
    w = (torch.randn(2 * I, H, device="npu") / H**0.5).bfloat16()
    wnz = torch_npu.npu_format_cast(w.t().contiguous(), 29)
    assert torch_npu.get_npu_format(wnz) == 29
    reports = []
    for m in [128, 257, 769]:
        x = torch.randn(m, H, device="npu", dtype=torch.bfloat16)
        ybuf = torch.full((m * I + 256,), 19, device="npu", dtype=torch.bfloat16)
        y = ybuf[128:-128].view(m, I)
        slab = 256
        sbuf = torch.full(
            (2 * slab * 2 * I + 256,), 19, device="npu", dtype=torch.bfloat16
        )
        scratch = sbuf[128:-128]
        x0 = x.cpu()
        w0 = wnz.cpu()
        ref = torch_npu.npu_swiglu(torch.mm(x, w.t()))
        for vc in [256, 512]:

            def candidate():
                err = launch(
                    x.data_ptr(),
                    wnz.data_ptr(),
                    y.data_ptr(),
                    scratch.data_ptr(),
                    m,
                    slab,
                    vc,
                    torch.npu.current_stream().npu_stream,
                )
                assert err == 0, err
                return y

            candidate()
            torch.npu.synchronize()
            e = metric(y, ref)
            print(json.dumps(dict(rows=m, vc=vc, eager=e)), flush=True)
            assert e["max_abs"] <= 0.0625 and e["rms"] <= 0.005, e
            assert torch.equal(x.cpu(), x0) and torch.equal(wnz.cpu(), w0)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                candidate()
            for step in range(3):
                x.mul_(0.875)
                oracle = torch_npu.npu_swiglu(torch.mm(x, w.t()))
                graph.replay()
                torch.npu.synchronize()
                ge = metric(y, oracle)
                assert ge["max_abs"] <= 0.0625 and ge["rms"] <= 0.005, ge
            assert torch.all(ybuf[:128] == 19) and torch.all(ybuf[-128:] == 19)
            assert torch.all(sbuf[:128] == 19) and torch.all(sbuf[-128:] == 19)
            reports.append(dict(rows=m, vc=vc, eager=e, graph=ge))
            (args.output / "results.json").write_text(json.dumps(reports, indent=2))
            del graph
            x.copy_(x0)
        del x, y, ybuf, scratch, sbuf, w0, ref
        torch.npu.empty_cache()
    (args.output / "complete.json").write_text(json.dumps(dict(status="PASS")))


if __name__ == "__main__":
    main()
