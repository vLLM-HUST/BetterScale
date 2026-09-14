"""Single-device ACL memcpy capture and GEMM contention; no serving changes.

Run only under selected-device admission. Native memcpy is used rather than a
Tensor.copy_ kernel; H2D/D2H host storage is allocated by aclrtMallocHost.
"""

import argparse
import ctypes as C
import json
from pathlib import Path

import torch
import torch_npu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = True
    acl = C.CDLL("libascendcl.so")
    signatures = {
        "aclrtMallocHost": [C.POINTER(C.c_void_p), C.c_size_t],
        "aclrtFreeHost": [C.c_void_p],
        "aclrtMemcpyAsync": [
            C.c_void_p,
            C.c_size_t,
            C.c_void_p,
            C.c_size_t,
            C.c_int,
            C.c_void_p,
        ],
        "aclmdlRICaptureTaskGrpBegin": [C.c_void_p],
        "aclmdlRICaptureTaskGrpEnd": [C.c_void_p, C.POINTER(C.c_void_p)],
        "aclmdlRICaptureTaskUpdateBegin": [C.c_void_p, C.c_void_p],
        "aclmdlRICaptureTaskUpdateEnd": [C.c_void_p],
    }
    for name, types in signatures.items():
        f = getattr(acl, name)
        f.argtypes, f.restype = types, C.c_int

    def call(name, *xs):
        rc = getattr(acl, name)(*xs)
        if rc:
            raise RuntimeError(f"{name} returned {rc}")

    compute, dma = torch.npu.Stream(), torch.npu.Stream()
    hosts = []
    graphs = []

    def host(size, value):
        p = C.c_void_p()
        call("aclrtMallocHost", C.byref(p), size)
        hosts.append(p)
        C.memset(p, value, size)
        return p.value

    def copy(dst, src, size, kind):
        call("aclrtMemcpyAsync", dst, size, src, size, kind, dma.npu_stream)

    def capture(fn, stream, repetitions=1):
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g, stream=stream):
            for _ in range(repetitions):
                fn()
        graphs.append(g)
        return g

    def replay(g, stream):
        with torch.npu.stream(stream):
            g.replay()
        stream.synchronize()

    records = []
    try:
        # Address/content oracle: changing host bytes is observable without a
        # new capture. No writes to DMA source until the previous replay drains.
        size = 4096
        h1, h2, hout = host(size, 17), host(size, 43), host(size, 0)
        d1 = torch.zeros(size, dtype=torch.uint8, device="npu")
        d2 = torch.zeros_like(d1)
        torch.npu.synchronize()
        group = C.c_void_p()

        def body():
            if args.update:
                call("aclmdlRICaptureTaskGrpBegin", dma.npu_stream)
            copy(d1.data_ptr(), h1, size, 1)
            if args.update:
                call("aclmdlRICaptureTaskGrpEnd", dma.npu_stream, C.byref(group))
            copy(d2.data_ptr(), d1.data_ptr(), size, 3)
            copy(hout, d2.data_ptr(), size, 2)

        g = capture(body, dma)
        for value in (17, 29, 61):
            C.memset(h1, value, size)
            replay(g, dma)
            assert C.string_at(hout, size) == bytes([value]) * size
        record = {"kind": "fixed_address_changing_content", "pass": True}
        records.append(record)
        print(json.dumps(record), flush=True)
        if args.update:
            # Quiescent update, not a claim of safe update during replay.
            call("aclmdlRICaptureTaskUpdateBegin", dma.npu_stream, group)
            copy(d1.data_ptr(), h2, size // 2, 1)
            call("aclmdlRICaptureTaskUpdateEnd", dma.npu_stream)
            dma.synchronize()
            replay(g, dma)
            assert C.string_at(hout, size) == bytes([43]) * (size // 2) + bytes(
                [61]
            ) * (size // 2)
            records.append({"kind": "update_source_and_length", "pass": True})
            args.out.joinpath("update.json").write_text(json.dumps(records, indent=2))
            return

        # Out-of-graph events avoid this runtime's in-graph elapsed_time limit.
        repeats = 16
        for label, k, n in [("bf16", 1024, 4096), ("int8_qb", 1024, 32768)]:
            for m in (512, 4096):
                a = torch.randint(-2, 3, (m, k), dtype=torch.int8, device="npu")
                w = torch.randint(-2, 3, (n, k), dtype=torch.int8, device="npu")
                if label == "bf16":
                    a, w = a.bfloat16(), w.bfloat16().t()

                    def mm():
                        return torch.mm(a, w)

                else:
                    w = torch_npu.npu_format_cast(w, 29).t()
                    scale = torch.ones(n, device="npu")
                    pt = torch.ones(m, device="npu")

                    def mm():
                        return torch_npu.npu_quant_matmul(
                            a, w, scale, pertoken_scale=pt, output_dtype=torch.bfloat16
                        )

                reference = mm().cpu()
                output = [None]

                def gemm():
                    output[0] = mm()

                with torch.npu.stream(compute):
                    gemm()
                compute.synchronize()
                cg = capture(gemm, compute, repeats)
                for mib in (16, 64):
                    size = mib * 1024**2
                    hp = host(size, 37)
                    ho = host(size, 0)
                    src = torch.full((size,), 37, dtype=torch.uint8, device="npu")
                    dst = torch.zeros_like(src)
                    torch.npu.synchronize()
                    for kind, target, source, direction in [
                        ("h2d", dst.data_ptr(), hp, 1),
                        ("d2h", ho, src.data_ptr(), 2),
                        ("d2d_local", dst.data_ptr(), src.data_ptr(), 3),
                    ]:

                        def transfer():
                            copy(target, source, size, direction)

                        with torch.npu.stream(dma):
                            transfer()
                        dma.synchronize()
                        dg = capture(transfer, dma, repeats)
                        replay(cg, compute)
                        replay(dg, dma)
                        for trial in range(3):
                            modes = ["compute", "copy", "serial", "overlap"]
                            if trial % 2:
                                modes.reverse()
                            for mode in modes:
                                torch.npu.synchronize()
                                begin, end, ms, me, cs, ce = [
                                    torch.npu.Event(enable_timing=True)
                                    for _ in range(6)
                                ]
                                with torch.npu.stream(compute):
                                    begin.record()
                                    if mode != "copy":
                                        ms.record()
                                        cg.replay()
                                        me.record()
                                if mode != "compute":
                                    with torch.npu.stream(dma):
                                        dma.wait_event(
                                            me if mode == "serial" else begin
                                        )
                                        cs.record()
                                        dg.replay()
                                        ce.record()
                                with torch.npu.stream(compute):
                                    if mode != "compute":
                                        compute.wait_event(ce)
                                    end.record()
                                end.synchronize()
                                if mode != "copy":
                                    assert torch.equal(output[0].cpu(), reference)
                                if mode != "compute":
                                    if direction == 2:
                                        assert (
                                            C.string_at(ho, size) == bytes([37]) * size
                                        )
                                    else:
                                        assert bool((dst.cpu() == 37).all())
                                row = dict(
                                    label=label,
                                    m=m,
                                    k=k,
                                    n=n,
                                    mib=mib,
                                    direction=kind,
                                    mode=mode,
                                    trial=trial,
                                    repeats=repeats,
                                    total_ms=begin.elapsed_time(end) / repeats,
                                    compute_ms=(
                                        ms.elapsed_time(me) / repeats
                                        if mode != "copy"
                                        else None
                                    ),
                                    copy_ms=(
                                        cs.elapsed_time(ce) / repeats
                                        if mode != "compute"
                                        else None
                                    ),
                                )
                                records.append(row)
                                with args.out.joinpath("records.jsonl").open("a") as f:
                                    f.write(json.dumps(row) + "\n")
                        dg.reset()
                        graphs.remove(dg)
                        print(f"PASS {label} M{m} {kind} {mib}MiB", flush=True)
                    call("aclrtFreeHost", hp)
                    call("aclrtFreeHost", ho)
                    hosts = [p for p in hosts if p.value not in (hp, ho)]
                cg.reset()
                graphs.remove(cg)
        args.out.joinpath("success.json").write_text(
            json.dumps({"pass": True, "records": len(records)})
        )
    finally:
        torch.npu.synchronize()
        for graph in graphs:
            graph.reset()
        for p in hosts:
            call("aclrtFreeHost", p)


if __name__ == "__main__":
    main()
