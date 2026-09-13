"""Bounded paired-stream KV-traffic/GEMM probe, not a serving speedup claim."""

import argparse
from datetime import timedelta
import json
import multiprocessing as mp
import os
from pathlib import Path
import time


def worker(rank, devices, out, payloads, rows):
    import torch
    import torch_npu
    import torch.distributed as dist

    torch.set_num_threads(2)
    torch.npu.config.allow_internal_format = True
    device = devices[rank]
    torch.npu.set_device(device)
    dist.init_process_group(
        "hccl",
        init_method="tcp://127.0.0.1:31726",
        rank=rank,
        world_size=len(devices),
        timeout=timedelta(seconds=180),
    )
    dev = f"npu:{device}"
    prime = torch.ones(1, device=dev)
    dist.all_reduce(prime)
    torch.npu.synchronize()
    main = torch.npu.Stream()
    comm = torch.npu.Stream()
    reports = []
    # Representative projections, not an extraction of every donor kernel shape.
    for label, k, n in [
        ("q_b_like", 1024, 32768),
        ("dense_like", 4096, 4096),
        ("bf16_oproj_like", 1024, 4096),
    ]:
        for m in rows:
            torch.manual_seed(617 + rank)
            a = torch.randint(-2, 3, (m, k), dtype=torch.int8, device=dev)
            w = torch_npu.npu_format_cast(
                torch.randint(-2, 3, (n, k), dtype=torch.int8, device=dev), 29
            ).t()
            scale = torch.ones(n, dtype=torch.float32, device=dev)
            pt = torch.ones(m, dtype=torch.float32, device=dev)

            def mm():
                return torch_npu.npu_quant_matmul(
                    a, w, scale, pertoken_scale=pt, output_dtype=torch.bfloat16
                )

            if label == "bf16_oproj_like":
                a = a.to(torch.bfloat16)
                w = torch_npu.npu_format_cast(w, 2).to(torch.bfloat16)

                def mm():
                    return torch.mm(a, w)

            reference = mm().cpu()
            for mib in payloads:
                # AllGather receive capacity is mib; broadcast transports mib.
                total = mib * 1024 * 1024 // 4
                for collective in ["allgather", "broadcast"]:
                    inp = torch.full(
                        (total // len(devices),),
                        rank + 1,
                        dtype=torch.int32,
                        device=dev,
                    )
                    buf = torch.full((total,), rank + 1, dtype=torch.int32, device=dev)

                    def transfer():
                        if collective == "allgather":
                            dist.all_gather_into_tensor(buf, inp)
                        else:
                            dist.broadcast(buf, src=0)

                    with torch.npu.stream(comm):
                        transfer()
                    comm.synchronize()
                    repeats = 32
                    graphs = {}
                    for mode in ["compute", "comm"]:
                        selected = main if mode == "compute" else comm
                        with torch.npu.stream(selected):
                            if mode == "compute":
                                warm_output = mm()
                            else:
                                transfer()
                        selected.synchronize()
                        graph = torch.npu.NPUGraph()
                        output = None
                        with torch.npu.graph(graph, stream=selected):
                            for _ in range(repeats):
                                if mode == "compute":
                                    output = mm()
                                else:
                                    transfer()
                        selected.synchronize()
                        graphs[mode] = (graph, output)
                    for mode, selected in [("compute", main), ("comm", comm)]:
                        with torch.npu.stream(selected):
                            for _ in range(3):
                                graphs[mode][0].replay()
                        selected.synchronize()
                    dist.barrier()
                    torch.npu.synchronize()
                    for trial in range(3):
                        modes = (
                            ["compute", "comm", "serial", "overlap"]
                            if trial % 2 == 0
                            else ["overlap", "serial", "comm", "compute"]
                        )
                        for mode in modes:
                            dist.barrier()
                            torch.npu.synchronize()
                            begin, end, ms, me, cs, ce = [
                                torch.npu.Event(enable_timing=True) for _ in range(6)
                            ]
                            compute_graph, output = graphs["compute"]
                            comm_graph, _ = graphs["comm"]
                            with torch.npu.stream(main):
                                begin.record(main)
                                if mode != "comm":
                                    ms.record(main)
                                    compute_graph.replay()
                                    me.record(main)
                            if mode != "compute":
                                with torch.npu.stream(comm):
                                    comm.wait_event(me if mode == "serial" else begin)
                                    cs.record(comm)
                                    comm_graph.replay()
                                    ce.record(comm)
                            with torch.npu.stream(main):
                                if mode != "compute":
                                    main.wait_event(ce)
                                end.record(main)
                            end.synchronize()
                            record = dict(
                                rank=rank,
                                physical_device=(
                                    int(
                                        os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(
                                            ","
                                        )[device]
                                    )
                                    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
                                    else device
                                ),
                                logical_device=device,
                                weight_format=torch_npu.get_npu_format(w),
                                repeats=repeats,
                                label=label,
                                m=m,
                                k=k,
                                n=n,
                                collective=collective,
                                receive_mib=mib,
                                mode=mode,
                                trial=trial,
                                wave_ms=begin.elapsed_time(end) / repeats,
                                mm_ms=(
                                    ms.elapsed_time(me) / repeats
                                    if mode != "comm"
                                    else None
                                ),
                                comm_ms=(
                                    cs.elapsed_time(ce) / repeats
                                    if mode != "compute"
                                    else None
                                ),
                            )
                            reports.append(record)
                            if mode != "comm":
                                assert torch.equal(output.cpu(), reference), (
                                    "compute differs",
                                    record,
                                )
                            if mode != "compute":
                                got = buf.cpu()
                                if collective == "allgather":
                                    for i in range(len(devices)):
                                        assert bool(
                                            (
                                                got[
                                                    i
                                                    * inp.numel() : (i + 1)
                                                    * inp.numel()
                                                ]
                                                == i + 1
                                            ).all()
                                        )
                                else:
                                    assert bool((got == 1).all())
                    for graph, *_ in graphs.values():
                        graph.reset()
                    graphs.clear()
                    Path(out, f"rank-{rank}.json").write_text(
                        json.dumps(reports, indent=2)
                    )
                    if rank == 0:
                        print(
                            f"PASS {label} M={m} {collective} receive={mib}MiB",
                            flush=True,
                        )
    dist.barrier()
    torch.npu.synchronize()
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--devices", default="3,6")
    p.add_argument("--out", required=True)
    p.add_argument("--payloads", default="16,64")
    p.add_argument("--rows", default="24,516,4096")
    args = p.parse_args()
    devices = list(map(int, args.devices.split(",")))
    assert len(devices) == len(set(devices)) and len(devices) >= 2
    ctx = mp.get_context("spawn")
    children = [
        ctx.Process(
            target=worker,
            args=(
                r,
                devices,
                args.out,
                list(map(int, args.payloads.split(","))),
                list(map(int, args.rows.split(","))),
            ),
        )
        for r in range(len(devices))
    ]
    try:
        for child in children:
            child.start()
        deadline = time.monotonic() + 600
        for child in children:
            child.join(max(0, deadline - time.monotonic()))
            assert child.exitcode == 0, child.exitcode
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
        for child in children:
            child.join(10)


if __name__ == "__main__":
    main()
