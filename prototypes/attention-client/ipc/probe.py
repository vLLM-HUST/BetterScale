"""External client packets against the unchanged 3532418 integer server.

No neural GEMM claim. Each client publishes eight independent routed packets
from a persistent FULL graph, then reduces returned top-k rows on the CPU oracle.
"""

import argparse
import ctypes as C
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from continuation import Continuations, Route
from server_contract import OraclePackets

WORKSPACE = Path("/workspace/my-ascend-workspace")
BYTES = 2 * 1024**2


def receive(pipe):
    if not pipe.poll(90):
        raise TimeoutError("joint client/server control timeout")
    return pipe.recv()


def packets(owner, epoch):
    continuations = Continuations(4)
    adapter = OraclePackets(continuations)
    result = []
    for lane in range(4):
        routes = [Route(0, 0, lane % 3, 0.25), Route(0, 1, (lane + 1) % 3, 0.75)]
        ticket = continuations.submit(lane, lane % 2, 1, 64, routes, [[1000] * 64])
        values = [epoch * 10000 + owner * 1000 + lane * 100 + c for c in range(64)]
        for route in routes:
            result.append(
                adapter.publish(ticket, route.expert, lane % 2, [route], [values])
            )
    return continuations, adapter, result


def worker(rank, clients, pipes, out, client_build, server_build):
    import torch
    import torch_npu

    torch.set_num_threads(2)
    torch.npu.set_device(rank)
    spec = importlib.util.spec_from_file_location(
        "ipc",
        WORKSPACE
        / "stateharbor/src/livemodule/arch/ascend/request_parallel/dsv4/prefix_copy_acl.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    api = module.PrefixCopyACL("/usr/local/Ascend/cann-9.0.1/lib64/libascendcl.so")
    local = api.allocate_staging(BYTES)
    api.lib.aclrtMemset.argtypes = [C.c_void_p, C.c_size_t, C.c_int32, C.c_size_t]
    api.lib.aclrtMemset.restype = C.c_int32
    pipes.send(api.pid())
    pids = receive(pipes)
    key = api.export(local, BYTES, tuple(pids[1:]) if rank == 0 else (pids[0],))
    pipes.send(key)
    keys = receive(pipes)
    peers = [api.import_memory(k) for k in keys]
    build = Path(server_build if rank == 0 else client_build)
    lib = C.CDLL(str(build / "launch.so"))
    lib.load_server.argtypes = [
        C.c_char_p,
        C.POINTER(C.c_void_p),
        C.POINTER(C.c_void_p),
    ]
    lib.launch_server.argtypes = [C.c_void_p] * 5
    lib.unload_server.argtypes = [C.c_void_p]
    binary, function = C.c_void_p(), C.c_void_p()
    name = "pull_expert_server.o" if rank == 0 else "pull_expert_client.o"
    assert (
        lib.load_server(str(build / name).encode(), C.byref(binary), C.byref(function))
        == 0
    )
    config = torch.zeros(16, dtype=torch.int64, device=f"npu:{rank}")
    plan = torch.zeros((8, 520), dtype=torch.int32, device=f"npu:{rank}")
    output = torch.full(
        (64 + 8 * 512 + 64,), -991, dtype=torch.int32, device=f"npu:{rank}"
    )
    trace = torch.full((16 * 8 + 64,), -991, dtype=torch.int32, device=f"npu:{rank}")
    base = (
        [local, *(peers + [0, 0])[:2], local, 0, clients, 0, 200000, 0, 0, 1, 0]
        if rank == 0
        else [local, peers[0], rank - 1, 0, 200000]
    )
    config[: len(base)].copy_(torch.tensor(base, dtype=torch.int64))
    stream = torch.npu.Stream()

    def launch():
        args = (output, trace) if rank == 0 else (plan, output)
        assert (
            lib.launch_server(
                function,
                stream.npu_stream,
                config.data_ptr(),
                args[0].data_ptr(),
                args[1].data_ptr(),
            )
            == 0
        )

    torch.npu.synchronize()
    launch()
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        launch()
    stream.synchronize()
    pipes.send("ready")
    # Outside measured work: allow the subset admission monitor to see workers.
    time.sleep(4)
    records = []
    for epoch in (1, 2):
        assert api.lib.aclrtMemset(local, BYTES, 0, BYTES) == 0
        output.fill_(-991)
        trace.fill_(-991)
        if rank == 0:
            config[6] = 8
        else:
            control, adapter, tasks = packets(rank - 1, epoch)
            cpu_plan = torch.zeros((8, 520), dtype=torch.int32)
            for packet in tasks:
                cpu_plan[packet.task_id, :8] = torch.tensor(
                    packet.descriptor, dtype=torch.int32
                )
                cpu_plan[packet.task_id, 8:72] = torch.tensor(
                    packet.payload[0], dtype=torch.int32
                )
            plan.copy_(cpu_plan)
            config[3] = 8
        torch.npu.synchronize()
        pipes.send("armed")
        start = receive(pipes)
        while time.monotonic() < start:
            time.sleep(0.001)
        with torch.npu.stream(stream):
            graph.replay()
        stream.synchronize()
        cpu = output.cpu()
        assert cpu[0].item() == 1 and cpu[1].item() == (
            8 * clients if rank == 0 else 8
        ), cpu[:8]
        assert bool((cpu[8:64] == -991).all())
        if rank:
            for packet in reversed(tasks):
                row = cpu[
                    64 + packet.task_id * 512 : 64 + packet.task_id * 512 + 64
                ].tolist()
                expected = [
                    x + packet.ticket.layer * 100 + packet.expert * 10
                    for x in packet.payload[0]
                ]
                assert row == expected
                adapter.complete(packet.slot, packet.generation, [row])
                assert bool(
                    (
                        cpu[
                            64
                            + packet.task_id * 512
                            + 64 : 64
                            + (packet.task_id + 1) * 512
                        ]
                        == -991
                    ).all()
                )
            retired = [control.retire() for _ in range(4)]
            for ticket, values in retired:
                lane = ticket.lane
                offset = (
                    epoch * 10000
                    + (rank - 1) * 1000
                    + lane * 100
                    + (lane % 2) * 100
                    + 1000
                )
                expert = 0.25 * (lane % 3) * 10 + 0.75 * ((lane + 1) % 3) * 10
                assert values == [[offset + expert + c for c in range(64)]]
            assert not adapter.active and not control.pending
            assert bool((cpu[64 + 8 * 512 :] == -991).all())
            record = dict(
                rank=rank,
                epoch=epoch,
                packets=8,
                retired_lanes=4,
                weighted_topk_exact=True,
            )
        else:
            batches = trace[: cpu[3].item() * 8].cpu().reshape(-1, 8).tolist()
            assert sum(x[3] for x in batches) == 8 * clients
            record = dict(rank=rank, epoch=epoch, batches=batches)
        records.append(record)
        pipes.send(record)
        assert receive(pipes) == "next"
    graph.reset()
    torch.npu.synchronize()
    assert lib.unload_server(binary) == 0
    for peer_key in keys:
        api.close_mapping(peer_key)
    pipes.send("unmapped")
    assert receive(pipes) == "release"
    api.close_mapping(key)
    api.free_staging(local)
    Path(out, f"rank{rank}.json").write_text(json.dumps(records, indent=2))
    pipes.send("released")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--clients", type=int, choices=(1, 2), default=1)
    parser.add_argument("--client-build", required=True)
    parser.add_argument("--server-build", required=True)
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    processes, pipes = [], []
    try:
        for rank in range(args.clients + 1):
            parent, child = ctx.Pipe()
            process = ctx.Process(
                target=worker,
                args=(
                    rank,
                    args.clients,
                    child,
                    args.out,
                    args.client_build,
                    args.server_build,
                ),
            )
            process.start()
            child.close()
            processes.append(process)
            pipes.append(parent)
        pids = [receive(p) for p in pipes]
        for p in pipes:
            p.send(pids)
        keys = [receive(p) for p in pipes]
        for rank, p in enumerate(pipes):
            p.send(keys[1:] if rank == 0 else [keys[0]])
        assert all(receive(p) == "ready" for p in pipes)
        results = []
        for _ in range(2):
            assert all(receive(p) == "armed" for p in pipes)
            start = time.monotonic() + 0.1
            for p in pipes:
                p.send(start)
            results.append([receive(p) for p in pipes])
            for p in pipes:
                p.send("next")
        assert all(receive(p) == "unmapped" for p in pipes)
        for p in pipes:
            p.send("release")
        assert all(receive(p) == "released" for p in pipes)
        for process in processes:
            process.join(15)
            assert process.exitcode == 0
        Path(args.out, "result.json").write_text(
            json.dumps(
                dict(
                    status="pass",
                    server_commit="3532418",
                    neural_gemm=False,
                    results=results,
                ),
                indent=2,
            )
        )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(10)


if __name__ == "__main__":
    main()
