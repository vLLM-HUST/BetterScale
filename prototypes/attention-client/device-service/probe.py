"""Device-authored batches drive unchanged native BF16 expert graph bodies.

Three cards: one expert server, two persistent AIV clients. Host prequeues a
bounded number of service graph cycles; it never reads per-wave counts/replies.
"""

import argparse
import ctypes as C
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import time

H, M, GROUPS, ROWS, CAP, TASKS = 2048, 768, 128, 8, 16, 16
BYTES = 2 * 1024**2


def recv(pipe):
    if not pipe.poll(120):
        raise TimeoutError("device service bootstrap/retirement timeout")
    return pipe.recv()


def plans(torch, owner, epoch):
    metadata = torch.tensor(
        [
            [
                t + 1,
                t,
                (t // 3 + epoch) % 2,
                (63 if t % 4 == 3 else t % 3),
                t % 2,
                1 + (t + owner) % ROWS,
                0,
                0,
            ]
            for t in range(TASKS)
        ],
        dtype=torch.int32,
    )
    inputs = (
        (
            (torch.arange(TASKS * ROWS * H, dtype=torch.float32) % 97 - 48) / 128
            + owner / 4
            + epoch / 10
        )
        .reshape(TASKS, ROWS, H)
        .to(torch.bfloat16)
    )
    return metadata, inputs


def worker(rank, pipe, out, build):
    import torch
    import torch_npu

    torch.set_num_threads(2)
    torch.npu.set_device(rank)
    path = "/workspace/my-ascend-workspace/stateharbor/src/livemodule/arch/ascend/request_parallel/dsv4/prefix_copy_acl.py"
    spec = importlib.util.spec_from_file_location("ipc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    api = module.PrefixCopyACL("/usr/local/Ascend/cann-9.0.1/lib64/libascendcl.so")
    api.lib.aclrtMemset.argtypes = [C.c_void_p, C.c_size_t, C.c_int32, C.c_size_t]
    api.lib.aclrtMemset.restype = C.c_int32
    local = api.allocate_staging(BYTES)
    pipe.send(api.pid())
    pids = recv(pipe)
    key = api.export(local, BYTES, tuple(pids[1:]) if rank == 0 else (pids[0],))
    pipe.send(key)
    keys = recv(pipe)
    peers = [api.import_memory(k) for k in keys]
    lib = C.CDLL(str(Path(build, "launch.so")))
    lib.load_server.argtypes = [
        C.c_char_p,
        C.c_char_p,
        C.POINTER(C.c_void_p),
        C.POINTER(C.c_void_p),
    ]
    lib.launch_server.argtypes = [C.c_void_p] * 5
    lib.unload_server.argtypes = [C.c_void_p]
    binaries = []

    def load(symbol):
        binary, fn = C.c_void_p(), C.c_void_p()
        rc = lib.load_server(
            str(Path(build, "queue_service.o")).encode(),
            symbol.encode(),
            C.byref(binary),
            C.byref(fn),
        )
        assert rc == 0, (symbol, rc)
        binaries.append(binary)
        return fn

    stream = torch.npu.Stream()
    config = torch.zeros(16, dtype=torch.int64, device=f"npu:{rank}")

    def launch(fn, a, b):
        rc = lib.launch_server(
            fn, stream.npu_stream, config.data_ptr(), a.data_ptr(), b.data_ptr()
        )
        assert rc == 0, rc

    waves = TASKS * 2 + 2
    if rank == 0:
        prepare, complete = load("queue_prepare"), load("queue_complete")
        torch.manual_seed(41)
        w13 = torch.empty(
            (GROUPS, H, 2 * M), dtype=torch.bfloat16, device="npu"
        ).normal_(0, 0.02)
        w2 = torch.empty((GROUPS, M, H), dtype=torch.bfloat16, device="npu").normal_(
            0, 0.02
        )
        packed = torch.zeros((CAP, H), dtype=torch.bfloat16, device="npu")
        groups = torch.zeros(GROUPS, dtype=torch.int64, device="npu")
        groups[-1] = CAP
        state = torch.zeros(8, dtype=torch.int32, device="npu")
        batch = torch.zeros(16, dtype=torch.int32, device="npu")
        trace = torch.full((waves, 8), -991, dtype=torch.int32, device="npu")
        base = [
            local,
            *peers,
            2,
            TASKS,
            0,
            2000000,
            state.data_ptr(),
            batch.data_ptr(),
            groups.data_ptr(),
            trace.data_ptr(),
        ]
        config[: len(base)].copy_(torch.tensor(base, dtype=torch.int64))

        def body():
            packed.zero_()
            launch(prepare, packed, state)
            up = torch_npu.npu_grouped_matmul(
                [packed],
                [w13],
                split_item=2,
                group_list=groups,
                group_type=0,
                group_list_type=0,
            )[0]
            act = torch_npu.npu_swiglu(up)
            output = torch_npu.npu_grouped_matmul(
                [act],
                [w2],
                split_item=2,
                group_list=groups,
                group_type=0,
                group_list_type=0,
            )[0]
            launch(complete, output, state)

        # Independent eager, single-expert arithmetic. Not service timing work.
        for epoch in (1, 2):
            for owner in range(2):
                metadata, inputs = plans(torch, owner, epoch)
                inputs = inputs.npu()
                gold = []
                for t, desc in enumerate(metadata.tolist()):
                    g = desc[2] * 64 + desc[3]
                    n = desc[5]
                    gold.append(
                        (torch_npu.npu_swiglu(inputs[t, :n] @ w13[g]) @ w2[g]).cpu()
                    )
                torch.save(gold, Path(out, f"gold-{owner}-{epoch}.pt"))
    else:
        client = load("queue_client")
        metadata = torch.zeros((TASKS, 8), dtype=torch.int32, device="npu")
        inputs = torch.zeros((TASKS, ROWS, H), dtype=torch.bfloat16, device="npu")
        status = torch.full((8,), -991, dtype=torch.int32, device="npu")
        storage = torch.full(
            (TASKS * ROWS + 2, H), -17, dtype=torch.bfloat16, device="npu"
        )
        output = storage[1:-1].view(TASKS, ROWS, H)
        base = [
            local,
            peers[0],
            rank - 1,
            TASKS,
            0,
            2000000,
            status.data_ptr(),
            output.data_ptr(),
        ]
        config[: len(base)].copy_(torch.tensor(base, dtype=torch.int64))

        def body():
            launch(client, metadata, inputs)

    torch.npu.synchronize()
    with torch.npu.stream(stream):
        body()
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        for _ in range(waves if rank == 0 else 1):
            body()
    stream.synchronize()
    pipe.send("ready")
    time.sleep(4)  # outside execution, let the subset lease monitor see processes
    records = []
    for epoch in (1, 2):
        assert api.lib.aclrtMemset(local, BYTES, 0, BYTES) == 0
        if rank == 0:
            state.zero_()
            batch.zero_()
            trace.fill_(-991)
            config[5] = 1
        else:
            cpu_meta, cpu_input = plans(torch, rank - 1, epoch)
            metadata.copy_(cpu_meta)
            inputs.copy_(cpu_input)
            status.fill_(-991)
            storage.fill_(-17)
            config[4] = 1
        torch.npu.synchronize()
        pipe.send("armed")
        start = recv(pipe)
        while time.monotonic() < start:
            time.sleep(0.001)
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
            enable_timing=True
        )
        with torch.npu.stream(stream):
            begin.record(stream)
            graph.replay()
            end.record(stream)
        end.synchronize()
        if rank == 0:
            s = state.cpu().tolist()
            assert s[:4] == [TASKS, TASKS, waves, 0], s
            records_cpu = trace.cpu().tolist()
            assert sum(r[0] for r in records_cpu) == TASKS * 2
            seen = [[], []]
            for record in records_cpu:
                assert record[6] == 0
                for c in range(2):
                    if record[2 + c]:
                        seen[c].append(record[2 + c])
            assert seen == [list(range(1, TASKS + 1))] * 2
            row = dict(
                epoch=epoch,
                state=s,
                trace=records_cpu,
                coalesced_waves=sum(r[0] == 2 for r in records_cpu),
            )
        else:
            s = status.cpu().tolist()
            assert s[0] == 1, s
            got = storage.cpu()
            gold = torch.load(Path(out, f"gold-{rank-1}-{epoch}.pt"), weights_only=True)
            assert bool((got[0] == -17).all()) and bool((got[-1] == -17).all())
            exact = True
            max_abs = 0.0
            max_relative_l2 = 0.0
            for t, (desc, reference) in enumerate(zip(cpu_meta.tolist(), gold)):
                n = desc[5]
                actual = got[1 + t * ROWS : 1 + t * ROWS + n]
                difference = (actual.float() - reference.float()).abs()
                relative = (
                    torch.linalg.vector_norm(difference)
                    / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
                ).item()
                torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.0002)
                assert relative < 0.01, relative
                assert bool(
                    (got[1 + t * ROWS + n : 1 + (t + 1) * ROWS] == -17).all()
                ), "padding clobbered"
                exact &= torch.equal(actual, reference)
                max_abs = max(max_abs, difference.max().item())
                max_relative_l2 = max(max_relative_l2, relative)
            row = dict(
                epoch=epoch,
                tasks=TASKS,
                output_exact=exact,
                max_abs=max_abs,
                relative_l2=max_relative_l2,
                guards_exact=True,
            )
        row["device_ms_including_waits"] = begin.elapsed_time(end)
        records.append(row)
        pipe.send(row)
        assert recv(pipe) == "next"
    graph.reset()
    torch.npu.synchronize()
    for binary in binaries:
        assert lib.unload_server(binary) == 0
    for k in keys:
        api.close_mapping(k)
    pipe.send("unmapped")
    assert recv(pipe) == "release"
    api.close_mapping(key)
    api.free_staging(local)
    Path(out, f"rank{rank}.json").write_text(json.dumps(records, indent=2))
    pipe.send("released")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--build", required=True)
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    pipes = []
    processes = []
    try:
        for rank in range(3):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=worker, args=(rank, child, a.out, a.build))
            process.start()
            child.close()
            pipes.append(parent)
            processes.append(process)
        pids = [recv(p) for p in pipes]
        for p in pipes:
            p.send(pids)
        keys = [recv(p) for p in pipes]
        for rank, p in enumerate(pipes):
            p.send(keys[1:] if rank == 0 else [keys[0]])
        assert all(recv(p) == "ready" for p in pipes)
        results = []
        for _ in range(2):
            assert all(recv(p) == "armed" for p in pipes)
            start = time.monotonic() + 0.1
            for p in pipes:
                p.send(start)
            results.append([recv(p) for p in pipes])
            for p in pipes:
                p.send("next")
        assert all(recv(p) == "unmapped" for p in pipes)
        for p in pipes:
            p.send("release")
        assert all(recv(p) == "released" for p in pipes)
        for process in processes:
            process.join(15)
            assert process.exitcode == 0
        Path(a.out, "result.json").write_text(
            json.dumps(
                dict(
                    status="pass",
                    results=results,
                    host_per_wave_decisions=False,
                    bounded_service_graph=True,
                    runtime_replays_per_rank_per_episode=1,
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
