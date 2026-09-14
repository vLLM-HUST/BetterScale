"""Two-rank, model-free shared-pool graph/eager AllGather lifetime probe."""

import argparse
import json
from datetime import timedelta
from pathlib import Path


def worker(rank, out):
    import torch
    import torch_npu  # noqa: F401
    import torch.distributed as dist

    torch.set_num_threads(2)
    torch.npu.set_device(rank)
    dist.init_process_group(
        "hccl",
        init_method="tcp://127.0.0.1:31739",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    dev = f"npu:{rank}"
    prime = torch.ones(1, device=dev)
    dist.all_reduce(prime)
    torch.npu.synchronize()
    pool = torch.npu.graph_pool_handle()
    stream = torch.npu.Stream()
    catalog = []
    # Target-sized program followed by twelve bounded draft programs. Inputs
    # are persistent, intermediates and communication output belong to ONE pool.
    for rows in (516, 3, 24, 20, 20, 18, 15, 15, 12, 10, 10, 6, 5, 5):
        x = torch.full((rows, 4096), rank + 1, device=dev, dtype=torch.bfloat16)
        graph = torch.npu.NPUGraph()
        with torch.npu.stream(stream):
            y = torch.empty((2 * rows, 4096), device=dev, dtype=x.dtype)
            dist.all_gather_into_tensor(y, x)
        stream.synchronize()
        del y
        with torch.npu.graph(graph, pool=pool, stream=stream):
            temp = x + 1
            y = torch.empty((2 * rows, 4096), device=dev, dtype=x.dtype)
            dist.all_gather_into_tensor(y, temp)
        stream.synchronize()
        catalog.append((rows, graph, x, y))
    checks = []
    for turn in range(3):
        for index in (0, 2, 3, 1, 13, 0, 9):
            rows, graph, x, y = catalog[index]
            print(
                f"rank={rank} graph turn={turn} index={index} rows={rows}", flush=True
            )
            with torch.npu.stream(stream):
                x.fill_(rank + turn + 1)
                graph.replay()
            stream.synchronize()
            host = y.cpu()
            for peer in range(2):
                assert torch.all(
                    host[peer * rows : (peer + 1) * rows] == peer + turn + 2
                )
            checks.append(dict(kind="graph", turn=turn, rows=rows))
            # Both capture-sized and unpadded context ingestion collectives.
            for eager_rows in (516, 514, 512, 1):
                a = torch.full((eager_rows, 4096), rank + 7, device=dev, dtype=x.dtype)
                b = torch.empty((2 * eager_rows, 4096), device=dev, dtype=x.dtype)
                print(f"rank={rank} eager rows={eager_rows}", flush=True)
                dist.all_gather_into_tensor(b, a)
                torch.npu.synchronize()
                host = b.cpu()
                for peer in range(2):
                    assert torch.all(
                        host[peer * eager_rows : (peer + 1) * eager_rows] == peer + 7
                    )
                checks.append(dict(kind="eager", turn=turn, rows=eager_rows))
                del a, b
    (out / f"rank{rank}.json").write_text(
        json.dumps(
            dict(checks=checks, peak_reserved=torch.npu.max_memory_reserved()), indent=2
        )
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import torch.multiprocessing

    torch.multiprocessing.spawn(worker, args=(args.out,), nprocs=2, join=True)
