"""Two-rank HCCL capture/retire/recapture lifecycle isolation (no model)."""

import argparse
import gc
import json
import os
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
        timeout=timedelta(seconds=120),
    )
    dev = f"npu:{rank}"
    prime = torch.ones(1, device=dev)
    dist.all_reduce(prime)
    torch.npu.synchronize()
    stream = torch.npu.Stream()

    def capture(pool):
        catalog = {}
        for rows in (1026, 12, 6):
            x = torch.full((rows, 4096), rank + 1, device=dev, dtype=torch.bfloat16)
            y = torch.empty((2 * rows, 4096), device=dev, dtype=x.dtype)
            with torch.npu.stream(stream):
                dist.all_gather_into_tensor(y, x)
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, pool=pool, stream=stream):
                dist.all_gather_into_tensor(y, x)
            stream.synchronize()
            catalog[rows] = graph, x, y
        return catalog

    before = torch.npu.memory_allocated()
    trial_pool = torch.npu.graph_pool_handle()
    trial = capture(trial_pool)
    torch.npu.synchronize()
    if os.environ.get("RETAIN_TRIAL") == "tensors":
        retained_tensors = [(x, y) for _, x, y in trial.values()]
    if os.environ.get("RETAIN_TRIAL") != "1":
        del trial, trial_pool
    gc.collect()
    torch.npu.empty_cache()
    retired = torch.npu.memory_allocated()
    # New backing perturbs freed addresses, as real KV allocation does.
    backing = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    final_pool = torch.npu.graph_pool_handle()
    catalog = capture(final_pool)
    checks = []
    for turn, rows in enumerate((6, 12, 6, 1026, 12, 1026)):
        print(f"rank={rank} replay turn={turn} rows={rows}", flush=True)
        graph, x, y = catalog[rows]
        with torch.npu.stream(stream):
            x.fill_(rank + 1 + turn)
            graph.replay()
        stream.synchronize()
        host = y.cpu()
        for peer in range(2):
            assert torch.all(host[peer * rows : (peer + 1) * rows] == peer + 1 + turn)
        checks.append(dict(turn=turn, rows=rows, exact=True))
    assert backing.sum().item() == 0
    (out / f"rank{rank}.json").write_text(
        json.dumps(
            dict(
                checks=checks,
                before=before,
                after_retirement=retired,
                peak_reserved=torch.npu.max_memory_reserved(),
            ),
            indent=2,
        )
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import torch.multiprocessing

    torch.multiprocessing.spawn(worker, args=(args.out,), nprocs=2, join=True)
