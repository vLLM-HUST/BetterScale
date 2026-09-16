"""Prequeued changing-input sources; no host retirement between expert requests."""

import json
import os
from pathlib import Path
import time
import torch
import torch_npu
from profile_capture import start, stop


def run(service, model, links, root, rank, base_up, base_down, bank_type):
    sizes = [
        int(v) for v in os.environ.get("DEVICE_SERVICE_BURST_ROWS", "32,32").split(",")
    ]
    assert len(sizes) == 2 and all(1 <= n <= 32 for n in sizes)
    rows = sizes[rank]
    pattern = os.environ.get("DEVICE_SERVICE_ROUTE_PATTERN", "balanced")
    assert pattern in ("balanced", "hot8", "owner0", "oneexpert")
    weight_sets = int(os.environ.get("DEVICE_SERVICE_WEIGHT_SETS", "2"))
    assert weight_sets in (1, 2)
    jobs = []
    for step in range(service.TASKS):
        torch.manual_seed(900 + rank * 100 + step)
        x = (torch.randn(rows, 2048) * 0.1).to(torch.bfloat16).npu()
        ids = (
            ((torch.arange(rows * 8).reshape(rows, 8) + rank * 8 + step * 8) % 128)
            .to(torch.int32)
            .npu()
        )
        if pattern != "balanced":
            routes = {
                "hot8": [0, 1, 2, 3, 64, 65, 66, 67],
                "owner0": list(range(8)),
                "oneexpert": [0] * 8,
            }[pattern]
            ids = (
                torch.tensor(routes, device="npu", dtype=torch.int32)
                .expand(rows, 8)
                .contiguous()
            )
        values = torch.stack(
            [
                torch_npu.npu_swiglu(x @ (base_up * factor).to(torch.bfloat16))
                @ base_down
                for factor in (0.5, 1.0)
            ],
            dim=1,
        )
        probabilities = torch.full((rows, 8), 1 / 8, device="npu", dtype=torch.bfloat16)
        if os.environ.get("DEVICE_SERVICE_RANDOM_PROBS") == "1":
            probabilities = torch.softmax(torch.randn(rows, 8, device="npu"), dim=1).to(
                torch.bfloat16
            )
        expected = (
            (
                values.gather(
                    1, (ids % 2).long()[:, :, None].expand(-1, -1, 2048)
                ).float()
                * probabilities.float().unsqueeze(-1)
            )
            .sum(1)
            .to(torch.bfloat16)
        )
        jobs.append((step % weight_sets, x, ids, expected, probabilities))
    remote = service.DeviceExperts(model, links, [])
    for layer in (0, 1):
        bank = bank_type(remote, layer, None, jobs[0][1])
        bank.config[6] = 0
        remote.banks[layer, rows] = bank
    history = torch.empty(
        (service.TASKS, rows, 2048), dtype=torch.bfloat16, device="npu"
    )
    episode = torch.npu.NPUGraph()
    with torch.npu.graph(episode):
        for step, (layer, x, ids, _, probabilities) in enumerate(jobs):
            bank = remote.banks[layer, rows]
            bank.input.copy_(x)
            bank.ids.copy_(ids)
            if os.environ.get("DEVICE_SERVICE_RANDOM_PROBS") == "1":
                bank.probs.copy_(probabilities)
            history[step].copy_(bank.body())
    for bank in remote.banks.values():
        bank.config[6] = 1
    torch.npu.synchronize()
    profiler = start(f"attention{rank}")
    delayed = float(os.environ.get("DEVICE_SERVICE_SOURCE1_DELAY_MS", "0"))
    delay = delayed if rank else 0.0
    assert 0 <= delayed <= 50
    a = torch.npu.Event(enable_timing=True)
    b = torch.npu.Event(enable_timing=True)
    if delayed or os.environ.get("DEVICE_SERVICE_BURST_PREQUEUE", "1") == "0":
        # Deliberate runtime absence, not merely delayed initialization.
        remote.activate()
        time.sleep(delay / 1000)
        a.record()
        episode.replay()
    else:
        # A common burst is already enqueued when servers begin consuming.
        # This is a startup rendezvous, never a per-wave source barrier.
        def enqueue():
            a.record()
            episode.replay()

        remote.activate(enqueue)
    b.record()
    b.synchronize()
    results = []
    for step, (layer, _, _, expected, _) in enumerate(jobs):
        actual = history[step]
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=2e-5)
        relative = (
            torch.linalg.vector_norm(actual.float() - expected.float())
            / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        ).item()
        assert relative < 0.01
        results.append(
            dict(
                pattern=pattern,
                rows_per_source=rows,
                repeat=step,
                layer=layer,
                relative_l2=relative,
                exact=torch.equal(actual, expected),
            )
        )
    elapsed = a.elapsed_time(b) * 1000
    episode.reset()
    if os.environ.get("DEVICE_SERVICE_ROUTE_PULL") == "1":
        Path(root, f"collect{rank}.json").write_text(
            json.dumps(
                {
                    str(layer): remote.banks[layer, rows].pull_timing.cpu().tolist()
                    for layer in (0, 1)
                }
            )
        )
    remote.close()
    stop(profiler)
    Path(root, f"client{rank}.json").write_text(json.dumps(results, indent=2))
    Path(root, f"burst{rank}.json").write_text(
        json.dumps(
            dict(
                source=rank,
                weight_address_sets=weight_sets,
                route_pattern=pattern,
                random_probabilities=os.environ.get("DEVICE_SERVICE_RANDOM_PROBS")
                == "1",
                jobs=len(jobs),
                one_host_replay=True,
                episode_us=elapsed,
                deliberate_delay_ms=delay,
                max_relative_l2=max(r["relative_l2"] for r in results),
            ),
            indent=2,
        )
    )
