"""Prequeued changing-input sources; no host retirement between expert requests."""

import json
import os
from pathlib import Path
import time
import torch
import torch_npu
from profile_capture import start, stop


def run(service, model, links, root, rank, base_up, base_down, bank_type):
    jobs = []
    for step in range(service.TASKS):
        torch.manual_seed(900 + rank * 100 + step)
        x = (torch.randn(32, 2048) * 0.1).to(torch.bfloat16).npu()
        ids = (
            ((torch.arange(256).reshape(32, 8) + rank * 8 + step * 8) % 128)
            .to(torch.int32)
            .npu()
        )
        values = torch.stack(
            [
                torch_npu.npu_swiglu(x @ (base_up * factor).to(torch.bfloat16))
                @ base_down
                for factor in (0.5, 1.0)
            ],
            dim=1,
        )
        expected = (
            values.gather(1, (ids % 2).long()[:, :, None].expand(-1, -1, 2048))
            .float()
            .mean(1)
            .to(torch.bfloat16)
        )
        jobs.append((step % 2, x, ids, expected))
    remote = service.DeviceExperts(model, links, [])
    for layer in (0, 1):
        bank = bank_type(remote, layer, None, jobs[0][1])
        bank.config[6] = 0
        remote.banks[layer, 32] = bank
    history = torch.empty((service.TASKS, 32, 2048), dtype=torch.bfloat16, device="npu")
    episode = torch.npu.NPUGraph()
    with torch.npu.graph(episode):
        for step, (layer, x, ids, _) in enumerate(jobs):
            bank = remote.banks[layer, 32]
            bank.input.copy_(x)
            bank.ids.copy_(ids)
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
    if delayed:
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
    for step, (layer, _, _, expected) in enumerate(jobs):
        actual = history[step]
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=2e-5)
        relative = (
            torch.linalg.vector_norm(actual.float() - expected.float())
            / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        ).item()
        assert relative < 0.01
        results.append(
            dict(
                pattern="balanced",
                rows_per_source=32,
                repeat=step,
                layer=layer,
                relative_l2=relative,
                exact=torch.equal(actual, expected),
            )
        )
    elapsed = a.elapsed_time(b) * 1000
    episode.reset()
    remote.close()
    stop(profiler)
    Path(root, f"client{rank}.json").write_text(json.dumps(results, indent=2))
    Path(root, f"burst{rank}.json").write_text(
        json.dumps(
            dict(
                source=rank,
                jobs=len(jobs),
                one_host_replay=True,
                episode_us=elapsed,
                deliberate_delay_ms=delay,
                max_relative_l2=max(r["relative_l2"] for r in results),
            ),
            indent=2,
        )
    )
