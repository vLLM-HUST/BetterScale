"""Bounded two-client concurrency probe; no native attention or shared MLP.

Host rendezvous precedes each measured episode, never enters the server policy.
Both clients issue eight dependency-ordered calls in one captured graph. Only
initial launch is aligned/offset: later waves follow real completion feedback.
"""

import json
import os
from pathlib import Path
import time

import torch
import torch_npu
from next_remote import Session
from profile_capture import start as start_profile, stop as stop_profile


def rendezvous(directory, source, label):
    (directory / f"{label}-ready{source}").touch()
    deadline = time.monotonic() + 60
    other = directory / f"{label}-ready{1-source}"
    while not other.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(label)
        time.sleep(0.001)
    release = directory / f"{label}-release"
    if source == 0:
        temporary = directory / f"{label}-release.tmp"
        temporary.write_text(str(time.monotonic_ns() + 100_000_000))
        temporary.rename(release)
    while not release.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(label)
        time.sleep(0.001)
    return int(release.read_text())


def run():
    source = int(os.environ["EXPERT_SOURCE_ID"])
    directory = Path(os.environ["EXPERT_ROLE_DIRECTORY"])
    session = Session()
    bank = session.banks[32]
    torch.manual_seed(824 + source)
    bank.x.copy_((torch.randn_like(bank.x) * 0.1).bfloat16())
    ids = (
        torch.arange(32, device="npu")[:, None] * 13
        + torch.arange(10, device="npu")[None, :] * 53
    ) % 512
    bank.ids.copy_(ids)
    bank.probs.fill_(0.1)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    bank.config[8] = 0
    with torch.npu.graph(graph, pool=session.pool):
        for _ in range(8):
            session.kernels.call(session.submit, bank.config, bank.x, bank.id_storage)
            session.kernels.call(
                session.collect, bank.config, bank.x, bank.id_storage, blocks=16
            )
            session.kernels.call(session.retire, bank.config, bank.x, bank.id_storage)
            if not session.route_pull:
                bank.output.copy_(
                    torch_npu.npu_moe_token_unpermute(
                        bank.raw, bank.indices, probs=bank.probs
                    )
                )
    bank.config[8] = 1
    profiler = start_profile(f"attention{source}")
    episodes = []
    for label, layer, offset_us in [
        ("same", 0, 0),
        ("different", source, 0),
        ("offset", 2, 100),
    ]:
        bank.config[5] = layer
        # Two unmeasured calls warm this exact catalog before the episode.
        for _ in range(2):
            bank.submit_graph.replay()
            bank.collect_graph.replay()
            torch.npu.synchronize()
        expected = bank.output.clone()
        torch.npu.synchronize()
        release = rendezvous(directory, source, label)
        target = release + (offset_us * 1000 if source else 0)
        while time.monotonic_ns() < target:
            pass
        submitted = time.monotonic_ns()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
            enable_timing=True
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        torch.testing.assert_close(bank.output, expected, rtol=0.02, atol=2e-5)
        episodes.append(
            dict(
                case=label,
                layer=layer,
                rows=32,
                calls=8,
                initial_offset_us=offset_us if source else 0,
                host_submit_ns=submitted,
                episode_ms=start.elapsed_time(end),
            )
        )
        # Don't let a fast source's next-layer warmup invade the peer's timing.
        rendezvous(directory, source, label + "-done")
    stop_profile(profiler)
    graph.reset()
    count = session.close()
    assert count == 30, count
    (directory / f"client{source}.json").write_text(
        json.dumps(dict(status="pass", completed=count, episodes=episodes), indent=2)
    )
