"""Verify the bounded device service's raw receipts; emit a compact report."""

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def summarize(primitive, joint):
    standalone = read(primitive / "result.json")
    assert standalone["status"] == "pass"
    assert standalone["bounded_service_graph"]
    assert standalone["runtime_replays_per_rank_per_episode"] == 1
    primitive_clients = [c for epoch in standalone["results"] for c in epoch[1:]]
    assert len(primitive_clients) == 4
    assert all(c["output_exact"] and c["guards_exact"] for c in primitive_clients)
    assert sum(c["tasks"] for c in primitive_clients) == 64
    result = read(joint / "result.json")
    assert result["status"] == "pass" and not result["host_control"]
    attention, servers = [], []
    for i in range(2):
        client = read(joint / f"attention{i}.json")
        done = read(joint / f"attention{i}-complete.json")
        checks = client["checks"]
        assert done["complete"] and done["banks"] == [6]
        assert len(checks) == 12
        assert all(c["output_exact"] and c["kv_exact"] for c in checks)
        assert sum(not c["preparing"] for c in checks) == 6
        assert sum(c["attention_replays"] for c in checks) == 24
        attention.append(
            dict(
                client=i,
                forwards=12,
                exact_output=True,
                exact_kv=True,
                sealed_forwards=6,
                attention_banks=6,
            )
        )
        expert = read(joint / f"expert{i}.json")
        assert expert["status"][:4] == [24, 24, 48, 0]
        assert expert["graph_replays"] == 1 and not expert["host_control"]
        trace = expert["trace"]
        assert len(trace) == 48 and sum(r[0] for r in trace) == 48
        assert all(r[4] == 0 for r in trace)
        for c in range(2):
            assert [r[2 + c] for r in trace if r[2 + c]] == list(range(1, 25))
        servers.append(
            dict(
                server=i,
                completed_source_layers=48,
                runtime_replays=1,
                cross_source_waves=sum(r[0] == 2 for r in trace),
            )
        )
    return dict(
        status="pass",
        primitive_artifact=str(primitive),
        joint_artifact=str(joint),
        primitive_exact_tasks=64,
        model="Qwen3-30B-A3B dimensions; two dummy BF16 layers",
        attention=attention,
        servers=servers,
        bounded_service_graph=True,
        persistent_server=False,
        host_attention_scheduling=True,
        performance_claim=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("primitive", type=Path)
    parser.add_argument("joint", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.primitive, args.joint), indent=2))
