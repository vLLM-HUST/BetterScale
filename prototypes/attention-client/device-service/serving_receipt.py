"""Audit native generation through persistent remote experts; no NPU imports."""

import argparse
import json
from pathlib import Path


def summarize(run):
    root = run / "run/measurements"

    def read(name):
        return json.loads((root / name).read_text())

    assert read("result.json")["status"] == "pass"
    clients, counts = [], []
    for index in range(2):
        data = read(f"attention{index}.json")
        complete = read(f"attention{index}-complete.json")
        checks = data["checks"]
        assert complete["complete"] and complete["banks"] == [6]
        assert checks and all(c["attention_replays"] == 2 for c in checks)
        shadow = complete.get("shadow", True)
        if shadow:
            assert all(c["output_exact"] and c["kv_exact"] for c in checks)
        else:
            assert all(
                c["shadow"] is False and c["reference_calls"] == 0 for c in checks
            )
        submissions = [e for e in data["events"] if e["event"] == "submit"]
        retirements = [e for e in data["events"] if e["event"] == "retire"]
        count = sum(c["attention_replays"] for c in checks)
        expected = list(range(1, count + 1))
        assert [e["generation"] for e in submissions] == expected
        assert [e["generation"] for e in retirements] == expected
        assert all(e["routing_device"] for e in submissions)
        assert all(e["completion_device"] for e in retirements)
        assert [e["layer"] for e in submissions] == [0, 1] * len(checks)
        counts.append(count)
        clients.append(
            dict(
                client=index,
                forwards=len(checks),
                sealed_forwards=sum(not c["preparing"] for c in checks),
                source_layer_jobs=count,
                shadow=shadow,
                reference_calls=len(checks) if shadow else 0,
                generations=complete.get("generations", []),
            )
        )
    servers = []
    for index in range(2):
        data = read(f"expert{index}.json")
        assert data["persistent"] and not data["deliberate_batch_wait"]
        for source, count in enumerate(counts):
            assert [r[source] for r in data["trace"] if r[source]] == list(
                range(1, count + 1)
            )
        servers.append(
            dict(
                server=index,
                waves=data["waves"],
                source_layer_jobs=sum(counts),
                paired=data["same_graph_cross_source_waves"],
            )
        )
    return dict(
        run=str(run),
        status="pass",
        model="Qwen3-30B-A3B dimensions",
        layers=2,
        dummy=True,
        clients=clients,
        servers=servers,
        native_generation=True,
        host_attention_continuation=True,
        bounded_episode=True,
        performance_claim=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run), indent=2))
