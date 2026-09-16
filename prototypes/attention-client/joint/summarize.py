"""Validate and compact one four-card receipt without importing an NPU runtime."""

import argparse
import json
from pathlib import Path


def summarize(directory):
    root = Path(directory)
    result = json.loads((root / "result.json").read_text())
    assert result["status"] == "pass"
    clients = []
    intervals = []
    submitted = {}
    for client in range(2):
        data = json.loads((root / f"attention{client}.json").read_text())
        complete = json.loads((root / f"attention{client}-complete.json").read_text())
        assert complete["complete"]
        active = {}
        for event in data["events"]:
            generation = event["generation"]
            if event["event"] == "submit":
                assert not active
                assert event["owners"] == [0, 1]
                active[generation] = event
                submitted[client, generation] = event
            else:
                start = active.pop(generation)
                assert event["servers"] == [0, 1]
                intervals.append(
                    (client, generation, start["rows"], start["time"], event["time"])
                )
        assert not active
        checks = data["checks"]
        assert len(checks) == 12
        assert len([c for c in checks if not c["preparing"]]) == 6
        assert {c["rows"] for c in checks} == {1, 16, 32}
        assert all(c["attention_replays"] == 2 for c in checks)
        clients.append(
            dict(
                client=client,
                forward_checks=len(checks),
                sealed_checks=6,
                attention_graphs=complete["banks"][0],
                attention_replays=sum(c["attention_replays"] for c in checks),
                output_exact=all(c["output_exact"] for c in checks),
                kv_exact=all(c["kv_exact"] for c in checks),
                max_abs=max(c["max_abs"] for c in checks),
                relative_l2=max(c["relative_l2"] for c in checks),
            )
        )
    servers = []
    for server in range(2):
        data = json.loads((root / f"expert{server}.json").read_text())
        seen = set()
        for event in data["records"]:
            key = event["client"], event["generation"]
            assert key not in seen
            seen.add(key)
            original = submitted[key]
            assert (
                original["layer"] == event["layer"]
                and original["rows"] == event["rows"]
            )
        assert seen == set(submitted)
        servers.append(
            dict(
                server=server,
                shared_graphs=data["graphs"],
                jobs=len(seen),
                routed_rows=sum(e["routed_rows"] for e in data["records"]),
            )
        )
    # Generation1..12 are preparation. Compare only sealed episode intervals;
    # this is a host ownership overlap witness, not device execution timing.
    a = [i for i in intervals if i[0] == 0 and i[1] > 12]
    b = [i for i in intervals if i[0] == 1 and i[1] > 12]
    overlap = sum(any(max(i[3], j[3]) < min(i[4], j[4]) for j in b) for i in a)
    return dict(
        **result,
        clients=clients,
        servers=servers,
        client0_sealed_inflight_intervals_overlapping_client1=overlap,
        overlap_is_host_ownership_not_device_timing=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), indent=2))
