"""Summarize actual source co-batching; rolling records are not full-run rates."""

import argparse
import json
from pathlib import Path


def summarize(directory):
    clients = [
        json.loads(p.read_text()) for p in sorted(directory.glob("attention*.json"))
    ]
    owners = []
    for path in sorted(directory.glob("expert*.json")):
        result = json.loads(path.read_text())
        records = [r for r in result["trace"] if r[0] >= 0 and r[1] >= 0]
        paired = [r for r in records if r[0] and r[1]]
        assert all(r[7] == r[8] for r in paired), "cross-layer batch is invalid"
        counts = result["completed_counts"]
        leaders = [c for c in clients if c["calls"] is not None]
        assert counts == [
            next(c["calls"] for c in leaders if c["source"] == i) for i in range(2)
        ]
        owners.append(
            dict(
                owner=path.stem,
                calls_by_source=counts,
                server_waves=result["waves"],
                # Each task is admitted exactly once; a wave handles one or two
                # source descriptors. This full-run count survives the trace ring.
                paired_waves=sum(counts) - result["waves"],
                sampled_waves=len(records),
                sampled_paired_waves=len(paired),
                sampled_pairs_same_layer=True,
                rolling_trace=result["rolling_trace"],
                admitted_promotions=result["admitted_promotions"],
            )
        )
    assert len(clients) == 4 and len(owners) == 4
    assert all(c["status"] == "PASS" for c in clients)
    return dict(
        scope="two TP2 sources / E4; short correctness and batching observation, not throughput",
        clients=clients,
        owners=owners,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), indent=2))
