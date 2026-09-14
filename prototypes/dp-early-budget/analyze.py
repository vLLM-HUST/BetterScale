"""Audit host-side budget lead; never label these timestamps device execution."""

import argparse
import json
from pathlib import Path
from statistics import median


def read(path, event):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    selected = [r for r in rows if r.get("event") == event]
    keyed = {r["sequence"]: r for r in selected}
    assert len(keyed) == len(selected), (path, event, "duplicate sequence")
    return keyed


def analyze(root, ranks):
    rank_rows = []
    for rank in range(ranks):
        parent = read(root / f"parent-rank{rank}.jsonl", "agreed")
        worker = root / f"worker-rank{rank}.jsonl"
        entered = read(worker, "enter")
        consumed = read(worker, "consume")
        forward = read(worker, "forward_enter")
        admitted = {seq for seq, row in parent.items() if row["admitted"]}
        assert admitted <= entered.keys() & consumed.keys() & forward.keys(), (
            rank,
            "incomplete admitted wave",
        )
        for seq in sorted(admitted):
            p, e, c, f = parent[seq], entered[seq], consumed[seq], forward[seq]
            assert (
                p["started_ns"]
                <= p["time_ns"]
                <= e["time_ns"]
                <= c["time_ns"]
                <= f["time_ns"]
            )
            rank_rows.append(
                dict(
                    rank=rank,
                    sequence=seq,
                    tokens=p["tokens"],
                    agreement_ms=(p["time_ns"] - p["started_ns"]) / 1e6,
                    worker_queue_lead_ms=(e["time_ns"] - p["time_ns"]) / 1e6,
                    forward_lead_ms=(f["time_ns"] - p["time_ns"]) / 1e6,
                )
            )
    by_sequence = {}
    for row in rank_rows:
        by_sequence.setdefault(row["sequence"], []).append(row)
    for seq, rows in by_sequence.items():
        assert len(rows) == ranks, (seq, "group admission differs")
        assert len({tuple(r["tokens"]) for r in rows}) == 1
    return dict(
        scope="Host epoch clocks on one host; NOT device launch/execution or a throughput measurement",
        admitted_waves=len(by_sequence),
        ranks=ranks,
        median_agreement_ms=(
            median(r["agreement_ms"] for r in rank_rows) if rank_rows else None
        ),
        median_worker_queue_lead_ms=(
            median(r["worker_queue_lead_ms"] for r in rank_rows) if rank_rows else None
        ),
        median_forward_lead_ms=(
            median(r["forward_lead_ms"] for r in rank_rows) if rank_rows else None
        ),
        rows=rank_rows,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--ranks", type=int, default=8)
    a = p.parse_args()
    result = analyze(a.root, a.ranks)
    (a.root / "early-budget-analysis.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
