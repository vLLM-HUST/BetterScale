"""Report matched isolated/serial/overlap block timing without hiding slowdown."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

p = argparse.ArgumentParser()
p.add_argument("capsule", type=Path)
p.add_argument("--out", required=True, type=Path)
p.add_argument("--expect-cases", type=int, default=36)
a = p.parse_args()
files = sorted(a.capsule.glob("rank-*.json"))
assert len(files) == 2, files
rows = [r for f in files for r in json.loads(f.read_text())]
groups = defaultdict(list)
for r in rows:
    groups[
        (
            r["label"],
            r["m"],
            r["k"],
            r["n"],
            r["collective"],
            r["receive_mib"],
            r["rank"],
            r["mode"],
        )
    ].append(r)
assert all(len(v) == 3 for v in groups.values())
med = {
    key: {
        metric: (
            statistics.median(r[metric] for r in values if r[metric] is not None)
            if any(r[metric] is not None for r in values)
            else None
        )
        for metric in ["wave_ms", "mm_ms", "comm_ms"]
    }
    for key, values in groups.items()
}
cases = sorted(set(key[:6] for key in med))
assert len(cases) == a.expect_cases, len(cases)
results = []
for key in cases:
    by_rank = []
    for rank in [0, 1]:
        v = {
            mode: med[(*key, rank, mode)]
            for mode in ["compute", "comm", "serial", "overlap"]
        }
        by_rank.append(
            dict(
                rank=rank,
                values=v,
                mm_slowdown_pct=100
                * (v["overlap"]["mm_ms"] / v["compute"]["mm_ms"] - 1),
                comm_slowdown_pct=100
                * (v["overlap"]["comm_ms"] / v["comm"]["comm_ms"] - 1),
            )
        )
    serial = max(r["values"]["serial"]["wave_ms"] for r in by_rank)
    overlap = max(r["values"]["overlap"]["wave_ms"] for r in by_rank)
    item = dict(
        label=key[0],
        m=key[1],
        k=key[2],
        n=key[3],
        collective=key[4],
        receive_mib=key[5],
        serial_ms=serial,
        overlap_ms=overlap,
        net_reduction_pct=100 * (1 - overlap / serial),
        mm_slowdown_pct=[r["mm_slowdown_pct"] for r in by_rank],
        comm_slowdown_pct=[r["comm_slowdown_pct"] for r in by_rank],
        ranks=by_rank,
    )
    results.append(item)
    if key[4] == "allgather":
        print(
            f"{key[0]:18} M={key[1]:4} {key[5]:2}MiB serial={serial:.4f} overlap={overlap:.4f} net={item['net_reduction_pct']:+.1f}% mm={min(item['mm_slowdown_pct']):+.1f}..{max(item['mm_slowdown_pct']):+.1f}%"
        )
a.out.write_text(
    json.dumps(
        dict(
            capsule=str(a.capsule.resolve()),
            rows=len(rows),
            caveat="Two-card block-throughput experiment, 32 repeated operations normalized per operation. Not donor layer latency or TP8 bandwidth. Negative slowdown may be cache/scheduling variance.",
            results=results,
        ),
        indent=2,
    )
)
