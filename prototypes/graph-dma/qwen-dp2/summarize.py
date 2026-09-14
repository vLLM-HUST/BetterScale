"""Preserve both-rank trials and variability; don't call a median a guarantee."""

import argparse
import json
from pathlib import Path
from statistics import median

p = argparse.ArgumentParser()
p.add_argument("root", type=Path)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
assert (a.root / "complete.json").exists()
rows = []
for rank in (0, 1):
    assert json.loads((a.root / f"passed-rank{rank}.json").read_text())["kv_exact"]
    rs = [
        json.loads(line)
        for line in (a.root / f"records-rank{rank}.jsonl").read_text().splitlines()
    ]
    rows.extend(rs)
keys = sorted(
    {(r["direction"], r["mib"], r.get("copies", 1), r["active"]) for r in rows}
)
cases = []
for direction, mib, copies, active in keys:
    selected = [
        r
        for r in rows
        if (r["direction"], r["mib"], r.get("copies", 1), r["active"])
        == (direction, mib, copies, active)
    ]

    def values(mode, column):
        return [
            max(
                r[column] for r in selected if r["trial"] == trial and r["mode"] == mode
            )
            for trial in range(3)
        ]

    def summary(v):
        return dict(min=min(v), median=median(v), max=max(v), trials=v)

    baseline = values("compute", "compute_ms")
    comp = values("overlap", "compute_ms")
    byrank = []
    for rank in (0, 1):
        local = [r for r in selected if r["rank"] == rank]
        byrank.append(
            dict(
                rank=rank,
                copy_overlap_ms=summary(
                    [r["copy_ms"] for r in local if r["mode"] == "overlap"]
                ),
                all_outputs_exact=all(r["output_exact"] for r in local),
            )
        )
    cases.append(
        dict(
            direction=direction,
            mib=mib,
            copies=copies,
            active=active,
            bytes_per_active_rank=mib * 1024**2 * copies,
            baseline_ms=summary(baseline),
            overlap_compute_ms=summary(comp),
            compute_slowdown_median=median(comp) / median(baseline) - 1,
            serial_joint_ms=summary(values("serial", "total_ms")),
            overlap_joint_ms=summary(values("overlap", "total_ms")),
            copy_alone_maxrank_ms=summary(values("copy", "copy_ms")),
            rank_details=byrank,
        )
    )
a.out.write_text(
    json.dumps(
        dict(
            scope="Qwen3-30B-A3B real BF16 DP2TP1EP2; one4096-token request per rank; native FULL forward+parameter update; max rank event durations, three alternating trials",
            record_count=len(rows),
            all_outputs_exact=all(r["output_exact"] for r in rows),
            all_kv_exact=True,
            cases=cases,
        ),
        indent=2,
    )
    + "\n"
)
for c in cases:
    print(
        c["direction"],
        c["mib"],
        c["copies"],
        c["active"],
        round(c["baseline_ms"]["median"], 2),
        round(c["overlap_joint_ms"]["median"], 2),
        round(100 * c["compute_slowdown_median"], 2),
        c["overlap_joint_ms"]["trials"],
    )
