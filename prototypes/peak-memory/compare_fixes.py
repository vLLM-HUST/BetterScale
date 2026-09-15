"""Compare the completed hw3 memory arms, not profiler timing or model quality."""

import argparse
import json
import re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("root", type=Path)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
arms = {}
cache_equivalents = {}
for arm, run in [
    ("control", "result"),
    ("candidate", "result-v2"),
    ("full", "result"),
    ("retained", "result"),
]:
    root = a.root / arm / run
    assert (root / "exit.txt").read_text().strip() == "0", f"{arm} not accepted"
    completion = json.loads((root / "engine" / "complete.json").read_text())
    assert completion == {"status": "PASS", "cohorts": 2}, completion
    cache_rows = re.findall(
        r"GPU KV cache size: ([0-9,]+) tokens",
        (root / "engine" / "server.log").read_text(),
    )
    assert len(cache_rows) == 1, (arm, cache_rows)
    cache_equivalents[arm] = int(cache_rows[0].replace(",", ""))
    ranks = []
    for rank in range(8):
        records = json.loads((root / "memory" / f"phases-rank-{rank}.json").read_text())

        def last(phase):
            return next(r for r in reversed(records) if r["phase"] == phase)

        trial = last("trial_complete_program")
        budget = last("budget")
        fitted = last("physical_budget_after_trial_release")
        ready = last("ready_after_capture")
        prepared = last("draft_prepare_after")
        ranks.append(
            dict(
                rank=rank,
                target_graph_bytes=trial["target_graph_bytes"],
                draft_extra_bytes=trial["draft_extra_bytes"],
                kv_budget_bytes=budget["kv_budget"],
                activation_bytes=budget["activation"],
                non_torch_bytes=budget["non_torch"],
                model_bytes=budget["model_bytes"],
                persistent_growth_bytes=fitted["persistent_growth"],
                safety_bytes=fitted["safety"],
                ready_allocated_bytes=ready["allocated"],
                ready_reserved_bytes=ready["reserved"],
                ready_peak_bytes=ready["peak"],
                ready_peak_reserved_bytes=ready["peak_reserved"],
                clear_high_water_increase_bytes=max(
                    0, ready["peak"] - prepared["peak"]
                ),
            )
        )
    arms[arm] = ranks

deltas = {}
for arm in ("candidate", "full", "retained"):
    deltas[arm] = [
        {key: row[key] - ref[key] for key in row if key != "rank"}
        | {"rank": row["rank"]}
        for row, ref in zip(arms[arm], arms["control"], strict=True)
    ]
result = dict(
    scope="hw3 TP8EP K5 dummy FULL24/4128, 512K ceiling; bytes per rank",
    reported_hybrid_token_equivalents_at_512k=cache_equivalents,
    quality_claim=False,
    throughput_claim=False,
    arms=arms,
    deltas=deltas,
)
a.output.write_text(json.dumps(result, indent=2) + "\n")
for arm, rows in arms.items():
    print(
        arm,
        {
            key: [
                round(min(r[key] for r in rows) / 2**20, 3),
                round(max(r[key] for r in rows) / 2**20, 3),
            ]
            for key in (
                "target_graph_bytes",
                "kv_budget_bytes",
                "ready_reserved_bytes",
                "clear_high_water_increase_bytes",
            )
        },
    )
