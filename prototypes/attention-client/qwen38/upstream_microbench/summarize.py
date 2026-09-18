"""Condense accepted matched leaf capsules, refusing incomplete/failed runs."""

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("capsule", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
cases = []
for arm in (
    "indexer",
    "attention",
    "ascend-attention",
    "prefill-final",
    "hc-final",
    "expansion-final",
):
    directory = a.capsule / arm
    assert (directory / "exit.txt").read_text().strip() == "0", directory
    records = [
        json.loads(line)
        for line in (directory / "run.log").read_text().splitlines()
        if line.startswith("{")
    ]
    assert any("timings" in record for record in records), directory
    for record in records:
        if "timings" not in record:
            continue
        item = {
            k: v
            for k, v in record.items()
            if k not in ("errors", "differences", "timings")
        }
        item["capsule"] = arm
        item["timings"] = {
            name: {k: v for k, v in metrics.items() if k != "samples_ms"}
            for name, metrics in record["timings"].items()
        }
        for field in ("errors", "differences"):
            if field in record:
                groups = {}
                for x in record[field]:
                    name = x.get("arm", "combined")
                    group = groups.setdefault(
                        name, dict(comparisons=0, max_relative_l2=0.0)
                    )
                    group["comparisons"] += 1
                    group["max_relative_l2"] = max(
                        group["max_relative_l2"], x["relative_l2"]
                    )
                    if "bitwise_equal" in x:
                        group["non_bitwise_comparisons"] = group.get(
                            "non_bitwise_comparisons", 0
                        ) + int(not x["bitwise_equal"])
                item[field] = groups
        cases.append(item)
result = dict(
    date="2026-09-18",
    host="hw0",
    device="Ascend 910B2",
    torch="2.10.0+cpu",
    torch_npu="2.10.0.post2",
    cann="9.0.1",
    scope="Dummy matched single-device graph leaf probes, not whole-model or serving gains",
    timing="7 alternating-order rounds, 3 replays per sample, NPU-event median",
    memory="Incremental peak allocated bytes during capture, not reserved-pool size or complete model peak",
    sources=json.loads((a.capsule / "manifest.json").read_text()),
    current_overlay="/workspace/betterscale-hw0/repo/runs/qwen38-bounded-qsa-runtime-20260917e/overlay",
    selected_candidate="BorrowedHCLeaves: norm/mix only, <=32 rows, no production default change",
    cases=cases,
)
a.output.write_text(json.dumps(result, indent=2) + "\n")
print(len(cases), "matched cases")
