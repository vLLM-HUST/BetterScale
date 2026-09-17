"""Summarize completed fit cases without confusing head shards with capacity."""

import argparse
import json
from pathlib import Path

from analyze_topology_matrix import TOPOLOGIES


def summarize(directory):
    directory = Path(directory)
    parameters = json.loads((directory / "parameters.json").read_text())
    tp, sources = TOPOLOGIES[parameters["layout"]]
    if parameters["mode"] != "capacity":
        raise ValueError("not a capacity case")
    capsule = Path((directory / "capsule").read_text().strip())
    result = dict(case=str(directory), capsule=str(capsule), **parameters)
    result["exit"] = int((directory / "exit").read_text())
    if result["exit"]:
        result["status"] = "FAILED; inspect capsule role logs, not necessarily OOM"
        return result
    records = [
        json.loads((capsule / "roles" / f"attention{i}.json").read_text())
        for i in range(tp * sources)
    ]
    for r in records:
        if r["status"] != "PASS" or (r["tp_size"], r["sources"]) != (tp, sources):
            raise ValueError("inconsistent role qualification")
    keys = (
        "history_tokens_per_group",
        "allocated_history_tokens_whole_machine",
        "pressure_prefix_per_request",
        "exercised_history_tokens_whole_machine",
        "requests_per_source",
        "model_max_context",
    )
    for key in keys:
        if len({r[key] for r in records}) != 1:
            raise ValueError(f"nonuniform fit geometry: {key}")
        result[key] = records[0][key]
    result.update(
        status="PASS",
        tp_size=tp,
        sources=sources,
        minimum_sampled_driver_free_bytes=min(
            r["minimum_sampled_driver_free_bytes"] for r in records
        ),
        maximum_allocator_peak_bytes=max(
            s["peak_allocated"] for r in records for s in r["samples"]
        ),
        maximum_allocator_reserved_peak_bytes=max(
            s["peak_reserved"] for r in records for s in r["samples"]
        ),
    )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cases", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    report = dict(
        scope="real full48+MTP K1, all State touched, zero synthetic history; not language quality or actual long-history throughput",
        bound="passed tested fit, not exhaustive maximum or production-safe free margin",
        results=[summarize(case) for case in args.cases],
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
