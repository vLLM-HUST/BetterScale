"""Reduce complete unprofiled HTTP cohorts, keeping repeats and workload visible."""

import argparse
import json
from pathlib import Path
import re
from statistics import median


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def counters(path):
    result = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([^ {]+)(?:\{.*\})?\s+([-+\deE.]+)", line)
        if match:
            name, value = match.groups()
            if (
                "spec_decode" in name
                or "iteration_tokens" in name
                or "generation_tokens_total" in name
                or "preemptions_total" in name
            ):
                if name.endswith(("_total", "_count", "_sum")):
                    result[name] = result.get(name, 0) + float(value)
    return result


def summarize(capsule):
    assert (capsule / "exit.txt").read_text().strip() == "0"
    root = capsule / "engine"
    assert json.loads((root / "complete.json").read_text())["status"] == "PASS"
    result = []
    for cohort in json.loads((root / "cohorts.json").read_text()):
        requests = cohort["requests"]
        gaps = [
            b - a
            for r in requests
            for a, b in zip(r["chunk_arrivals_s"], r["chunk_arrivals_s"][1:])
        ]
        label = cohort["label"]
        before = counters(root / f"{label}-metrics-before.txt")
        after = counters(root / f"{label}-metrics-after.txt")
        result.append(
            dict(
                label=label,
                requests=len(requests),
                input_tokens=sum(r["input_tokens"] for r in requests),
                output_tokens=sum(r["output_tokens"] for r in requests),
                elapsed_s=cohort["elapsed_s"],
                output_tps=cohort["output_tps"],
                ttft_p50_s=median(r["ttft_s"] for r in requests),
                ttft_p95_s=percentile([r["ttft_s"] for r in requests], 0.95),
                completion_p95_s=percentile([r["elapsed_s"] for r in requests], 0.95),
                chunk_gap_p99_s=percentile(gaps, 0.99),
                native_counter_delta={
                    k: v - before.get(k, 0) for k, v in after.items()
                },
            )
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("capsules", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps({str(p): summarize(p) for p in args.capsules}, indent=2)
    )
