"""Whole-session throughput/latency; retain paired or candidate-only rounds."""

import argparse
import json
from pathlib import Path
import statistics


def percentile(values, fraction):
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def metrics(cohorts):
    rows = [r for c in cohorts for r in c["requests"]]
    assert all(r["usage"]["prompt_tokens"] == r["prompt_tokens"] for r in rows)
    ttft = [r["ttft_s"] * 1000 for r in rows]
    latency = [r["latency_s"] for r in rows]
    tpot = [
        (r["latency_s"] - r["ttft_s"]) * 1000 / (r["output_tokens"] - 1)
        for r in rows
        if r["output_tokens"] > 1
    ]
    result = dict(
        requests=len(rows),
        output_tokens=sum(r["output_tokens"] for r in rows),
        elapsed_s=sum(c["elapsed_s"] for c in cohorts),
        tokens_per_s=sum(r["output_tokens"] for r in rows)
        / sum(c["elapsed_s"] for c in cohorts),
        mean_ttft_ms=statistics.mean(ttft),
        p95_ttft_ms=percentile(ttft, 0.95),
        mean_latency_s=statistics.mean(latency),
        p95_latency_s=percentile(latency, 0.95),
        mean_tpot_ms=statistics.mean(tpot),
        p95_tpot_ms=percentile(tpot, 0.95),
    )
    if all(r["usage"].get("prompt_tokens_details") is not None for r in rows):
        cached = sum(r["usage"]["prompt_tokens_details"]["cached_tokens"] for r in rows)
        prompt = sum(r["prompt_tokens"] for r in rows)
        result.update(
            prompt_tokens=prompt,
            cached_prompt_tokens=cached,
            cached_prompt_fraction=cached / prompt,
        )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("capsule", type=Path)
    a = p.parse_args()
    root = a.capsule.resolve()
    trace = json.loads((root / "trace.json").read_text())
    comparison = json.loads((root / "comparison.json").read_text())
    assert comparison["status"] == "PASS"
    expected = {
        (i, t): (len(c["prompt_ids"]), c["output_tokens"])
        for i, s in enumerate(trace["sessions"])
        for t, c in enumerate(s["calls"])
    }
    arms = sorted({r["arm"] for r in comparison["rounds"]})
    assert arms in (["candidate"], ["baseline", "candidate"])
    first = json.loads((root / "round0" / arms[0] / "receipt.json").read_text())
    apc = first.get("prefix_caching", False)
    concurrencies = sorted(c["concurrency"] for c in first["rounds"])
    assert concurrencies and len(concurrencies) == len(set(concurrencies))
    cohorts = {arm: {c: [] for c in concurrencies} for arm in arms}
    rounds = []
    for repeat in range(2):
        for arm in cohorts:
            d = json.loads((root / f"round{repeat}" / arm / "receipt.json").read_text())
            assert d["status"] == "PASS"
            assert d.get("prefix_caching", False) == apc
            if apc:
                assert d["cache_start"] == "empty before each cohort"
            assert sorted(c["concurrency"] for c in d["rounds"]) == concurrencies
            for c in d["rounds"]:
                assert len(c["requests"]) == len(expected)
                assert {
                    (r["session"], r["turn"]): (r["prompt_tokens"], r["output_tokens"])
                    for r in c["requests"]
                } == expected
                cohorts[arm][c["concurrency"]].append(c)
                rounds.append(
                    dict(
                        repeat=repeat,
                        arm=arm,
                        devices=d["devices"],
                        concurrency=c["concurrency"],
                        **metrics([c]),
                    )
                )
    pooled = {
        arm: {c: metrics(rows) for c, rows in cases.items()}
        for arm, cases in cohorts.items()
    }
    output = dict(
        scope=comparison["scope"],
        concurrencies=concurrencies,
        prefix_caching=apc,
        plan=(
            json.loads((root / "plan.json").read_text())
            if (root / "plan.json").exists()
            else None
        ),
        fixture=dict(
            dataset=trace["dataset"],
            revision=trace["revision"],
            subset=trace["subset"],
            scanned=trace["scanned"],
            selection=trace["selection"],
            transforms=trace["transforms"],
            trajectories=[s["trajectory_id"] for s in trace["sessions"]],
        ),
        rounds=rounds,
        pooled=pooled,
        throughput_gain_percent={
            c: 100
            * (
                pooled["candidate"][c]["tokens_per_s"]
                / pooled["baseline"][c]["tokens_per_s"]
                - 1
            )
            for c in concurrencies
            if "baseline" in pooled
        },
        limits="Eight selected <=8K trajectories, APC setting recorded separately, no MTP, fixed output budgets, no tool latency. Two repeats per arm (ordering/arms in scope), not population or confidence evidence. No fresh baseline comparison is implied for candidate-only runs. Profile excluded. TPOT is HTTP completion-minus-first-content per remaining output token, not SSE event gaps.",
    )
    (root / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                pooled=pooled, throughput_gain_percent=output["throughput_gain_percent"]
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
