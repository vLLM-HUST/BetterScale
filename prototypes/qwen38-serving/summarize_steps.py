"""Compare actual dispatched shapes, never HTTP concurrency labels alone."""

import argparse
import json
from pathlib import Path
import statistics


METRICS = ("period_ms", "forward_ms", "prepare_ms", "after_forward_ms")


def selected(row, following, case):
    if following is None or "period_ms" not in row:
        return False
    scheduled, computed = row["scheduled"], row["computed"]
    n, b = case["context"], case["batch"]
    if case["kind"] == "decode":
        # Discard ramp-up and tail; match batch and a common advancing KV window.
        return (scheduled == [1] * b and following["scheduled"] == [1] * b
                and row["prompt_tokens"] == [n] * b
                and all(n + 8 <= x < n + 40 for x in computed))
    if case["kind"] == "prefill":
        return scheduled == [n] and computed == [case.get("prefix", 0)]
    return (sorted(scheduled) == [1] * b + [case["joining"]]
            and all(c == 0 if q > 1 else n <= c < n + 16
                    for q, c in zip(scheduled, computed)))


def cohort_rows(cohort):
    ranks = sorted(cohort["measurement"]["results"], key=lambda r: r["rank"])
    assert [r["rank"] for r in ranks] == [0, 1]
    assert len(ranks[0]["steps"]) == len(ranks[1]["steps"])
    for a, b in zip(ranks[0]["steps"], ranks[1]["steps"]):
        assert (a["scheduled"], a["computed"]) == (b["scheduled"], b["computed"])
    chosen = []
    for rank in ranks:
        rows = rank["steps"]
        for row in rows:
            assert row["forward_ms"] > 0 and row["prepare_ms"] >= 0
            if "period_ms" in row:
                assert row["after_forward_ms"] >= 0
                assert abs(row["period_ms"] - sum(row[m] for m in METRICS[1:])) < 0.01
        selected_rows = [row for i, row in enumerate(rows)
                         if selected(row, rows[i + 1] if i + 1 < len(rows) else None, cohort["case"])]
        chosen.append(dict(rank=rank["rank"], rows=selected_rows))
    assert len(chosen[0]["rows"]) == len(chosen[1]["rows"])
    return chosen


def summarize(root):
    points, missing = {}, []
    for folder in sorted(root.glob("[0-3]-*")):
        receipt = json.loads((folder / "receipt.json").read_text())
        assert receipt["status"] == "PASS", folder
        cohorts = []
        for original in receipt["cohorts"]:
            if original["warmup"]:
                continue
            case = original["case"]
            if case["kind"] == "prefill" and case["context"] == 2048:
                # Native align-mode APC splits this prompt at its 1536-token
                # state checkpoint. Expose actual work, not a fictitious 2048 step.
                steps = original["measurement"]["results"][0]["steps"]
                chunks = [(r["scheduled"], r["computed"]) for r in steps if max(r["scheduled"]) > 1]
                assert chunks == [([1536], [0]), ([512], [1536])], chunks
                for n, prefix in ((1536, 0), (512, 1536)):
                    cohorts.append({**original, "case": dict(kind="prefill", batch=1,
                                    context=n, prefix=prefix, source_prompt=2048)})
            else:
                cohorts.append(original)
        for cohort in cohorts:
            if cohort["warmup"]:
                continue
            case = cohort["case"]
            key = json.dumps(case, sort_keys=True)
            point = points.setdefault(key, dict(case=case, arms={}))
            ranks = cohort_rows(cohort)
            if not ranks[0]["rows"]:
                missing.append(dict(round=folder.name, label=cohort["label"], case=case))
                continue
            # One independent cohort summary, not two independent TP samples.
            # Keep the slower rank's mean cycle, with its additive components.
            rank = max(ranks, key=lambda r: statistics.mean(x["period_ms"] for x in r["rows"]))
            rows = rank["rows"]
            item = dict(round=folder.name, label=cohort["label"], rank=rank["rank"], count=len(rows),
                        metrics={m: statistics.mean(row[m] for row in rows) for m in METRICS},
                        computed_min=min(min(row["computed"]) for row in rows),
                        computed_max=max(max(row["computed"]) for row in rows),
                        modes=sorted(set(row["mode"] for row in rows)),
                        descriptors=sorted(set(row["descriptor"] for row in rows)),
                        samples_ms=[row["period_ms"] for row in rows],
                        between_forward_samples_ms=[
                            row["after_forward_ms"] + rows[j + 1]["prepare_ms"]
                            for j, row in enumerate(rows[:-1])
                            if rows[j + 1]["index"] == row["index"] + 1
                        ])
            point["arms"].setdefault(receipt["arm"], []).append(item)
    for point in points.values():
        point["summary"] = {}
        for arm, cohorts in point["arms"].items():
            point["summary"][arm] = dict(
                cohorts=len(cohorts), steps=sum(c["count"] for c in cohorts),
                **{m: dict(mean=statistics.mean(c["metrics"][m] for c in cohorts),
                           low=min(c["metrics"][m] for c in cohorts),
                           high=max(c["metrics"][m] for c in cohorts)) for m in METRICS})
        if set(point["summary"]) == {"baseline", "candidate"}:
            before = point["summary"]["baseline"]["period_ms"]["mean"]
            after = point["summary"]["candidate"]["period_ms"]["mean"]
            point["period_reduction_pct"] = (1 - after / before) * 100
            point["step_rate_gain_pct"] = (before / after - 1) * 100
    return dict(
        scope="Same hw3 6/7 ABBA, two measured cohorts after one shape warmup pass per process. APC enabled but cold shape cohorts, no MTP, AIV both. Slower TP rank's per-cohort mean; equally weight cohorts. Ranges are observed cohort extrema, not confidence intervals.",
        metric="External device execute-start to next execute-start. Forward envelope includes host submission/update waits; prepare/after-forward are not pure CPU overhead or device-idle time.",
        missing_cohorts=missing, points=list(points.values()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("capsule", type=Path)
    a = p.parse_args()
    result = summarize(a.capsule)
    (a.capsule / "step-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("points", len(result["points"]), "missing cohorts", len(result["missing_cohorts"]))
    for point in result["points"]:
        print(point["case"], {k: round(v["period_ms"]["mean"], 3) for k, v in point["summary"].items()},
              "gain%", round(point.get("step_rate_gain_pct", 0), 2))
