"""Compare a fresh candidate to explicitly reused, unchanged native receipts."""

import argparse
import json
import statistics
from pathlib import Path
from report import stats


def read(path):
    return json.loads(path.read_text())


def compare_candidate(candidate, baselines):
    engine = candidate / "engine"
    config = read(engine / "config.json")
    trace = read(candidate / "trace.json")
    complete = read(engine / "complete.json")
    if (
        complete["profiled"]
        or complete["status"] != "PASS"
        or (candidate / "timing-exclusion.json").exists()
    ):
        raise ValueError("candidate timing is not eligible")
    owned = [
        read(engine / f"candidate-rank{rank}.json")
        for rank in range(config["tensor_parallel_size"])
    ]
    if not all(x["status"] == "PASS" and not x["runner_calls"] for x in owned):
        raise ValueError("candidate protocol failed")
    for rank in owned:
        attention = rank.get("static_fia") or {}
        if attention.get("protocol") == "native-host-wave":
            waves = sum(row["waves"] for row in rank["rounds"])
            if (
                attention["wave_plans"] != waves
                or sum(attention["dispatches"].values()) != waves
            ):
                raise ValueError(
                    "native attention metadata was not planned once per wave"
                )
    expected = {
        (s, t): (len(call["prompt_ids"]), call["output_tokens"])
        for s, session in enumerate(trace["sessions"])
        for t, call in enumerate(session["calls"])
    }

    def index(row):
        result = {(x["session"], x["turn"]): x for x in row["records"]}
        if len(result) != len(row["records"]) or result.keys() != expected.keys():
            raise ValueError("request population differs")
        if any(
            (v["prompt_tokens"], v["output_tokens"]) != expected[k]
            for k, v in result.items()
        ):
            raise ValueError("request work differs")
        if row["output_tokens"] != sum(v[1] for v in expected.values()):
            raise ValueError("output total differs")
        return result

    native = {"cold": [], "retained": []}
    sources = []
    for baseline in baselines:
        if (
            read(baseline / "engine/config.json") != config
            or read(baseline / "trace.json") != trace
        ):
            raise ValueError(f"baseline configuration/trace differs: {baseline}")
        data = read(baseline / "engine/native.json")
        if data["profiled"] or (baseline / "timing-exclusion.json").exists():
            raise ValueError(f"baseline timing excluded: {baseline}")
        if (baseline / "run/exit.txt").read_text().strip() != "0":
            raise ValueError(f"baseline supervisor did not pass: {baseline}")
        for row in data["rounds"]:
            index(row)
            native["cold" if row["repeat"] == 0 else "retained"].append(row)
        sources.append(str(baseline.resolve()))
    rows = []
    for round_rows in zip(*(x["rounds"] for x in owned), strict=True):
        row = round_rows[0]
        records = index(row)
        for peer in round_rows[1:]:
            other = index(peer)
            if any(records[k]["token_ids"] != other[k]["token_ids"] for k in records):
                raise ValueError("TP disagreement")
            if (row["waves"], row["hit_tokens"]) != (peer["waves"], peer["hit_tokens"]):
                raise ValueError("TP work differs")
        cache = "cold" if row["repeat"] == 0 else "retained"
        old = native[cache]
        if not old:
            raise ValueError(f"no {cache} baseline")
        mean = statistics.mean(r["elapsed_s"] for r in old)
        rows.append(
            dict(
                repeat=row["repeat"],
                cache=cache,
                candidate=stats(row),
                reused_native_samples=len(old),
                reused_native_mean_s=mean,
                elapsed_reduction=1 - row["elapsed_s"] / mean,
                native_hit_tokens=[r["hit_tokens"] for r in old],
            )
        )
    return dict(
        status="PASS",
        candidate=str(candidate.resolve()),
        baselines=sources,
        scope="historical native baselines reused without rerun; same config/trace; not a fresh contemporaneous A/B",
        model=config["model"],
        rounds=rows,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("candidate", type=Path)
    p.add_argument("--baseline", type=Path, action="append", required=True)
    a = p.parse_args()
    result = compare_candidate(a.candidate, a.baseline)
    (a.candidate / "reused-baseline-comparison.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result, indent=2))
