"""Unprofiled ABBA service statistics; preserve round spread and workload identity."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics as st

p = argparse.ArgumentParser()
p.add_argument("capsule", type=Path)
a = p.parse_args()
manifest = json.loads((a.capsule / "comparison.json").read_text())
assert manifest["status"] == "PASS" and len(manifest["rounds"]) == 4
arms = defaultdict(lambda: defaultdict(list))
texts = defaultdict(lambda: defaultdict(list))
for round in manifest["rounds"]:
    d = json.loads(Path(round["path"]).read_text())
    assert d["status"] == "PASS" and "--speculative-config" not in d["command"]
    assert d["comparison"] == round["arm"]
    for c in d["cohorts"]:
        key = f"C{c['concurrency']}-" + (
            str(c["single_length"]) if c["single_length"] else "mixed"
        )
        arms[key][round["arm"]].append((round["index"], c))
        if c["concurrency"] == 1:
            texts[key][round["arm"]].extend(x["text"] for x in c["requests"])
summary = []
for case, groups in arms.items():
    result = dict(case=case)
    for arm, pairs in groups.items():
        cohorts = [c for _, c in pairs]
        rows = [x for c in cohorts for x in c["requests"]]
        round_rates = []
        for index in sorted({i for i, _ in pairs}):
            subset = [c for i, c in pairs if i == index]
            round_rates.append(
                sum(sum(x["output_tokens"] for x in c["requests"]) for c in subset)
                / sum(c["elapsed_s"] for c in subset)
            )
        result[arm] = dict(
            requests=len(rows),
            output_tokens=sum(x["output_tokens"] for x in rows),
            throughput_tokens_s=sum(x["output_tokens"] for x in rows)
            / sum(c["elapsed_s"] for c in cohorts),
            mean_ttft_ms=st.mean(x["ttft_s"] * 1000 for x in rows),
            mean_e2e_ms=st.mean(x["latency_s"] * 1000 for x in rows),
            mean_tpot_ms=st.mean(
                (x["latency_s"] - x["ttft_s"]) * 1000 / (x["output_tokens"] - 1)
                for x in rows
            ),
            round_throughput_tokens_s=round_rates,
        )
    result["throughput_change_pct"] = (
        result["candidate"]["throughput_tokens_s"]
        / result["baseline"]["throughput_tokens_s"]
        - 1
    ) * 100
    result["ttft_reduction_pct"] = (
        1 - result["candidate"]["mean_ttft_ms"] / result["baseline"]["mean_ttft_ms"]
    ) * 100
    if case in texts:
        result["c1_text_sets_match"] = set(texts[case]["baseline"]) == set(
            texts[case]["candidate"]
        )
    summary.append(result)
out = dict(
    status="PASS",
    scope=manifest["scope"],
    rows=summary,
    limits="Four cohorts per case per arm; round ranges are not confidence intervals. Fixed synthetic prompts, 64 output tokens, APC off. TPOT is (HTTP completion time - first content time)/63, not pure device kernel time.",
)
(a.capsule / "summary.json").write_text(json.dumps(out, indent=2))
for r in summary:
    print(
        r["case"],
        "tok/s",
        *[round(r[x]["throughput_tokens_s"], 2) for x in ["baseline", "candidate"]],
        "change%",
        round(r["throughput_change_pct"], 2),
        "TTFT ms",
        *[round(r[x]["mean_ttft_ms"], 2) for x in ["baseline", "candidate"]],
    )
