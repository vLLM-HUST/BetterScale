"""Bound complete model bodies and metadata gaps in TraceLoom's native augmented DBs."""

import argparse
import json
from pathlib import Path
import sqlite3

from compare_full_prefill import inspect


def analyze(root, arm, rank):
    capsule = root / "round1" / arm
    db = capsule / f"native-graph/traceloom/{arm}-rank{rank}.db"
    schedule = json.loads((capsule / f"profiles/schedule-rank{rank}.json").read_text())
    dispatch = [e for e in schedule["events"] if e["event"] == "dispatch"]
    data = inspect(
        db,
        matmul_type=("MatMulV2", "MatMulV3"),
        matmul_count=304,
        synchronous=False,
        expected_steps=6,
    )
    assert len(dispatch) == len(data["steps"]) == 6
    data["dispatch"] = dispatch
    data["gaps"] = []
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        data["exact_graph_counts"] = c.execute(
            "select evidence_level,count(*) from traceloom_graph_launch group by evidence_level"
        ).fetchall()
        for step in data["steps"]:
            step["operator_sums_ms"] = c.execute(
                """select s.value,count(*),sum(t.endNs-t.startNs)/1e6 from TASK t
                join COMPUTE_TASK_INFO i using(globalTaskId) join STRING_IDS s on s.id=i.opType
                where t.startNs>=? and t.endNs<=? group by s.value order by 3 desc""",
                (step["start_ns"], step["end_ns"]),
            ).fetchall()
        for i, (before, after) in enumerate(zip(data["steps"], data["steps"][1:])):
            lo, hi = before["end_ns"], after["start_ns"]
            current, following = dispatch[i]["scheduled"], dispatch[i + 1]["scheduled"]
            steady = current == following and all(n == 1 for n in current)
            api = c.execute(
                """select s.value,count(*),sum(min(a.endNs,?)-max(a.startNs,?))/1e6
                from CANN_API a join STRING_IDS s on s.id=a.name
                where a.startNs<? and a.endNs>? group by s.value order by 3 desc""",
                (hi, lo, hi, lo),
            ).fetchall()
            sync = c.execute(
                """select s.value,(a.startNs-?)/1e6,(a.endNs-?)/1e6,a.globalTid
                from CANN_API a join STRING_IDS s on s.id=a.name
                where a.startNs<? and a.endNs>? and (s.value like '%Synchronize%'
                or s.value in ('aclrtMemcpy','aclmdlRIExecuteAsync')) order by a.startNs""",
                (lo, lo, hi, lo),
            ).fetchall()
            data["gaps"].append(
                dict(
                    after_step=i,
                    steady_decode=steady,
                    model_end_to_next_start_ms=(hi - lo) / 1e6,
                    sample_end_to_next_start_ms=(hi - lo) / 1e6
                    - before["model_end_to_sample_end_ms"],
                    api_sums_ms=api,
                    synchronization_relative_to_model_end_ms=sync,
                )
            )
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("capsule", type=Path)
    a = p.parse_args()
    root = a.capsule.resolve()
    output = dict(
        scope="Six diagnostic steps/rank after warmup; not HTTP timing. All bodies pass 304 MatMul,16 FIA,128 communication guards. Operator/API sums are not exclusive critical-path time. Rank-local clocks; profile overhead can amplify host submission gaps. Exact graph reconstruction can omit unique/under-repeated shapes.",
        arms={},
    )
    for arm in ("baseline", "candidate"):
        output["arms"][arm] = [analyze(root, arm, rank) for rank in (0, 1)]
    assert [e["scheduled"] for e in output["arms"]["baseline"][0]["dispatch"]] == [
        e["scheduled"] for e in output["arms"]["candidate"][0]["dispatch"]
    ]
    (root / "profile-analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    for arm, ranks in output["arms"].items():
        for rank, data in enumerate(ranks):
            print(
                arm,
                rank,
                "bodies_ms",
                [round(s["body_ms"], 3) for s in data["steps"]],
                "steady_gaps_ms",
                [
                    round(g["model_end_to_next_start_ms"], 3)
                    for g in data["gaps"]
                    if g["steady_decode"]
                ],
            )


if __name__ == "__main__":
    main()
