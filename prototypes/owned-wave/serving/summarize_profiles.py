"""Bounded native-provider observations; windows are NOT matched neural work."""

import argparse
import json
import sqlite3
from pathlib import Path


def summarize(entry):
    with sqlite3.connect(f"file:{entry['db']}?mode=ro", uri=True) as db:
        apis = db.execute("""select s.value,count(*),sum(a.endNs-a.startNs)/1e6
            from CANN_API a join STRING_IDS s on s.id=a.name
            group by s.value order by sum(a.endNs-a.startNs) desc""").fetchall()
        sampling = db.execute("""select s.value,t.modelId,count(*) from TASK t
            join COMPUTE_TASK_INFO i using(globalTaskId)
            join STRING_IDS s on s.id=i.opType
            where lower(s.value) like '%argmax%'
            group by s.value,t.modelId""").fetchall()
        intervals = db.execute("""select t.startNs,t.endNs from TASK t
            join COMPUTE_TASK_INFO i using(globalTaskId)
            union all select startNs,endNs from COMMUNICATION_OP""").fetchall()
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    gaps = [b[0] - a[1] for a, b in zip(merged, merged[1:])]
    return dict(
        arm=entry["arm"],
        rank=entry["rank"],
        db=entry["db"],
        timeline=entry["timeline"],
        apis=[dict(name=n, calls=c, summed_host_ms=t) for n, c, t in apis],
        sampler_tasks=[dict(op=n, model_id=m, count=c) for n, m, c in sampling],
        compute_comm_envelope_ms=(merged[-1][1] - merged[0][0]) / 1e6,
        compute_comm_union_ms=sum(b - a for a, b in merged) / 1e6,
        uncovered_gaps_over_50us=sum(g > 50000 for g in gaps),
        uncovered_gaps_over_50us_ms=sum(g for g in gaps if g > 50000) / 1e6,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("capsule", type=Path)
    args = parser.parse_args()
    entries = json.loads((args.capsule / "profile-exports.json").read_text())
    result = dict(
        scope="first64 waves per arm; unequal prompt/batch work; not speedup attribution",
        caveats=[
            "Host API durations may nest/overlap and are not additive wall time.",
            "Synchronization duration can include productive queued device work.",
            "Compute+communication coverage omits other engines; gaps are not proven idle time.",
            "Provider timestamps are used within rank only; no cross-rank clock alignment.",
        ],
        ranks=[summarize(entry) for entry in entries],
    )
    (args.capsule / "profile-summary.json").write_text(json.dumps(result, indent=2))
    for row in result["ranks"]:
        print(
            row["arm"],
            row["rank"],
            "sampler tasks",
            sum(x["count"] for x in row["sampler_tasks"]),
        )
