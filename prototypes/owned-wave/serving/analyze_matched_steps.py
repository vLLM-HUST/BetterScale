"""Compare ordered provider graph bodies in TraceLoom augmented DBs.

Not an exact replay partition: Ascend reconstruction in this frozen TraceLoom
is unsupported. Consecutive model-ID bodies must each have 48 FIA + one sampler.
Communication is clipped envelope time (includes waits), not link utilization.
"""

import argparse
import collections
import itertools
import json
import sqlite3
import statistics
from pathlib import Path


def category(name):
    if "FusedInferAttention" in name:
        return "attention"
    if "GroupedMatmul" in name:
        return "expert_matmul"
    if "MatMul" in name or "Matmul" in name:
        return "dense_matmul"
    if "Moe" in name or "Routing" in name:
        return "routing"
    if "RmsNorm" in name:
        return "norm"
    return "other_compute"


def union_length(intervals):
    total = 0
    end = None
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


def summarize(path, start, count, layers):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        records = db.execute("""select t.startNs,t.endNs,t.modelId,s.value from TASK t
            join COMPUTE_TASK_INFO i using(globalTaskId) join STRING_IDS s on s.id=i.opType
            where t.modelId not in (-1,4294967295) order by t.startNs""").fetchall()
        comm = db.execute("select startNs,endNs from COMMUNICATION_OP").fetchall()
        apis = dict(
            db.execute(
                "select api_name,count(*) from traceloom_runtime_call group by api_name"
            )
        )
    groups = [list(g) for _, g in itertools.groupby(records, key=lambda x: x[2])]
    selected = groups[start : start + count]
    if len(selected) != count:
        raise ValueError("incomplete graph body window")
    waves = []
    all_ops = collections.defaultdict(list)
    for group in selected:
        lo = min(r[0] for r in group)
        hi = max(r[1] for r in group)
        counts = collections.Counter()
        totals = collections.Counter()
        for a, b, _, name in group:
            cat = category(name)
            counts[cat] += 1
            totals[cat] += (b - a) / 1e6
            all_ops[name].append((b - a) / 1000)
        if counts["attention"] != layers or sum("ArgMax" in x[3] for x in group) != 1:
            raise ValueError("body is not one complete model+sampler invocation")
        clipped = [(max(a, lo), min(b, hi)) for a, b in comm if a < hi and b > lo]
        occupied = union_length([(a, b) for a, b, _, _ in group] + clipped) / 1e6
        waves.append(
            dict(
                model_id=group[0][2],
                start_ns=lo,
                end_ns=hi,
                envelope_ms=(hi - lo) / 1e6,
                compute_sums_ms=dict(totals),
                counts=dict(counts),
                collective_envelope_union_ms=union_length(clipped) / 1e6,
                compute_collective_union_ms=occupied,
                uncovered_ms=(hi - lo) / 1e6 - occupied,
            )
        )
    keys = waves[0]["compute_sums_ms"].keys()
    return dict(
        db=str(path.resolve()),
        body_indices=[start, start + count],
        waves=waves,
        mean=dict(
            envelope_ms=statistics.mean(w["envelope_ms"] for w in waves),
            compute_sums_ms={
                k: statistics.mean(w["compute_sums_ms"].get(k, 0) for w in waves)
                for k in keys
            },
            collective_envelope_union_ms=statistics.mean(
                w["collective_envelope_union_ms"] for w in waves
            ),
            uncovered_ms=statistics.mean(w["uncovered_ms"] for w in waves),
            between_body_gap_ms=statistics.mean(
                (b["start_ns"] - a["end_ns"]) / 1e6 for a, b in zip(waves, waves[1:])
            ),
        ),
        kernels={
            k: dict(count=len(v), mean_us=statistics.mean(v), total_ms=sum(v) / 1000)
            for k, v in all_ops.items()
        },
        whole_profile_host_calls={
            k: v
            for k, v in apis.items()
            if "FusedInferAttention" in k or "TaskUpdate" in k or "RIExecute" in k
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("old", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--old-start", type=int, default=40)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--layers", type=int, default=48)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    old = summarize(a.old, a.old_start, a.steps, a.layers)
    new = summarize(a.candidate, 0, a.steps, a.layers)
    result = dict(
        scope="four ordered decode bodies; same original first prompts and step positions; sampled token routes may differ; historical control not rerun",
        caveats=[
            "No supported exact TraceLoom Ascend replay partition; provider model-ID/ordered-body assertions used.",
            "Kernel sums can overlap; do not add them to collective envelopes as wall time.",
            "Uncovered time is not proof all device engines are idle.",
        ],
        old=old,
        candidate=new,
    )
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(old=old["mean"], candidate=new["mean"]), indent=2))
