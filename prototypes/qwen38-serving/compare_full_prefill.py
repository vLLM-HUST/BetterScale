"""Compare paired historical prefill spans; never equate host waits with GPU idle."""

import argparse
import collections
import json
from pathlib import Path
import sqlite3


def union(rows):
    result = []
    for lo, hi in sorted(rows):
        if result and lo <= result[-1][1]:
            result[-1] = (result[-1][0], max(hi, result[-1][1]))
        else:
            result.append((lo, hi))
    return result


def duration(rows):
    return sum(hi - lo for lo, hi in union(rows)) / 1e6


def inspect(
    path, matmul_type="MatMulV3", matmul_count=256, synchronous=True, expected_steps=4
):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = c.execute(
        """select t.startNs,t.endNs,s.value,t.modelId,t.globalTaskId from TASK t
        join COMPUTE_TASK_INFO i using(globalTaskId) join STRING_IDS s on s.id=i.opType order by t.startNs"""
    ).fetchall()
    samples = [r for r in rows if r[2] == "ArgMaxV2"]
    assert len(samples) == expected_steps, (len(samples), expected_steps)
    firsts = [r for r in rows if r[2] == "GemmaRmsNorm"]
    assert len(firsts) == expected_steps, (len(firsts), expected_steps)
    all_comm = c.execute(
        "select startNs,endNs from COMMUNICATION_OP order by startNs"
    ).fetchall()
    apis = c.execute("""select a.startNs,a.endNs,s.value,a.globalTid from CANN_API a
        join STRING_IDS s on s.id=a.name order by a.startNs""").fetchall()
    result = []
    for i, (first, sample) in enumerate(zip(firsts, samples)):
        lo = first[0]
        tail = [r for r in rows if lo <= r[0] < sample[0] and r[2] == "AddRmsNormBias"][
            -1
        ]
        hi = tail[1]
        body = [r for r in rows if lo <= r[0] < hi]
        assert (
            sum(
                r[2]
                in ((matmul_type,) if isinstance(matmul_type, str) else matmul_type)
                for r in body
            )
            == matmul_count
        ), collections.Counter(r[2] for r in body)
        # The owned raw wrapper can expose the compiled FIA suffix rather than
        # the canonical op type. Both names still denote one FIA per FA layer.
        assert sum(r[2].split("_", 1)[0] == "FusedInferAttentionScore" for r in body) == 16
        compute = [(r[0], r[1]) for r in body]
        comm = [(max(a, lo), min(b, hi)) for a, b in all_comm if a < hi and b > lo]
        assert len(comm) == 128, len(comm)
        intervals = [(max(a, lo), min(b, hi)) for a, b, *_ in apis if a < hi and b > lo]
        waits = [
            (max(a, lo), min(b, hi))
            for a, b, name, tid in apis
            if a < hi and b > lo and "Synchronize" in name
        ]
        events = [
            (a, b)
            for a, b, name, tid in apis
            if name == "aclrtSynchronizeEvent" and lo < a < sample[1] + 1000000
        ]
        if synchronous:
            assert len(events) == 1, events
        nonwait = [
            (max(a, lo), min(b, hi))
            for a, b, name, tid in apis
            if a < hi and b > lo and "Synchronize" not in name
        ]
        cov = duration(compute + comm)
        cu = duration(compute)
        co = duration(comm)
        counts = collections.Counter(name for a, b, name, tid in apis if lo <= a < hi)
        result.append(
            dict(
                step=i,
                start_ns=lo,
                end_ns=hi,
                body_ms=(hi - lo) / 1e6,
                compute_union_ms=cu,
                communication_union_ms=co,
                compute_comm_overlap_ms=cu + co - cov,
                uncovered_by_compute_comm_ms=(hi - lo) / 1e6 - cov,
                api_nonwait_union_across_threads_ms=duration(nonwait),
                api_wait_union_across_threads_ms=duration(waits),
                final_event_wait_start_relative_ms=(
                    (events[0][0] - lo) / 1e6 if synchronous else None
                ),
                final_event_wait_ms=(
                    (events[0][1] - events[0][0]) / 1e6 if synchronous else None
                ),
                model_end_to_sample_end_ms=(sample[1] - hi) / 1e6,
                api_calls_started_in_body=sum(counts.values()),
                top_api_counts=counts.most_common(8),
                compute_ops=collections.Counter(r[2] for r in body),
            )
        )
    return dict(source=str(path), steps=result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    roots = {
        "native": Path(
            "/root/my-ascend-workspace/runs/qwen38-27b-tp2-baseline/20260916-donor0251-v5"
        ),
        "full": Path(
            "/root/my-ascend-workspace/runs/qwen38-27b-tp2-fullgraph/20260916-pilot2"
        ),
    }
    output = dict(
        scope="same fixed2048 C1 workload, historical separate runs; no causal proof from profiled durations; API unions are wall coverage across threads, not CPU utilization; uncovered may contain memory/control tasks",
        arms={},
    )
    for arm, root in roots.items():
        output["arms"][arm] = [
            inspect(root / f"traceloom-msprof-v1/prefill/rank{rank}/analysis.db")
            for rank in [0, 1]
        ]
    a.output.write_text(json.dumps(output, indent=2))
    for arm, ranks in output["arms"].items():
        for rank, data in enumerate(ranks):
            keys = [
                "body_ms",
                "compute_union_ms",
                "communication_union_ms",
                "uncovered_by_compute_comm_ms",
                "final_event_wait_start_relative_ms",
                "final_event_wait_ms",
                "api_calls_started_in_body",
            ]
            print(
                arm,
                rank,
                {k: round(sum(s[k] for s in data["steps"]) / 4, 3) for k in keys},
            )
