"""Bounded candidate hotspot audit from TraceLoom's augmented provider tables.

Ordered body/API association, NOT supported exact Ascend replay reconstruction.
Communication envelopes include waiting; uncovered gaps do not prove chip idle.
"""

import argparse
import collections
import itertools
import json
import sqlite3
from pathlib import Path


def name(op):
    return "split_qkv_rmsnorm_rope" if op.startswith("split_qkv_") else op.split("_")[0]


def union(intervals):
    result = []
    for a, b in sorted(intervals):
        if result and a <= result[-1][1]:
            result[-1][1] = max(b, result[-1][1])
        else:
            result.append([a, b])
    return result


def covered(intervals, lo, hi):
    return sum(max(0, min(b, hi) - max(a, lo)) for a, b in intervals)


def inspect(path, start=0):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    compute = c.execute("""select t.startNs,t.endNs,t.modelId,s.value from TASK t
        join COMPUTE_TASK_INFO i using(globalTaskId) join STRING_IDS s on s.id=i.opType
        where t.modelId not in (-1,4294967295) order by t.startNs""").fetchall()
    bodies = [list(g) for _, g in itertools.groupby(compute, key=lambda r: r[2])]
    comm = c.execute("""select startNs,endNs,s.value,o.count from COMMUNICATION_OP o
        join STRING_IDS s on s.id=o.opType order by startNs""").fetchall()
    comm_union = union([(a, b) for a, b, _, _ in comm])
    apis = c.execute("""select a.startNs,a.endNs,s.value from CANN_API a
        join STRING_IDS s on s.id=a.name order by a.startNs""").fetchall()
    replay = [(a, b) for a, b, n in apis if n == "aclmdlRIExecuteAsync"]
    previous_end = replay[start - 1][1] if start else 0
    bodies = bodies[start : start + 4]
    replay = replay[start : start + 4]
    assert len(bodies) == len(replay) == 4
    ops = collections.defaultdict(lambda: [0, 0])
    gaps = collections.defaultdict(lambda: [0, 0, 0])
    waves = []
    for step, body in enumerate(bodies):
        assert sum("FusedInferAttention" in r[3] for r in body) == 48
        assert sum("ArgMax" in r[3] for r in body) == 1
        lo, hi = body[0][0], max(r[1] for r in body)
        for a, b, _, op in body:
            x = ops[name(op)]
            x[0] += 1
            x[1] += b - a
        # End of the running compute union, with its last-ending consumer.
        end, prev = body[0][1], name(body[0][3])
        for a, b, _, op in body[1:]:
            if a > end:
                uncovered = a - end - covered(comm_union, end, a)
                x = gaps[prev + " -> " + name(op)]
                x[0] += 1
                x[1] += a - end
                x[2] += uncovered
            if b > end:
                end, prev = b, name(op)
        embedding = next(i for i, r in enumerate(body) if name(r[3]) == "GreaterEqual")
        sample = next(i for i, r in enumerate(body) if name(r[3]) == "ArgMaxV2")
        plans = [
            (a, b, n)
            for a, b, n in apis
            if "FusedInferAttention" in n
            and (replay[step - 1][1] if step else previous_end) <= a < replay[step][0]
        ]
        # Do not add nested tiling durations. Inner WS and execute are disjoint.
        plans = [(a, b, n) for a, b, n in plans if not n.endswith("Tiling")]
        waves.append(
            dict(
                step=step,
                start_ns=lo,
                end_ns=hi,
                body_ms=(hi - lo) / 1e6,
                replay_end_to_device_start_ms=(lo - replay[step][1]) / 1e6,
                next_body_gap_ms=(
                    (bodies[step + 1][0][0] - hi) / 1e6
                    if step + 1 < len(bodies)
                    else None
                ),
                control_prefix_kernels=embedding,
                control_suffix_kernels=len(body) - sample - 1,
                control_prefix_span_ms=(body[embedding][0] - lo) / 1e6,
                control_suffix_span_ms=(hi - body[sample][1]) / 1e6,
                compute_union_ms=sum(
                    b - a for a, b in union([(r[0], r[1]) for r in body])
                )
                / 1e6,
                communication_union_ms=covered(comm_union, lo, hi) / 1e6,
                host_plan_ms=sum(b - a for a, b, _ in plans) / 1e6,
                host_plan_overlap_previous_body_ms=(
                    sum(
                        max(
                            0,
                            min(b, waves[-1]["end_ns"]) - max(a, waves[-1]["start_ns"]),
                        )
                        for a, b, _ in plans
                    )
                    / 1e6
                    if step
                    else None
                ),
                communications=dict(
                    collections.Counter(n for a, b, n, _ in comm if lo <= a < hi)
                ),
            )
        )
    return dict(
        db=str(path.resolve()),
        scope="four ordered bodies, rank-local provider timestamps",
        waves=waves,
        operators=[
            dict(op=n, count_per_wave=v[0] / 4, sum_ms_per_wave=v[1] / 4e6)
            for n, v in sorted(ops.items(), key=lambda x: -x[1][1])
        ],
        gap_boundaries=[
            dict(
                boundary=n,
                count_per_wave=v[0] / 4,
                raw_gap_ms_per_wave=v[1] / 4e6,
                communication_uncovered_ms_per_wave=v[2] / 4e6,
            )
            for n, v in sorted(gaps.items(), key=lambda x: -x[1][2])
        ],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    result = dict(
        caveats=[
            "Kernel sums overlap other engines; not additive critical-path attribution.",
            "Prefix boundary is first embedding GreaterEqual; suffix follows ArgMax and includes output cast.",
            "Ordered API/body association is diagnostic, not exact TraceLoom replay partition.",
        ],
        ranks=[inspect(a.directory / f"owned-rank{r}.db") for r in range(2)],
    )
    (a.directory / "hotspots.json").write_text(json.dumps(result, indent=2))
