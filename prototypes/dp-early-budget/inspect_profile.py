"""Compare native Qwen replay cadence and inter-body coverage, not host clocks."""

import argparse
import bisect
import json
from pathlib import Path
import sqlite3
from statistics import median


def inspect(root, rank):
    source = next(
        (root / "profile").glob(
            f"rank{rank}_*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_{rank}.db"
        )
    )
    with sqlite3.connect(source) as c:
        norms = c.execute("""select t.startNs,t.endNs,t.streamId,t.taskId
            from TASK t join COMPUTE_TASK_INFO i using(globalTaskId)
            join STRING_IDS s on s.id=i.name
            where t.modelId!=4294967295 and s.value='AddRmsNormBias'
            order by t.startNs""").fetchall()
        # The first and final norms delimit the captured transformer body.
        # Name alone is not unique: keep each captured stream/task identity.
        # Profiling may start halfway through the preceding replay. Do not
        # choose the first observed norm: use ordered capture task identities
        # on this one model stream, then retain only complete cycles.
        assert len({r[2] for r in norms}) == 1
        first, last = min(r[2:] for r in norms), max(r[2:] for r in norms)
        assert first != last
        starts = [r[0] for r in norms if r[2:] == first]
        ends = [r[1] for r in norms if r[2:] == last]
        tasks = c.execute("""select t.startNs,t.endNs from TASK t
            where exists(select 1 from COMPUTE_TASK_INFO i where i.globalTaskId=t.globalTaskId)
            or exists(select 1 from COMMUNICATION_TASK_INFO i where i.globalTaskId=t.globalTaskId)
            order by t.startNs""").fetchall()
    merged = []
    for start, end in tasks:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    rows = []
    for start, following in zip(starts, starts[1:]):
        hits = ends[
            bisect.bisect_right(ends, start) : bisect.bisect_left(ends, following)
        ]
        assert len(hits) == 1, "Incomplete or ambiguous transformer body"
        end = hits[0]
        covered = sum(max(0, min(b, following) - max(a, end)) for a, b in merged)
        rows.append(
            dict(
                body_ms=(end - start) / 1e6,
                cadence_ms=(following - start) / 1e6,
                between_bodies_ms=(following - end) / 1e6,
                uncovered_between_bodies_ms=(following - end - covered) / 1e6,
            )
        )
    return dict(
        rank=rank,
        cycles=len(rows),
        first_norm_identity=first,
        last_norm_identity=last,
        medians={key: median(row[key] for row in rows) for key in rows[0]},
        rows=rows,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    a = p.parse_args()
    result = dict(
        scope="Profiled two-request FULL ordinary decode; first/final captured norm delimit transformer body. Between-body interval includes logits/sampling/metadata, NOT idle time. Uncovered means no observed compute/communication task, not all hardware idle.",
        ranks=[inspect(a.root, rank) for rank in range(2)],
    )
    (a.root / "profile-comparison.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            [
                {k: v for k, v in rank.items() if k != "rows"}
                for rank in result["ranks"]
            ],
            indent=2,
        )
    )
