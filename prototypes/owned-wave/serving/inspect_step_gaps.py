"""Graph boundaries vs actual compute/communication coverage in provider time.

Handles repeated native graph IDs through their repeated first compute task.
Ordered body grouping is guarded by48 FIA tasks; it is not exact replay partition.
"""

import argparse
import collections
import json
import sqlite3
from pathlib import Path
from inspect_hotspots import union, covered


def inspect(path, controls=False):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    compute = c.execute(
        """select t.startNs,t.endNs,t.modelId,t.streamId,t.taskId,s.value
      from TASK t join COMPUTE_TASK_INFO i using(globalTaskId)
      join STRING_IDS s on s.id=i.opType order by t.startNs"""
    ).fetchall()
    graph = [r for r in compute if r[2] not in (None, -1, 4294967295)]
    first = {}
    for r in graph:
        first.setdefault(r[2], tuple(r[2:5]))
    starts = [i for i, r in enumerate(graph) if tuple(r[2:5]) == first[r[2]]]
    bodies = [graph[a:b] for a, b in zip(starts, starts[1:] + [len(graph)])]
    replay_count = c.execute(
        "select count(*) from CANN_API a join STRING_IDS s on s.id=a.name where s.value='aclmdlRIExecuteAsync'"
    ).fetchone()[0]
    assert len(bodies) == replay_count, (len(bodies), replay_count)
    comm = c.execute("select startNs,endNs from COMMUNICATION_OP").fetchall()
    all_coverage = union([(r[0], r[1]) for r in compute] + comm)
    compute_coverage = union([(r[0], r[1]) for r in compute])
    comm_coverage = union(comm)
    waves = []
    for i, body in enumerate(bodies):
        assert len({r[2] for r in body}) == 1
        assert sum("FusedInferAttention" in r[5] for r in body) == 48, (i, len(body))
        lo, hi = body[0][0], max(r[1] for r in body)
        next_lo = bodies[i + 1][0][0] if i + 1 < len(bodies) else None
        row = dict(
            step=i,
            model_id=body[0][2],
            start_ns=lo,
            end_ns=hi,
            body_ms=(hi - lo) / 1e6,
            ops=dict(collections.Counter(r[5] for r in body)),
        )
        if next_lo is not None:
            extra = [r for r in compute if hi <= r[0] < next_lo]
            row.update(
                cycle_ms=(next_lo - lo) / 1e6,
                boundary_gap_us=(next_lo - hi) / 1e3,
                boundary_compute_us=covered(compute_coverage, hi, next_lo) / 1e3,
                boundary_communication_us=covered(comm_coverage, hi, next_lo) / 1e3,
                boundary_uncovered_us=(
                    (next_lo - hi) - covered(all_coverage, hi, next_lo)
                )
                / 1e3,
                cycle_uncovered_ms=((next_lo - lo) - covered(all_coverage, lo, next_lo))
                / 1e6,
                boundary_ops=dict(collections.Counter(r[5] for r in extra)),
            )
        if controls and next_lo is not None:
            tasks = c.execute(
                """select t.startNs,t.endNs,t.streamId,t.modelId,s.value
                from TASK t join STRING_IDS s on s.id=t.taskType
                where t.startNs>=? and t.startNs<? order by t.startNs""",
                (hi, next_lo),
            ).fetchall()
            groups = collections.defaultdict(list)
            for task in tasks:
                groups[tuple(task[2:])].append(task)
            row["tasks_starting_in_boundary"] = [
                dict(
                    stream=stream,
                    model_id=model,
                    type=kind,
                    count=len(rs),
                    start_us=(rs[0][0] - hi) / 1e3,
                    clipped_end_us=(min(next_lo, max(r[1] for r in rs)) - hi) / 1e3,
                    sum_clipped_us=sum(min(next_lo, r[1]) - r[0] for r in rs) / 1e3,
                )
                for (stream, model, kind), rs in groups.items()
            ]
        waves.append(row)
    return dict(
        db=str(path.resolve()),
        scope="rank-local ordered compute body boundaries; uncovered excludes compute/comm envelopes but NOT all control/memory work",
        waves=waves,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("db", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--controls", action="store_true")
    a = p.parse_args()
    a.output.write_text(json.dumps(inspect(a.db, a.controls), indent=2))
