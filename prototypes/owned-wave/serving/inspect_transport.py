"""Join host submit -> CANN copy API -> device DMA using provider connectionId.

Direction/bytes follow this closed reactor's three-ingress/one-egress contract;
provider rows independently supply copy timing and stream. No new NPU run.
"""

import argparse
import json
import sqlite3
from pathlib import Path
from inspect_hotspots import union, covered
from inspect_step_gaps import inspect


def inspect_transport(capsule, rank):
    out = capsule / "traceloom"
    schedule = json.loads(
        (capsule / f"engine/owned-profile/schedule-rank{rank}.json").read_text()
    )["events"]
    stages = {}
    for r in schedule:
        if "sequence" in r:
            stages.setdefault(r["sequence"], {})[r["event"]] = r
    audit = json.loads((out / "schedule-audit.json").read_text())[rank]
    seqs = audit["sampled_sequences"]
    c = sqlite3.connect(f"file:{out/f'owned-rank{rank}.db'}?mode=ro", uri=True)
    bodies = inspect(out / f"owned-rank{rank}.db")["waves"]
    body_union = union([(r["start_ns"], r["end_ns"]) for r in bodies])
    kernels = union(
        c.execute(
            "select t.startNs,t.endNs from TASK t join COMPUTE_TASK_INFO i using(globalTaskId)"
        ).fetchall()
    )
    apis = c.execute(
        "select a.startNs,a.endNs,a.connectionId,s.value from CANN_API a join STRING_IDS s on s.id=a.name where s.value like '%Memcpy%' order by a.startNs"
    ).fetchall()
    transfers = []
    for lo, hi, cid, name in apis:
        matches = [
            seq
            for seq in seqs
            if stages[seq]["submit_begin"]["wall_ns"]
            <= lo
            <= hi
            <= stages[seq]["submit_end"]["wall_ns"]
        ]
        assert len(matches) == 1
        seq = matches[0]
        index = seqs.index(seq)
        rows = c.execute(
            "select t.startNs,t.endNs,t.streamId,t.modelId,s.value from TASK t join STRING_IDS s on s.id=t.taskType where t.connectionId=?",
            (cid,),
        ).fetchall()
        assert len(rows) == 1 and rows[0][4] == "MEMCPY_ASYNC"
        a, b, stream, model, _ = rows[0]
        direction = "D2H" if name == "aclrtMemcpyAsyncWithCondition" else "H2D"
        assert name in ("aclrtMemcpyAsync", "aclrtMemcpyAsyncWithCondition")
        transfers.append(
            dict(
                sequence=seq,
                direction=direction,
                connection_id=cid,
                stream=stream,
                model_id=model,
                api=name,
                api_duration_us=(hi - lo) / 1e3,
                start_ns=a,
                end_ns=b,
                dma_us=(b - a) / 1e3,
                body_overlap_us=covered(body_union, a, b) / 1e3,
                compute_overlap_us=covered(kernels, a, b) / 1e3,
                relative_to_own_body_start_us=(a - bodies[index]["start_ns"]) / 1e3,
                relative_to_own_body_end_us=(a - bodies[index]["end_ns"]) / 1e3,
            )
        )
    for seq in seqs:
        assert (
            sum(r["sequence"] == seq and r["direction"] == "H2D" for r in transfers)
            == 3
        )
        assert (
            sum(r["sequence"] == seq and r["direction"] == "D2H" for r in transfers)
            == 1
        )
    syncs = c.execute(
        "select a.startNs,a.endNs,s.value from CANN_API a join STRING_IDS s on s.id=a.name where s.value like '%Synchronize%' order by a.startNs"
    ).fetchall()
    origin = next(r["wall_ns"] for r in schedule if r["event"] == "profile_start")
    return dict(
        rank=rank,
        scope=__doc__,
        transfers=transfers,
        synchronizations=[
            dict(api=n, start_ms=(a - origin) / 1e6, duration_ms=(b - a) / 1e6)
            for a, b, n in syncs
        ],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("capsule", type=Path)
    a = p.parse_args()
    result = [inspect_transport(a.capsule, r) for r in range(2)]
    (a.capsule / "traceloom/transport-audit.json").write_text(
        json.dumps(result, indent=2)
    )
