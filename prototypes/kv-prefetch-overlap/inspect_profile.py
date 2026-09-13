"""Summarize same-rank native GEMM shapes and collective-envelope overlap."""

import argparse
import bisect
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("db")
p.add_argument("--out", required=True)
a = p.parse_args()
c = sqlite3.connect(f"file:{Path(a.db).resolve()}?mode=ro", uri=True)
comm = list(
    c.execute(
        "select o.startNs,o.endNs from COMMUNICATION_OP o join STRING_IDS s on s.id=o.opType where lower(s.value) like '%allgather%' order by o.startNs"
    )
)
union = []
for start, end in comm:
    if union and start <= union[-1][1]:
        union[-1][1] = max(end, union[-1][1])
    else:
        union.append([start, end])
starts = [v[0] for v in union]
records = defaultdict(lambda: dict(count=0, duration_us=0, overlap_us=0))
q = """select s.value,sh.value,dt.value,ct.blockNum,ct.mixBlockNum,t.startNs,t.endNs
from COMPUTE_TASK_INFO ct join STRING_IDS s on s.id=ct.opType
join STRING_IDS sh on sh.id=ct.inputShapes join STRING_IDS dt on dt.id=ct.inputDataTypes
join TASK t on t.globalTaskId=ct.globalTaskId where lower(s.value) like '%matmul%' """
for name, shape, dtype, blocks, mixed, start, end in c.execute(q):
    x = records[(name, shape, dtype, blocks, mixed)]
    x["count"] += 1
    x["duration_us"] += (end - start) / 1000
    i = max(0, bisect.bisect_right(starts, start) - 1)
    while i < len(union) and union[i][0] < end:
        x["overlap_us"] += (
            max(0, min(end, union[i][1]) - max(start, union[i][0])) / 1000
        )
        i += 1
result = []
for (name, shape, dtype, blocks, mixed), x in records.items():
    result.append(
        dict(
            name=name,
            shape=shape,
            dtype=dtype,
            block_num=blocks,
            mix_block_num=mixed,
            mean_us=x["duration_us"] / x["count"],
            **x,
        )
    )
result.sort(key=lambda x: -x["duration_us"])
Path(a.out).write_text(
    json.dumps(
        dict(
            source=str(Path(a.db).resolve()),
            caveat="Native collective envelopes include wait/queue time; overlap is not proof of transferred bytes or zero interference.",
            allgather_envelopes=len(comm),
            matmuls=result,
        ),
        indent=2,
    )
)
for r in result[:10]:
    print(
        r["name"],
        r["shape"],
        r["block_num"],
        r["mix_block_num"],
        round(r["mean_us"], 2),
        round(r["overlap_us"] / r["duration_us"] * 100, 1),
    )
