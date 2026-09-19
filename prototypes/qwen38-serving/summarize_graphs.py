"""Exact TraceLoom graph costs, retaining operator identity and physical shapes."""

import argparse
import json
from pathlib import Path
import sqlite3

p = argparse.ArgumentParser()
p.add_argument("db", type=Path)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
with sqlite3.connect(f"file:{a.db.resolve()}?mode=ro", uri=True) as c:
    graphs = []
    for launch, duration, evidence in c.execute(
        "select launch_id,dur_us,evidence_level from traceloom_graph_launch order by start_ns"
    ).fetchall():
        if evidence != "exact_direct":
            raise ValueError(f"non-exact graph: {launch} {evidence}")
        rows = c.execute(
            """
            select e.op_type, count(*), sum(m.dur_us)
            from traceloom_graph_body_member m join traceloom_event e using(event_id)
            where m.launch_id=? group by e.op_type order by sum(m.dur_us) desc
        """,
            (launch,),
        ).fetchall()
        transpose = c.execute(
            """
            select s.value,count(*),sum(m.dur_us)
            from traceloom_graph_body_member m join traceloom_event e using(event_id)
            join TASK t on t.rowid=m.source_row_id
            join COMPUTE_TASK_INFO i on i.globalTaskId=t.globalTaskId
            join STRING_IDS s on s.id=i.inputShapes
            where m.launch_id=? and m.source_table='TASK' and e.op_type='Transpose'
            group by s.value
        """,
            (launch,),
        ).fetchall()
        graphs.append(
            dict(
                launch_id=launch,
                body_us=duration,
                evidence=evidence,
                operators=[dict(type=k, count=n, sum_us=d) for k, n, d in rows],
                transpose_shapes=[
                    dict(shape=s, count=n, sum_us=d) for s, n, d in transpose
                ],
            )
        )
result = dict(
    db=str(a.db.resolve()),
    scope="rank-local exact graph bodies; operator sums are not exclusive critical-path time",
    graphs=graphs,
)
a.output.write_text(json.dumps(result, indent=2))
print(json.dumps(dict(output=str(a.output), exact_graphs=len(graphs))))
