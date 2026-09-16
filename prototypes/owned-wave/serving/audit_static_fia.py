"""Provider graph-membership and TraceLoom host-update counts, not speedup attribution."""

import argparse
import json
import sqlite3
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("capsule", type=Path)
p.add_argument("--layers", type=int, required=True)
a = p.parse_args()
exports = json.loads((a.capsule / "traceloom/exports.json").read_text())["files"]
rows = []
for entry in exports:
    with sqlite3.connect(f"file:{entry['analysis']}?mode=ro", uri=True) as db:
        calls = dict(
            db.execute(
                "select api_name,count(*) from traceloom_runtime_call group by api_name"
            )
        )
    with sqlite3.connect(f"file:{entry['source']}?mode=ro", uri=True) as db:
        kernels = db.execute("""select s.value,t.modelId,count(*) from TASK t
            join COMPUTE_TASK_INFO i using(globalTaskId)
            join STRING_IDS s on s.id=i.opType
            where lower(s.value) like '%fusedinferattention%' or lower(s.value) like '%argmax%'
            group by s.value,t.modelId""").fetchall()
        provider_calls = dict(db.execute("""select s.value,count(*) from CANN_API a
            join STRING_IDS s on s.id=a.name group by s.value"""))
    names = [
        "aclmdlRIExecuteAsync",
        "aclmdlRICaptureTaskUpdateBegin",
        "aclmdlRICaptureTaskUpdateEnd",
        "aclnnInnerFusedInferAttentionScore",
        "aclnnInnerFusedInferAttentionScoreGetWorkspaceSize",
        "aclnnInnerFusedInferAttentionScoreTiling",
    ]
    counts = {n: calls.get(n, 0) for n in names}
    assert all(counts[n] == provider_calls.get(n, 0) for n in names)
    replays = counts["aclmdlRIExecuteAsync"]
    attention = [r for r in kernels if "fusedinferattention" in r[0].lower()]
    assert replays and sum(r[2] for r in attention) == a.layers * replays
    assert all(r[1] not in (-1, 4294967295) for r in attention)
    if entry["arm"] == "owned":
        assert all(counts[n] == 0 for n in names[1:])
        assert all(r[1] not in (-1, 4294967295) for r in kernels)
    else:
        assert counts["aclmdlRICaptureTaskUpdateBegin"] == a.layers * replays
    rows.append(
        dict(
            arm=entry["arm"],
            rank=entry["rank"],
            host_api_counts=counts,
            graph_tasks=[dict(op=r[0], model_id=r[1], count=r[2]) for r in kernels],
            analysis=entry["analysis"],
        )
    )
result = dict(
    status="PASS",
    scope="bounded per-arm windows; graph inclusion and host-boundary audit, not matched kernel timing",
    rows=rows,
)
(a.capsule / "traceloom/static-fia-audit.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
