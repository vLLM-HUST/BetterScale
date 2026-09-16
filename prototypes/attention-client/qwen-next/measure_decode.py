"""Measure native FULL decode from provider DB; never mix eager prefill samples.

Uses the single compute stream of this TP1 attention graph, one occurrence of
its first task ID per replay. Fails if the captured topology changes rather than
silently treating host submission duration as device execution latency.
"""

import argparse
import collections
import json
from pathlib import Path
import sqlite3
import statistics

p = argparse.ArgumentParser()
p.add_argument("capsule", type=Path)
p.add_argument("--source", type=int, default=1, choices=(0, 1))
a = p.parse_args()
dbs = list(
    a.capsule.glob(
        f"profile/attention{a.source}_*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler*.db"
    )
)
assert len(dbs) == 1
with sqlite3.connect(dbs[0]) as db:
    rows = db.execute("""
        SELECT t.startNs,t.endNs,t.modelId,t.streamId,t.taskId,s.value
        FROM TASK t JOIN COMPUTE_TASK_INFO i USING(globalTaskId)
        JOIN STRING_IDS s ON s.id=i.name ORDER BY t.startNs
    """).fetchall()
models = {r[2] for r in rows if r[5].startswith("neural_collect") and r[2] != 2**32 - 1}
assert len(models) == 1, models
model = models.pop()
rows = [r for r in rows if r[2] == model]
assert (
    len({r[3] for r in rows}) == 1
), "multi-stream graph needs a different boundary analysis"
first_id = min(r[4] for r in rows)
steps = []
for row in rows:
    if row[4] == first_id:
        steps.append([])
    assert steps
    steps[-1].append(row)
counts = {len(s) for s in steps}
assert len(counts) == 1, counts
assert all(sum(r[5].startswith("neural_collect") for r in s) == 48 for s in steps)
groups = collections.defaultdict(list)
for start, end, _, _, _, name in rows:
    groups[name].append((end - start) / 1000)
remote = {
    k: dict(count=len(v), median_us=statistics.median(v), sum_ms=sum(v) / 1000)
    for k, v in groups.items()
    if k.startswith("neural_") or "Unpermute" in k
}
spans = [(max(r[1] for r in s) - s[0][0]) / 1e6 for s in steps]
result = dict(
    capsule=a.capsule.name,
    source=a.source,
    model_id=model,
    decode_steps=len(steps),
    layer_calls=len(steps) * 48,
    graph_median_ms=statistics.median(spans),
    graph_spans_ms=spans,
    remote_tasks=remote,
    scope="Profiled native FULL decode only; not steady serving throughput or pure server GEMM",
)
output = a.capsule / "analysis" / f"attention{a.source}-decode-metrics.json"
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
