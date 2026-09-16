"""Summarize unprofiled local FULL expert and remote client event brackets."""

import argparse
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("local", type=Path)
parser.add_argument("remote", type=Path)
args = parser.parse_args()
local = json.loads((args.local / "result.json").read_text())
assert len(local) == 6 and all(r["exact"] for r in local)
assert {(r["layer"], r["rows"]) for r in local} == {
    (layer, rows) for layer in (0, 1) for rows in (1, 16, 32)
}
remote = []
for rank in (0, 1):
    result = json.loads(
        (args.remote / f"run/measurements/attention{rank}.json").read_text()
    )
    assert len(result["checks"]) == 12
    assert all(r["output_exact"] and r["kv_exact"] for r in result["checks"])
    events = [
        r for r in result["events"] if r["event"] == "retire" and r["generation"] >= 13
    ]
    assert len(events) == 12
    remote.extend(events)
table = []
for rows in (1, 16, 32):
    baseline = [value for r in local if r["rows"] == rows for value in r["replay_us"]]
    candidate = [r["client_graph_us"] for r in remote if r["rows"] == rows]
    table.append(
        dict(
            rows=rows,
            local_graph_median_us=statistics.median(baseline),
            remote_graph_median_us=statistics.median(candidate),
            remote_min_us=min(candidate),
            remote_max_us=max(candidate),
            local_trial_count=len(baseline),
            remote_invocations=len(candidate),
        )
    )
print(
    json.dumps(
        dict(
            local_artifact=str(args.local),
            remote_artifact=str(args.remote),
            scope="Same host, full Qwen layer dimensions, two dummy BF16 layers; no profiler",
            local="TP1 native full MLP graph; each trial averages64 prequeued replays",
            remote="Attention2+Expert2; gate through returned result reduction; individual graph events",
            limitations=[
                "Different request scheduling; not an equal-resource throughput comparison",
                "Local repeated input; remote live layer inputs; dummy routing only",
                "Remote fixture retains host oracle gaps; asynchronous other-request progress not benchmarked",
            ],
            table=table,
        ),
        indent=2,
    )
)
