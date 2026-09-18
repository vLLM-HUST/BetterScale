"""Add recorded host wave ranges to a TraceLoom provider-clock device view.

No first-event normalization or fitted clock correction. Both timestamp sets
come from the same CANN/PyTorch provider on one host.
"""

import argparse
import gzip
import json
import sqlite3
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("analysis", type=Path)
a = p.parse_args()
base = a.analysis / "qwen38-prefill-attention5-provider-clock.json.gz"
with gzip.open(base, "rt") as f:
    trace = json.load(f)
origin = trace["metadata"]["time_origin_ns"]
for event in trace["traceEvents"]:
    if event.get("pid") == 120 and event.get("name") == "process_name":
        event["args"]["name"] = "Qwen3.8 attention · provider clock"
wave_count = 0
trace["traceEvents"].append(
    dict(
        ph="M",
        name="process_name",
        pid=121,
        tid=0,
        args=dict(name="Prefill waves · recorded host API ranges"),
    )
)
for item in trace["metadata"]["roles"]:
    rank = item["display_rank"]
    with sqlite3.connect(item["source"]) as db:
        rows = db.execute(
            'select a.startNs,a.endNs,s.value from PYTORCH_API a join STRING_IDS s on a.name=s.id where s.value like "swe_prefill/%" order by cast(a.startNs as integer)'
        ).fetchall()
    assert len(rows) == 2, (rank, len(rows))
    trace["traceEvents"].append(
        dict(
            ph="M",
            name="thread_name",
            pid=121,
            tid=rank + 1,
            args=dict(name=item["role"]),
        )
    )
    for begin, end, label in rows:
        fields = label.split("/")
        valid = int(fields[3].removeprefix("valid"))
        bucket = int(fields[4].removeprefix("bucket"))
        trace["traceEvents"].append(
            dict(
                ph="X",
                name=label,
                pid=121,
                tid=rank + 1,
                ts=(int(begin) - origin) / 1000,
                dur=(int(end) - int(begin)) / 1000,
                args=dict(
                    valid_tokens=valid,
                    bucket_rows=bucket,
                    fill_ratio=valid / bucket,
                    source_database=item["source"],
                    clock="provider timestamps; no additional fitted calibration",
                ),
            )
        )
        wave_count += 1
trace["metadata"]["host_wave_markers"] = wave_count
output = a.analysis / "qwen38-prefill-attention5-annotated.json.gz"
with gzip.open(output, "wt", compresslevel=6) as f:
    json.dump(trace, f)
print(output)
