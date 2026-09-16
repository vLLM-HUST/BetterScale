"""Join profiler server cycles to device-selected source generations, not timestamps."""

import argparse
import json
from pathlib import Path
import sqlite3
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
root = parser.parse_args().root.resolve()
clients = [
    json.loads((root / f"run/measurements/client{c}.json").read_text()) for c in (0, 1)
]
report = {"artifact": str(root), "servers": []}
for server in (0, 1):
    trace = json.loads((root / f"run/measurements/expert{server}.json").read_text())[
        "trace"
    ]
    db = sqlite3.connect(root / f"analysis/expert{server}.db")
    db.row_factory = sqlite3.Row
    events = [
        dict(e)
        for e in db.execute(
            "select label,start_ns,end_ns,dur_us from traceloom_event where source_table='TASK' order by start_ns"
        )
    ]
    starts = [i for i, e in enumerate(events) if e["label"] == "neural_prepare"]
    assert len(starts) == len(trace) == 48
    samples = []
    for wave, (start, t) in enumerate(zip(starts, trace)):
        jobs = [
            dict(source=c, generation=t[c + 2], **clients[c][t[c + 2] - 1])
            for c in (0, 1)
            if t[c + 2]
        ]
        if not jobs:
            continue
        stop = starts[wave + 1] if wave + 1 < len(starts) else len(events)
        costs = {}
        for e in events[start:stop]:
            if e["label"] in (
                "neural_prepare",
                "neural_pack",
                "neural_scatter",
                "neural_complete",
                "GroupedMatmul",
                "SwiGlu",
            ):
                costs[e["label"]] = costs.get(e["label"], 0) + e["dur_us"]
        samples.append(dict(wave=wave, jobs=jobs, live_routes=t[1], costs_us=costs))
    summaries = []
    for pattern in ("balanced", "hot8"):
        for n in (1, 16, 32):
            for sources in (1, 2):
                selected = [
                    s
                    for s in samples
                    if len(s["jobs"]) == sources
                    and all(
                        j["pattern"] == pattern
                        and j["rows_per_source"] == n
                        and j["repeat"] > 0
                        for j in s["jobs"]
                    )
                ]
                if not selected:
                    continue
                summaries.append(
                    dict(
                        pattern=pattern,
                        rows_per_source=n,
                        sources=sources,
                        waves=len(selected),
                        costs_us={
                            k: statistics.median(s["costs_us"][k] for s in selected)
                            for k in selected[0]["costs_us"]
                        },
                        warning="neural_prepare includes waiting for source work",
                    )
                )
    report["servers"].append(dict(server=server, summaries=summaries, samples=samples))
(root / "analysis/route-spread-costs.json").write_text(json.dumps(report, indent=2))
print(json.dumps([s["summaries"] for s in report["servers"]], indent=2))
