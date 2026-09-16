"""Export bounded evidence and coordinator phase envelopes, not core utilization."""

import argparse
import csv
import gzip
import json
from pathlib import Path
import statistics

p = argparse.ArgumentParser()
p.add_argument("result", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
x = json.loads(a.result.read_text())
a.output.mkdir(parents=True, exist_ok=True)
summary = dict(
    scope=x["scope"],
    source=str(a.result),
    samples=len(x["samples"]),
    max_selected_relative_l2=max(s["selected_route_relative_l2"] for s in x["samples"]),
    summary=x["summary"],
)
(a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with (a.output / "curve.csv").open("w") as f:
    w = csv.writer(f)
    w.writerow(["case", "median_us", "median_us_per_input_token"])
    for k, v in x["summary"].items():
        w.writerow([k, v["median_us"], v["median_us_per_token"]])
names = {
    (0, 1): "fetch",
    (0, 2): "pack",
    (0, 3): "activate",
    (0, 4): "return",
    (1, 1): "up",
    (1, 2): "down",
    (2, 3): "urgent_activate",
}
trace = []
for pid, label in enumerate(x["summary"]):
    samples = [s for s in x["samples"] if s["case"] == label and s["repeat"] >= 2]
    s = min(samples, key=lambda s: abs(s["span_us"] - x["summary"][label]["median_us"]))
    origin = min(e[3] for e in s["events"])
    trace.append(
        dict(
            ph="M",
            pid=pid,
            tid=0,
            name="process_name",
            args=dict(name=label + " (independent origin)"),
        )
    )
    for e in s["events"]:
        trace.append(
            dict(
                ph="X",
                pid=pid,
                tid=e[0],
                name=names.get(tuple(e[:2]), str(e[:2])),
                ts=(e[3] - origin) / 50,
                dur=(e[4] - e[3]) / 50,
                args=dict(slot=e[2], rows=e[5], part=e[7]),
            )
        )
with gzip.open(a.output / "server-phases.json.gz", "wt") as f:
    json.dump(
        dict(
            traceEvents=trace,
            displayTimeUnit="ms",
            scope="One server coordinator command envelopes; independent case origins; not per-core or cross-device alignment.",
        ),
        f,
    )
