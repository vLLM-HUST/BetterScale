"""Report bounded paired-client episodes and coordinator-observed server phases.

Server time origins are independent, NOT cross-device clock alignment. Occupancy
is an up/down command envelope, not measured utilization of individual Cubes.
"""

import argparse
import collections
import gzip
import json
from pathlib import Path
import statistics

p = argparse.ArgumentParser()
p.add_argument("capsule", type=Path)
a = p.parse_args()
clients = [
    json.loads((a.capsule / "roles" / f"client{i}.json").read_text()) for i in range(2)
]
report = dict(scope=__doc__.strip(), clients=clients, servers=[])
display = []
names = {
    (0, 1): "pull",
    (0, 2): "pack",
    (0, 3): "swiglu",
    (0, 4): "return",
    (1, 1): "up",
    (1, 2): "down",
}
for owner in range(4):
    d = json.loads((a.capsule / "roles" / f"expert{owner}.json").read_text())[
        "diagnostics"
    ]
    assert not d["rolling_trace"] and len(d["events"]) < 512
    waves = [[], []]
    for row in d["trace"]:
        waves[row[3]].append(row)
    cursors = [0, 0]
    phases = collections.defaultdict(list)
    paired = collections.Counter()
    origin = min(e[3] for e in d["events"])
    for e in d["events"]:
        engine, kind, slot, begin, end, rows, seq, part = e
        row = waves[slot][cursors[slot]]
        gens = [g for g in row[:2] if g]
        case = (max(gens) - 1) // 10
        assert all((g - 1) // 10 == case for g in gens)
        measured = all((g - 1) % 10 >= 2 for g in gens)
        if measured:
            phases[case].append(e)
        if (engine, kind) == (0, 4):
            if measured:
                paired[case, "paired" if len(gens) == 2 else "solo"] += 1
            cursors[slot] += 1
        display.append(
            dict(
                name=names[engine, kind],
                ph="X",
                pid=owner,
                tid=engine,
                ts=(begin - origin) / 50,
                dur=(end - begin) / 50,
                args=dict(
                    slot=slot, generations=row[:2], layers=row[7:9], measured=measured
                ),
            )
        )
    summary = {}
    for case, events in phases.items():
        span = (max(e[4] for e in events) - min(e[3] for e in events)) / 50
        costs = collections.defaultdict(list)
        for engine, kind, slot, begin, end, *_ in events:
            costs[names[engine, kind]].append((end - begin) / 50)
        cube = sum(costs["up"]) + sum(costs["down"])
        summary[("same", "different", "offset")[case]] = dict(
            solo=paired[case, "solo"],
            paired=paired[case, "paired"],
            envelope_us=span,
            cube_command_envelope_fraction=cube / span,
            median_us={k: statistics.median(v) for k, v in costs.items()},
        )
    report["servers"].append(dict(owner=owner, cases=summary))
out = a.capsule / "analysis"
out.mkdir(exist_ok=True)
(out / "concurrency-summary.json").write_text(json.dumps(report, indent=2) + "\n")
with gzip.open(out / "server-phases-relative.json.gz", "wt") as f:
    json.dump(dict(traceEvents=display), f)
print(json.dumps(report["servers"], indent=2))
