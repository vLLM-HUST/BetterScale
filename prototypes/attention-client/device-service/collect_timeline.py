"""Render client-local pull/reduce/drain markers; never fit cross-device clocks."""

import argparse
import gzip
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("run", type=Path)
a = p.parse_args()
events = []
for source in (0, 1):
    banks = json.loads((a.run / f"run/measurements/collect{source}.json").read_text())
    rows = [
        (int(layer), gen, core, times)
        for layer, generations in banks.items()
        for gen, cores in enumerate(generations)
        for core, times in enumerate(cores)
        if times[0] and times[1]
    ]
    origin = min(t[0] for _, _, _, t in rows)
    events.append(
        dict(
            ph="M",
            name="process_name",
            pid=source,
            args=dict(name=f"client{source}: independent clock"),
        )
    )
    for layer, gen, core, t in rows:
        assert t[0] <= t[1] <= t[2] <= t[3]
        for j, name in enumerate(
            (
                "wait for first contribution",
                "pull + weighted reduce (may wait)",
                "drain remaining server DONE",
            )
        ):
            events.append(
                dict(
                    ph="X",
                    name=name,
                    pid=source,
                    tid=core,
                    ts=(t[j] - origin) / 50,
                    dur=(t[j + 1] - t[j]) / 50,
                    args=dict(layer=layer, generation=gen, flag_block_polls=t[4]),
                )
            )
out = a.run / "analysis/client-pull-relative.json.gz"
out.parent.mkdir(parents=True, exist_ok=True)
with gzip.open(out, "wt") as f:
    json.dump(dict(traceEvents=events, displayTimeUnit="ms"), f)
print(out)
