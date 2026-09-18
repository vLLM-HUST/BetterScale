"""Summarize retained coordinator intervals, not pure operator times.

Ring sequence numbers restore order. Do not align different device clocks or
interpret inter-request gaps as idle hardware. Cycles/us is explicit.
"""

import argparse
import collections
import json
from pathlib import Path
import statistics

NAMES = {
    (0, 1): "fetch",
    (0, 2): "pack",
    (0, 3): "activate_quant",
    (0, 4): "convert_export",
    (1, 1): "up",
    (1, 2): "down",
}
p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
p.add_argument("--cycles-per-us", type=float, default=50)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
report = {}
for path in sorted(a.directory.glob("expert*.json")):
    receipt = json.loads(path.read_text())
    events = sorted(receipt["events"], key=lambda e: e[6])
    phases = collections.defaultdict(list)
    gaps = []
    for event in events:
        engine, kind, slot, begin, end, rows, sequence, part = event
        phases[(NAMES[(engine, kind)], rows)].append((end - begin) / a.cycles_per_us)
    for prev, cur in zip(events, events[1:]):
        if (
            prev[:2] == [0, 1]
            and cur[:2] == [0, 2]
            and prev[2] == cur[2]
            and cur[6] == prev[6] + 1
        ):
            gaps.append((cur[3] - prev[4]) / a.cycles_per_us)
    report[path.stem] = dict(
        source=str(path),
        sequence_range=[events[0][6], events[-1][6]],
        phases=[
            dict(
                phase=name,
                routed_rows=rows,
                samples=len(values),
                median_us=statistics.median(values),
            )
            for (name, rows), values in sorted(phases.items())
        ],
        post_fetch_pre_pack=dict(
            samples=len(gaps), median_us=statistics.median(gaps) if gaps else None
        ),
    )
a.output.write_text(
    json.dumps(
        dict(
            cycles_per_us=a.cycles_per_us,
            scope="Coordinator command intervals; post-fetch gap includes Accept/Group/descriptor preparation, not an isolated Group measurement. Retained ring mixes test arms.",
            owners=report,
        ),
        indent=2,
    )
    + "\n"
)
