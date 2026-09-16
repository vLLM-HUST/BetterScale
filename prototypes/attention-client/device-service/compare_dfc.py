"""Compare qualified EP2 DFC and remote synthetic controls, not serving throughput."""

import argparse
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("dfc", type=Path)
parser.add_argument("remote", type=Path)
a = parser.parse_args()
assert (a.dfc / "run/exit.txt").read_text().strip() == "0"
assert (a.remote / "run/exit.txt").read_text().strip() == "0"
baseline = sum(
    [json.loads((a.dfc / f"results/rank{i}.json").read_text()) for i in (0, 1)], []
)
remote = sum(
    [
        json.loads((a.remote / f"run/measurements/client{i}.json").read_text())
        for i in (0, 1)
    ],
    [],
)
assert len(baseline) == 12 and len(remote) == 48
assert max(r["relative_l2"] for r in baseline + remote) < 0.01
rows = []
for pattern in ("balanced", "hot8"):
    for n in (1, 16, 32):
        bs = [
            r for r in baseline if r["pattern"] == pattern and r["rows_per_source"] == n
        ]
        rs = [
            r["client_us"]
            for r in remote
            if r["pattern"] == pattern and r["rows_per_source"] == n and r["repeat"] > 0
        ]
        # Each DFC trial ends after both EP participants process their source tokens;
        # retain the slower participant's per-trial time instead of hiding skew.
        trials = [max(bs[0]["trials_us"][i], bs[1]["trials_us"][i]) for i in range(5)]
        b = statistics.median(trials)
        r = statistics.median(rs)
        rows.append(
            dict(
                pattern=pattern,
                rows_per_source=n,
                global_rows=2 * n,
                dfc_ep2_us=b,
                remote_client_us=r,
                remote_range_us=[min(rs), max(rs)],
                diagnostic_ratio=r / b,
            )
        )
print(
    json.dumps(
        dict(
            dfc=str(a.dfc),
            remote=str(a.remote),
            rows=rows,
            scope="BF16 H2048 M768 E128 K8, precomputed routes, no gate, full graphs, no profiler",
            limitations=[
                "DFC uses paired synchronous EP2 sources; remote sources are independent",
                "DFC averages64 replay cycles per trial; remote has6 warm individual samples/case",
                "Two versus four cards; same host but not same card set or equal-resource throughput",
                "DFC is lab A2 build, not native pinned donor package; structured synthetic weights/routes",
            ],
        ),
        indent=2,
    )
)
