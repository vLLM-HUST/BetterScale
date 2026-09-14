"""Reduce matched graph-block trials without discarding regressions."""

import argparse
import json
from pathlib import Path
from statistics import median

p = argparse.ArgumentParser()
p.add_argument("records", type=Path)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
rows = [json.loads(x) for x in a.records.read_text().splitlines()]
keys = sorted({(r["label"], r["m"], r["mib"], r["direction"]) for r in rows})
result = []
for key in keys:
    selected = [
        r for r in rows if (r["label"], r["m"], r["mib"], r["direction"]) == key
    ]

    def med(mode, column):
        records = [r[column] for r in selected if r["mode"] == mode]
        assert len(records) == 3
        return median(records)

    serial, overlap = med("serial", "total_ms"), med("overlap", "total_ms")
    result.append(
        dict(
            label=key[0],
            m=key[1],
            mib=key[2],
            direction=key[3],
            serial_ms=serial,
            overlap_ms=overlap,
            total_reduction_fraction=1 - overlap / serial,
            compute_slowdown_fraction=med("overlap", "compute_ms")
            / med("compute", "compute_ms")
            - 1,
            copy_slowdown_fraction=med("overlap", "copy_ms") / med("copy", "copy_ms")
            - 1,
            isolated_compute_ms=med("compute", "compute_ms"),
            isolated_copy_ms=med("copy", "copy_ms"),
        )
    )
a.out.write_text(
    json.dumps(
        dict(
            scope="single local910B2; CANN9.0.1; torch-npu2.10; 16-operation blocks; 3 alternating trials; not serving",
            cases=result,
        ),
        indent=2,
    )
    + "\n"
)
