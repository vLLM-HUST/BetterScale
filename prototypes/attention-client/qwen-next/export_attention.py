"""Export one complete native attention timeline, without resident server bars."""

import argparse
import gzip
import json
from pathlib import Path
import sqlite3
import sys

from profile_export import TOOL, run

p = argparse.ArgumentParser()
p.add_argument("root", type=Path)
p.add_argument("--source", type=int, choices=(0, 1), default=1)
a = p.parse_args()
root = a.root.resolve()
role = f"attention{a.source}"
output = root / "analysis"
output.mkdir(exist_ok=True)
profiles = list((root / "profile").glob(f"{role}_*"))
assert len(profiles) == 1
profile = profiles[0]
sources = list((profile / "ASCEND_PROFILER_OUTPUT").glob("ascend_pytorch_profiler*.db"))
if not sources:
    run(
        [
            sys.executable,
            "-c",
            "from torch_npu.profiler.profiler import analyse; import sys; analyse(sys.argv[1], max_process_number=1, export_type='db')",
            str(profile),
        ],
        output / f"{role}-parse.log",
    )
    sources = list(
        (profile / "ASCEND_PROFILER_OUTPUT").glob("ascend_pytorch_profiler*.db")
    )
assert len(sources) == 1
with sqlite3.connect(sources[0]) as db:
    assert db.execute("pragma quick_check").fetchone()[0] == "ok"
    mapping = db.execute("select * from RANK_DEVICE_MAP").fetchall()
target = output / f"{role}.db"
if not target.exists():
    run(
        [str(TOOL), str(sources[0]), "--threads", "4", "--output", str(target)],
        output / f"{role}-analyze.log",
    )
with sqlite3.connect(target) as db:
    db.execute(
        "create index if not exists strengthen_viz_edge_child on traceloom_viz_edge(child_node_id)"
    )
    db.commit()
partial = output / f"{role}-full-decode.partial.json.gz"
final = output / f"{role}-full-decode.json.gz"
run(
    [str(TOOL), "export-perfetto", str(target), "--output", str(partial)],
    output / f"{role}-export.log",
    timeout=600,
)
with gzip.open(partial, "rt") as f:
    result = json.load(f)
assert result["traceEvents"]
partial.rename(final)
(output / f"{role}-export-receipt.json").write_text(
    json.dumps(
        dict(
            source=str(sources[0]),
            analysis=str(target),
            provider_mapping=mapping,
            timeline=str(final),
            event_count=len(result["traceEvents"]),
            scope="Native single-source timeline, includes eager prefill and FULL decode; no cross-device clock fit or server residency bars.",
        ),
        indent=2,
    )
)
print(final)
