"""Analyze native DBs and export separate, readable TraceLoom timelines."""

import argparse
import gzip
import json
import sqlite3
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("capsule", type=Path)
a = p.parse_args()
root = a.capsule.resolve()
tool = Path("/workspace/my-ascend-workspace/.tools/traceloom-37323af")
out = root / "traceloom"
out.mkdir(exist_ok=False)
entries = json.loads((root / "profile-exports.json").read_text())
manifest = dict(
    tool_commit=(tool / "SOURCE_COMMIT").read_text().strip(),
    clock_scope="separate rank-local timelines; no cross-rank alignment",
    files=[],
)
for entry in entries:
    name = f"{entry['arm']}-rank{entry['rank']}"
    source = Path(entry["db"])
    db = out / f"{name}.db"
    partial = out / f"{name}.perfetto.partial.json.gz"
    final = out / f"{name}.perfetto.json.gz"
    with (out / f"{name}-analyze.log").open("w") as log:
        subprocess.run(
            [
                str(tool / "build/traceloom"),
                str(source),
                "--threads",
                "4",
                "--output",
                str(db),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=300,
        )
    with sqlite3.connect(db) as c:
        assert c.execute("pragma quick_check").fetchone()[0] == "ok"
        metadata = dict(c.execute("select key,value from traceloom_metadata"))
        assert metadata["source_path"] == str(source)
        # Established workaround for the frozen exporter's quadratic root lookup.
        # Index only the derived DB, never the provider's source DB.
        c.execute(
            "create index if not exists strengthen_viz_edge_child "
            "on traceloom_viz_edge(child_node_id)"
        )
    print(name, "analyzed", flush=True)
    with (out / f"{name}-export.log").open("w") as log:
        subprocess.run(
            [
                str(tool / "build/traceloom"),
                "export-perfetto",
                str(db),
                "--output",
                str(partial),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=600,
        )
    with gzip.open(partial, "rt") as f:
        trace = json.load(f)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    assert events and any(e.get("ph") == "X" for e in events)
    count = len(events)
    del trace, events
    partial.replace(final)
    manifest["files"].append(
        dict(
            arm=entry["arm"],
            rank=entry["rank"],
            source=str(source),
            analysis=str(db),
            timeline=str(final),
            events=count,
            bytes=final.stat().st_size,
        )
    )
    (out / "exports.json").write_text(json.dumps(manifest, indent=2))
    print(name, "exported", count, "events", final.stat().st_size, "bytes", flush=True)
