"""Official offline export followed by the retained TraceLoom pipeline."""

import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("capsule", type=Path)
a = p.parse_args()
root = a.capsule.resolve()
entries = []
for rank in [0, 1]:
    dirs = list((root / "profiles/profile").glob(f"rank{rank}_*"))
    if len(dirs) != 1:
        raise RuntimeError(f"expected one profile for rank{rank}: {dirs}")
    directory = dirs[0]
    output = directory / "ASCEND_PROFILER_OUTPUT"
    if output.exists():
        raise RuntimeError(f"preserve existing parse: {output}")
    with (root / f"parse-rank{rank}.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-c",
                'from torch_npu.profiler.profiler import analyse; import sys; analyse(sys.argv[1], max_process_number=1, export_type=["db","text"])',
                str(directory),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=240,
        )
    dbs = list(output.glob("ascend_pytorch_profiler_*.db"))
    if len(dbs) != 1:
        raise RuntimeError(f"expected one integrated provider DB: {dbs}")
    with sqlite3.connect(f"file:{dbs[0]}?mode=ro", uri=True) as c:
        if c.execute("pragma quick_check").fetchone()[0] != "ok":
            raise RuntimeError("provider DB corrupt")
        mapping = c.execute("select * from RANK_DEVICE_MAP").fetchall()
    entries.append(
        dict(arm=root.name, rank=rank, db=str(dbs[0]), rank_device_map=mapping)
    )
    print("parsed rank", rank, flush=True)
(root / "profile-exports.json").write_text(json.dumps(entries, indent=2))
subprocess.run(
    [
        sys.executable,
        "/workspace/strengthen-dsv4/prototypes/owned-wave/serving/export_traceloom.py",
        str(root),
    ],
    check=True,
)
