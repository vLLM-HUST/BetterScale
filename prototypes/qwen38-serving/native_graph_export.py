"""Preserve CaptureStreamInfo by feeding TraceLoom native msprof DBs, not Torch DB."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("capsule", type=Path)
p.add_argument("--label", choices=["none", "full"])
a = p.parse_args()
root = a.capsule.resolve()
out = root / ("native-graph-" + a.label if a.label else "native-graph")
profile_root = root / "profiles"
if a.label:
    profile_root /= a.label
out.mkdir(exist_ok=False)
entries = []
for rank in [0, 1]:
    sources = list((profile_root / "profile").glob(f"rank{rank}_*/PROF_*"))
    if len(sources) != 1:
        raise RuntimeError(sources)
    dest = out / f"rank{rank}" / sources[0].name
    dest.parent.mkdir()
    shutil.copytree(sources[0], dest)
    with (out / f"export-rank{rank}.log").open("w") as log:
        subprocess.run(
            ["msprof", "--export=on", "--type=db", f"--output={dest}"],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=240,
        )
    entries.append(
        dict(
            arm=root.name + ("-" + a.label if a.label else ""),
            rank=rank,
            db=str(dest),
            raw_source=str(sources[0]),
        )
    )
(out / "profile-exports.json").write_text(json.dumps(entries, indent=2))
subprocess.run(
    [
        sys.executable,
        str(Path(__file__).with_name("export_timeline.py")),
        str(out),
    ],
    check=True,
)
