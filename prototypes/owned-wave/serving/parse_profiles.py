"""Fresh-process official CANN/msprof + torch-npu offline exports, rank checked."""

import argparse
import gzip
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("capsule", type=Path)
a = p.parse_args()
manifest = []
config = json.loads((a.capsule / "engine" / "config.json").read_text())
complete = json.loads((a.capsule / "engine" / "complete.json").read_text())
arms = ["native", "owned"] if complete["arm"] == "both" else [complete["arm"]]
for arm in arms:
    for rank in range(config["tensor_parallel_size"]):
        roots = list(
            (a.capsule / "engine" / f"{arm}-profile" / "profile").glob(f"rank{rank}_*")
        )
        assert len(roots) == 1, (arm, rank, roots)
        root = roots[0]
        output = root / "ASCEND_PROFILER_OUTPUT"
        assert not output.exists(), f"preserve prior offline parse: {output}"
        with (a.capsule / f"parse-{arm}-rank{rank}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    'from torch_npu.profiler.profiler import analyse; import sys; analyse(sys.argv[1], max_process_number=1, export_type=["db","text"])',
                    str(root),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=240,
                check=True,
            )
        dbs = list(output.glob("ascend_pytorch_profiler_*.db"))
        assert (
            len(dbs) == 1 and dbs[0].name == f"ascend_pytorch_profiler_{rank}.db"
        ), dbs
        with sqlite3.connect(f"file:{dbs[0]}?mode=ro", uri=True) as c:
            assert c.execute("pragma quick_check").fetchone()[0] == "ok"
            mapping = c.execute("select * from RANK_DEVICE_MAP").fetchall()
            assert mapping and all(row[0] == rank for row in mapping), mapping
            tables = [
                row[0]
                for row in c.execute(
                    "select name from sqlite_master where type='table'"
                )
            ]
        trace = output / "trace_view.json"
        assert trace.exists() and trace.stat().st_size > 0
        # Validate the official export before offering a compressed viewer artifact.
        with trace.open() as f:
            json.load(f)
        compressed = trace.with_suffix(".json.gz")
        with (
            trace.open("rb") as src,
            gzip.open(compressed, "wb", compresslevel=1) as dst,
        ):
            shutil.copyfileobj(src, dst)
        native = list(root.glob("PROF_*"))
        assert native, root
        manifest.append(
            dict(
                arm=arm,
                rank=rank,
                rank_device_map=mapping,
                db=str(dbs[0]),
                native_prof=[str(x) for x in native],
                timeline=str(compressed),
                tables=tables,
            )
        )
        print(arm, rank, "parsed", flush=True)
(a.capsule / "profile-exports.json").write_text(json.dumps(manifest, indent=2))
