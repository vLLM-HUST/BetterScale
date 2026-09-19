"""Short native msprof -> TraceLoom diagnostics, separate from step timings."""

import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess

from compare_full_prefill import inspect


def main():
    p = argparse.ArgumentParser()
    p.add_argument("round", type=Path)
    p.add_argument("--traceloom", type=Path, required=True)
    a = p.parse_args()
    out = a.round / "traceloom"
    out.mkdir(exist_ok=True)
    result = dict(scope="Six separately profiled steps per rank, not performance curve samples. Kernel-body boundaries use semantic guards; gaps include sampling, preparation and waits, not just idle.", ranks=[])
    for rank in (0, 1):
        sources = list((a.round / "profiles/profile").glob(f"rank{rank}_*/PROF_*"))
        assert len(sources) == 1, sources
        dest = a.round / "native-graph" / f"rank{rank}" / sources[0].name
        if not dest.exists():
            shutil.copytree(sources[0], dest)
        stem = f"{a.round.name}-rank{rank}"
        if not list(dest.glob("msprof_*.db")):
            with (out / f"{stem}-msprof.log").open("w") as log:
                subprocess.run(["msprof", "--export=on", "--type=db", f"--output={dest}"],
                               stdout=log, stderr=subprocess.STDOUT, timeout=240, check=True)
        db = out / f"{stem}.db"
        if not db.exists():
            with (out / f"{stem}-analyze.log").open("w") as log:
                subprocess.run([str(a.traceloom), str(dest), "--threads", "4", "--output", str(db),
                                "--loop-tree-out", str(out / f"{stem}.md")],
                               stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
        with (out / f"{stem}-export.log").open("w") as log:
            subprocess.run([str(a.traceloom), "export-perfetto", str(db), "--output", str(out / f"{stem}.perfetto.json.gz")],
                           stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True)
        schedule = json.loads((a.round / f"profiles/schedule-rank{rank}.json").read_text())
        dispatch = [r for r in schedule["events"] if r["event"] == "dispatch"]
        data = inspect(db, matmul_type=("MatMulV2", "MatMulV3"), matmul_count=304,
                       synchronous=False, expected_steps=6)
        assert len(dispatch) == len(data["steps"])
        data.update(rank=rank, dispatch=dispatch)
        for i, step in enumerate(data["steps"][:-1]):
            next_step = data["steps"][i + 1]
            step["body_start_period_ms"] = (next_step["start_ns"] - step["start_ns"]) / 1e6
            step["body_to_next_body_ms"] = (next_step["start_ns"] - step["end_ns"]) / 1e6
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
            assert c.execute("pragma quick_check").fetchone()[0] == "ok"
            data["api_counts"] = c.execute("""select s.value,count(*) from CANN_API a
                join STRING_IDS s on s.id=a.name where s.value like '%FusedInferAttention%'
                or s.value like '%Update%' or s.value='aclmdlRIExecuteAsync'
                group by s.value order by s.value""").fetchall()
        result["ranks"].append(data)
        (out / "profile-summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(stem, "PASS", [r["scheduled"] for r in dispatch], flush=True)


if __name__ == "__main__":
    main()
