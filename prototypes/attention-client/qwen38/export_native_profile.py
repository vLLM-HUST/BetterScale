"""Offline native graph evidence export; never turn Torch missing metadata into eager.

Uses the Qwen27 native_graph_export.py recipe: copy the raw PROF, export with
msprof, then feed native DBs to TraceLoom. Captures and timing runs stay immutable.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path, help="one raw PROF_* directory")
    parser.add_argument("output", type=Path, help="new analysis directory")
    parser.add_argument("--tool", type=Path, required=True)
    args = parser.parse_args()
    if not args.profile.is_dir() or not args.profile.name.startswith("PROF_"):
        raise ValueError("select exactly one raw PROF directory")
    args.output.mkdir(parents=True, exist_ok=False)
    copied = args.output / args.profile.name
    shutil.copytree(args.profile, copied)
    commands = [
        ["msprof", "--export=on", "--type=db", f"--output={copied}"],
        [
            str(args.tool),
            str(copied),
            "--classification-rules",
            str(args.tool.parent / "default_signal_classification_rules.tsv"),
            "--symbol-rules",
            str(args.tool.parent / "default_structural_symbol_rules.tsv"),
            "--event-reconciliation-rules",
            str(args.tool.parent / "default_event_reconciliation_rules.tsv"),
            "--threads",
            "2",
            "--output",
            str(args.output / "analysis.db"),
            "--perfetto-out",
            str(args.output / "timeline.json.gz"),
        ],
    ]
    for index, command in enumerate(commands):
        with (args.output / f"stage{index}.log").open("w") as log:
            subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600
            )
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(
                raw_source=str(args.profile), commands=commands, clock="single-device"
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
