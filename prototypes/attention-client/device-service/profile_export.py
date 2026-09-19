"""Parse four role profiles and export native TraceLoom event lanes.

No synthetic clock fit: the optional provider-clock view restores the exact
source timestamps retained by TraceLoom, rather than first-event normalization.
It is not independent cross-device clock calibration.
"""

import argparse
import gzip
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

TOOL = Path("/workspace/my-ascend-workspace/.tools/traceloom-37323af/build/traceloom")
ROLES = ("attention0", "attention1", "expert0", "expert1")


def run(command, log, timeout=300):
    with log.open("w") as stream:
        subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )


def main(
    root,
    roles=ROLES,
    label="attention2-expert2",
    scope="two dummy layers, includes native oracle and startup; not throughput",
):
    assert len(set(roles)) == len(roles) and len(roles) >= 2
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    manifest = []
    for rank, role in enumerate(roles):
        paths = list((root / "profile").glob(f"{role}_*"))
        assert len(paths) == 1, (role, paths)
        profile = paths[0]
        databases = list((profile / "ASCEND_PROFILER_OUTPUT").glob("*.db"))
        if not databases:
            code = (
                "from torch_npu.profiler.profiler import analyse; import sys; "
                "analyse(sys.argv[1],max_process_number=1,export_type='db')"
            )
            run(
                [sys.executable, "-c", code, str(profile)], output / f"{role}-parse.log"
            )
            databases = list(
                (profile / "ASCEND_PROFILER_OUTPUT").glob("ascend_pytorch_profiler*.db")
            )
        assert len(databases) == 1, databases
        source = databases[0]
        with sqlite3.connect(source) as db:
            assert db.execute("pragma quick_check").fetchone()[0] == "ok"
            mapping = db.execute("select * from RANK_DEVICE_MAP").fetchall()
        target = output / f"{role}.db"
        if not target.exists():
            run(
                [str(TOOL), str(source), "--threads", "4", "--output", str(target)],
                output / f"{role}-analyze.log",
            )
        with sqlite3.connect(target) as db:
            db.execute(
                "create index if not exists strengthen_viz_edge_child "
                "on traceloom_viz_edge(child_node_id)"
            )
            db.commit()
        manifest.append(
            dict(
                display_rank=rank,
                role=role,
                provider_mapping=mapping,
                source=str(source),
                analysis=str(target),
            )
        )
        print(f"analyzed {role}", flush=True)
    partial = output / f"{label}-normalized.partial.json.gz"
    command = [
        str(TOOL),
        "export-perfetto",
        manifest[0]["analysis"],
        "--output",
        str(partial),
    ]
    for item in manifest:
        command.extend(
            ["--distributed-rank", f'{item["display_rank"]}={item["analysis"]}']
        )
    run(command, output / "export.log", timeout=600)
    with gzip.open(partial, "rt") as stream:
        exported = json.load(stream)
    partial.rename(output / f"{label}-normalized.json.gz")
    # Native exporter already constructs/labels the event hierarchy. Retain its
    # four distributed lanes, changing ONLY their display translation back to
    # provider timestamps; do not invent collective matches for point-to-point IPC.
    events = [e for e in exported["traceEvents"] if e.get("pid") == 120]
    slices = [e for e in events if e.get("ph") == "X"]
    assert {e["args"]["rank"] for e in slices} == set(range(len(roles)))
    origin = min(e["args"]["source_start_ns"] for e in slices)
    for event in events:
        if event.get("ph") == "X":
            args = event["args"]
            event["ts"] = (args["source_start_ns"] - origin) / 1000
            event["dur"] = (args["source_end_ns"] - args["source_start_ns"]) / 1000
            args["alignment"] = "provider_timestamps_no_additional_calibration"
        elif event.get("name") == "process_name":
            event["args"]["name"] = "Device expert service · provider clock"
        elif event.get("name") == "thread_name":
            event["args"]["name"] = roles[event["tid"] - 1]
    receipt = dict(
        source=str(root),
        roles=manifest,
        time_origin_ns=origin,
        alignment="native provider timestamps; no independently fitted clock",
        scope=scope,
        event_count=len(slices),
    )
    with gzip.open(
        output / f"{label}-provider-clock.json.gz", "wt", compresslevel=1
    ) as stream:
        json.dump(dict(traceEvents=events, metadata=receipt), stream)
    (output / "profile-receipt.json").write_text(json.dumps(receipt, indent=2))
    print(output / f"{label}-provider-clock.json.gz", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--dfc", action="store_true")
    parser.add_argument("--roles", nargs="+", default=ROLES)
    parser.add_argument("--label", default="attention2-expert2")
    parser.add_argument(
        "--scope",
        default="two dummy layers, includes native oracle and startup; not throughput",
    )
    args = parser.parse_args()
    if args.dfc:
        args.roles, args.label, args.scope = (
            ("dfc0", "dfc1"),
            "dfc2",
            "warm DFC control; not throughput",
        )
    main(args.root.resolve(), tuple(args.roles), args.label, args.scope)
