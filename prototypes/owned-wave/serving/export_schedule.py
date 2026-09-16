"""Add verified host scheduler lanes to a TraceLoom export without rewriting it.

Wall-clock containment of all captured replay APIs gates overlay. This associates
ordered steps, not an exact runtime/device replay reconstruction or cross-rank fit.
"""

import argparse
import gzip
import json
import sqlite3
from pathlib import Path
from inspect_step_gaps import inspect


def export(capsule, rank):
    out = capsule / "traceloom"
    schedule = json.loads(
        (capsule / f"engine/owned-profile/schedule-rank{rank}.json").read_text()
    )
    rows = schedule["events"]
    stages = {}
    for r in rows:
        if "sequence" in r:
            stages.setdefault(r["sequence"], {})[r["event"]] = r
    c = sqlite3.connect(f"file:{out/f'owned-rank{rank}.db'}?mode=ro", uri=True)
    replays = c.execute(
        "select a.startNs,a.endNs from CANN_API a join STRING_IDS s on s.id=a.name where s.value='aclmdlRIExecuteAsync' order by a.startNs"
    ).fetchall()
    matches = []
    for lo, hi in replays:
        matching = [
            seq
            for seq, s in stages.items()
            if s["submit_begin"]["wall_ns"] <= lo <= hi <= s["submit_end"]["wall_ns"]
        ]
        assert len(matching) == 1, (
            "provider replay not uniquely bracketed by recorded host submission",
            lo,
            hi,
            matching,
        )
        matches.append(matching[0])
    assert matches == list(
        range(
            schedule["warmup_steps"],
            schedule["warmup_steps"] + schedule["profile_steps"],
        )
    )
    offsets = [r["wall_ns"] - r["monotonic_ns"] for r in rows]
    assert max(offsets) - min(offsets) < 100_000, "host clock changed by >100us"
    trace = json.load(gzip.open(out / f"owned-rank{rank}.perfetto.json.gz", "rt"))
    origin = trace["metadata"]["time_origin_ns"]
    pid = 900 + rank
    events = [
        dict(
            ph="M",
            pid=pid,
            tid=0,
            name="process_name",
            args=dict(
                name=f"Owned scheduler rank{rank} · host observations, not device durations"
            ),
        )
    ]
    lanes = {
        1: "Host submit (native plan + ingress + replay enqueue)",
        2: "Host receive / D2H wait",
        3: "TP quorum retirement",
        4: "Authorized lifetime bank0",
        5: "Authorized lifetime bank1",
        6: "Profile drain / instrumentation",
        7: "Verified ordered device body (compute envelope)",
    }
    for tid, name in lanes.items():
        events.append(
            dict(ph="M", pid=pid, tid=tid, name="thread_name", args=dict(name=name))
        )

    def span(name, tid, lo, hi, args):
        assert hi >= lo
        events.append(
            dict(
                ph="X",
                pid=pid,
                tid=tid,
                ts=(lo - origin) / 1000,
                dur=(hi - lo) / 1000,
                name=name,
                cat="owned.schedule",
                args=args,
            )
        )

    selected = set(matches)
    for seq, s in stages.items():
        b = s["submit_begin"]
        args = {
            k: v for k, v in b.items() if k not in ("wall_ns", "monotonic_ns", "event")
        }
        args["device_profiled"] = seq in selected
        span(
            f"step{seq} {b['key']} submit",
            1,
            b["wall_ns"],
            s["submit_end"]["wall_ns"],
            args,
        )
        span(
            f"step{seq} receive",
            2,
            s["receive_begin"]["wall_ns"],
            s["receive_end"]["wall_ns"],
            args,
        )
        span(
            f"step{seq} TP quorum",
            3,
            s["receive_end"]["wall_ns"],
            s["quorum"]["wall_ns"],
            args,
        )
        span(
            f"step{seq} {b['key']} authorized",
            4 + seq % 2,
            b["wall_ns"],
            s["quorum"]["wall_ns"],
            args,
        )
    for start, end in [
        ("warmup_drain_begin", "warmup_drain_end"),
        ("profile_drain_begin", "profile_drain_end"),
    ]:
        a = next(r for r in rows if r["event"] == start)
        b = next(r for r in rows if r["event"] == end)
        span(
            start.removesuffix("_begin"),
            6,
            a["wall_ns"],
            b["wall_ns"],
            dict(scope="profiling artifact, not serving wait"),
        )
    waves = inspect(out / f"owned-rank{rank}.db")["waves"]
    for seq, w in zip(matches, waves, strict=True):
        span(
            f"step{seq} device body",
            7,
            w["start_ns"],
            w["end_ns"],
            dict(
                sequence=seq,
                model_id=w["model_id"],
                association="ordered bodies + replay API host bracket; not exact replay partition",
            ),
        )
    transport_path = out / "transport-audit.json"
    if transport_path.exists():
        transport = json.loads(transport_path.read_text())[rank]
        for tid, name in [
            (8, "H2D DMA · provider connectionId joined"),
            (9, "D2H DMA · provider connectionId joined"),
        ]:
            events.append(
                dict(ph="M", pid=pid, tid=tid, name="thread_name", args=dict(name=name))
            )
        for transfer in transport["transfers"]:
            span(
                f"step{transfer['sequence']} {transfer['direction']}",
                8 if transfer["direction"] == "H2D" else 9,
                transfer["start_ns"],
                transfer["end_ns"],
                transfer,
            )
    # Keep the original TraceLoom-only file; offer a separate all-host schedule
    # and an overlay restricted to sampled waves so warmup does not drown it out.
    full = dict(
        traceEvents=events,
        displayTimeUnit="ms",
        metadata=dict(
            time_origin_ns=origin,
            scope="complete bounded host scheduling; device bodies only in sampled window",
        ),
    )
    with gzip.open(out / f"schedule-rank{rank}.perfetto.json.gz", "wt") as f:
        json.dump(full, f)
    trace["traceEvents"].extend(
        e
        for e in events
        if e["ph"] == "M"
        or e.get("args", {}).get("sequence") in selected
        or e.get("tid") == 6
    )
    trace["metadata"][
        "scheduler_overlay"
    ] = "host wall times bracket every replay API; ordered device-body association; no cross-rank alignment"
    with gzip.open(out / f"owned-rank{rank}.with-schedule.perfetto.json.gz", "wt") as f:
        json.dump(trace, f)
    result = dict(
        rank=rank,
        status="PASS",
        sampled_sequences=matches,
        host_clock_offset_spread_ns=max(offsets) - min(offsets),
        waves=[],
    )
    for seq, w, (lo, hi) in zip(matches, waves, replays, strict=True):
        s = stages[seq]
        result["waves"].append(
            dict(
                sequence=seq,
                key=s["submit_begin"]["key"],
                lengths=s["submit_begin"]["lengths"],
                submit_ms=(s["submit_end"]["wall_ns"] - s["submit_begin"]["wall_ns"])
                / 1e6,
                receive_wait_ms=(
                    s["receive_end"]["wall_ns"] - s["receive_begin"]["wall_ns"]
                )
                / 1e6,
                quorum_ms=(s["quorum"]["wall_ns"] - s["receive_end"]["wall_ns"]) / 1e6,
                replay_lead_ms=(w["start_ns"] - hi) / 1e6,
            )
        )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("capsule", type=Path)
    a = p.parse_args()
    result = [export(a.capsule, rank) for rank in range(2)]
    (a.capsule / "traceloom/schedule-audit.json").write_text(
        json.dumps(result, indent=2)
    )
