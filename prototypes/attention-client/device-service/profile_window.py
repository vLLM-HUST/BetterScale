"""Crop the sealed-catalog episode using recorded host envelopes and clock metadata."""

import copy
import gzip
import json
from pathlib import Path
import sys


def window(root):
    analysis = root / "analysis"
    with gzip.open(analysis / "attention2-expert2-provider-clock.json.gz", "rt") as f:
        full = json.load(f)
    offsets, envelopes = {}, []
    for role in ("attention0", "attention1", "expert0", "expert1"):
        profile = next((root / "profile").glob(f"{role}_*"))
        info = json.loads(next(profile.glob("profiler_info*.json")).read_text())
        end = info["end_info"]
        offsets[role] = end["collectionTimeEnd"] - end["MonotonicTimeEnd"]
    for role in ("attention0", "attention1"):
        record = json.loads((root / "run/measurements" / f"{role}.json").read_text())
        pending = {}
        for event in record["events"]:
            if event.get("generation", 0) < 13:
                continue
            generation = event["generation"]
            stamp = round(event["time"] * 1e9) + offsets[role]
            if event["event"] == "submit":
                pending[generation] = stamp
            elif event["event"] == "retire":
                envelopes.append((role, generation, pending.pop(generation), stamp))
        assert not pending
    assert len(envelopes) == 24
    origin = full["metadata"]["time_origin_ns"]
    lo = (min(e[2] for e in envelopes) - origin) / 1000 - 10000
    hi = (max(e[3] for e in envelopes) - origin) / 1000 + 2000
    events = []
    for original in full["traceEvents"]:
        event = copy.deepcopy(original)
        if event.get("ph") == "X":
            start, end = event["ts"], event["ts"] + event["dur"]
            if end <= lo or start >= hi:
                continue
            a, b = max(start, lo), min(end, hi)
            event["ts"], event["dur"] = a - lo, b - a
            event["args"]["window_clipped"] = a != start or b != end
        events.append(event)
    for i, role in enumerate(("attention0", "attention1")):
        events.append(
            dict(
                ph="M",
                name="thread_name",
                pid=121,
                tid=i,
                args=dict(name=f"{role} host submit→observed completion"),
            )
        )
    for role, generation, start, end in envelopes:
        events.append(
            dict(
                ph="X",
                name=f"expert request {generation}",
                pid=121,
                tid=int(role[-1]),
                ts=(start - origin) / 1000 - lo,
                dur=(end - start) / 1000,
                args=dict(source="host audit; includes host polling delay"),
            )
        )
    metadata = dict(
        full["metadata"],
        window="generations13–24; sealed catalog",
        provider_realtime_minus_monotonic_ns=offsets,
        clock_mapping_spread_ns=max(offsets.values()) - min(offsets.values()),
        window_start_offset_us=lo,
        window_duration_us=hi - lo,
        caveat="Host clock metadata is not an independent device synchronization measurement.",
    )
    path = analysis / "attention2-expert2-sealed-window.json.gz"
    with gzip.open(path, "wt", compresslevel=1) as f:
        json.dump(dict(traceEvents=events, metadata=metadata), f)
    (analysis / "window-receipt.json").write_text(json.dumps(metadata, indent=2))
    print(path)


if __name__ == "__main__":
    window(Path(sys.argv[1]).resolve())
