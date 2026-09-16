"""Audit persistent slot ordering and export coordinator-observed phase intervals.

Each expert device has an independent zero origin. This is NOT distributed clock
alignment. Durations use the A2 timer's50 cycles/us, as in the pinned DFC provider.
Intervals include publication/observation overhead, not just math instructions.
"""

import argparse
import gzip
import json
from pathlib import Path
import statistics

p = argparse.ArgumentParser()
p.add_argument("run", type=Path)
a = p.parse_args()
names = {
    (0, 1): "pull",
    (0, 2): "pack",
    (0, 3): "swiglu",
    (0, 4): "return",
    (1, 1): "up",
    (1, 2): "down",
}


def merge_intervals(intervals):
    merged = []
    for begin, end in sorted(intervals):
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([begin, end])
    return merged


display = []
core_display = []
summaries = []
for server in range(2):
    receipt = json.loads((a.run / f"run/measurements/expert{server}.json").read_text())
    events = receipt["events"]
    assert len(events) <= 336  # at most7 stages *48 single-source waves
    origin = min(e[3] for e in events)
    stages = [[], []]
    costs = {}
    for engine, kind, slot, begin, end, rows, seq, _ in events:
        assert 0 <= slot < 2 and end >= begin
        name = names[engine, kind]
        stages[slot].append(name)
        costs.setdefault(name, []).append((end - begin) / 50)
        display.append(
            dict(
                name=name,
                ph="X",
                pid=server,
                tid=engine,
                ts=(begin - origin) / 50,
                dur=(end - begin) / 50,
                args=dict(
                    slot=slot,
                    rows=rows,
                    sequence=seq,
                    scope="coordinator observed; per-device relative clock",
                ),
            )
        )
    for sequence in stages:
        cursor = 0
        while cursor < len(sequence):
            assert sequence[cursor] == "pull", sequence
            cursor += 1
            if cursor < len(sequence) and sequence[cursor] == "pull":
                cursor += 1  # second source joined after first payload arrived
            assert sequence[cursor : cursor + 5] == [
                "pack",
                "up",
                "swiglu",
                "down",
                "return",
            ], sequence
            cursor += 5
    for engine in range(2):
        ordered = sorted((e[3], e[4]) for e in events if e[0] == engine)
        assert all(y[0] >= x[1] for x, y in zip(ordered, ordered[1:]))
    overlap_ticks = sum(
        max(0, min(v[4], c[4]) - max(v[3], c[3]))
        for v in events
        if v[0] == 0 and v[1] in (1, 2)
        for c in events
        if c[0] == 1
    )
    core_overlap = None
    if "core_work" in receipt:
        commands = {}
        generation = [0, 0]
        for e in events:
            engine = e[0]
            generation[engine] += 1
            commands[engine, generation[engine]] = e
        counts = {}
        preparation, compute = [], []
        for engine, gen, core, begin, end in receipt["core_work"]:
            e = commands[engine, gen]
            assert e[3] <= begin <= end <= e[4], (e, begin, end)
            counts[engine, gen] = counts.get((engine, gen), 0) + 1
            if engine == 1:
                compute.append((begin, end))
            elif e[1] in (1, 2):
                preparation.append((begin, end))
            core_display.append(
                dict(
                    name=names[engine, e[1]],
                    ph="X",
                    pid=server,
                    tid=engine * 100 + core,
                    ts=(begin - origin) / 50,
                    dur=(end - begin) / 50,
                    args=dict(
                        slot=e[2],
                        generation=gen,
                        scope="per-core routine; no mailbox wait",
                    ),
                )
            )
        assert all(counts[key] == (24 if key[0] else 16) for key in commands)
        core_overlap = (
            sum(
                max(0, min(v[1], c[1]) - max(v[0], c[0]))
                for v in merge_intervals(preparation)
                for c in merge_intervals(compute)
            )
            / 50
        )
    summaries.append(
        dict(
            server=server,
            waves=receipt["waves"],
            paired=receipt["same_graph_cross_source_waves"],
            preparation_cube_overlap_us=overlap_ticks / 50,
            measured_core_routine_overlap_us=core_overlap,
            medians_us={k: statistics.median(v) for k, v in costs.items()},
            active_envelope_us=(max(e[4] for e in events) - origin) / 50,
        )
    )
out = a.run / "analysis"
out.mkdir(exist_ok=True)
(out / "persistent-phase-costs.json").write_text(json.dumps(summaries, indent=2))
with gzip.open(out / "persistent-phases-relative.json.gz", "wt") as f:
    json.dump(dict(traceEvents=display), f)
print(json.dumps(summaries, indent=2))

if core_display:
    with gzip.open(out / "persistent-work-relative.json.gz", "wt") as f:
        json.dump(dict(traceEvents=core_display), f)
