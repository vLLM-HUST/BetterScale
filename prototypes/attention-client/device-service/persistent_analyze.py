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
    parts = 2 if receipt.get("segmented") else 1
    internal = receipt.get("internal_pipeline", False)
    prefix_ready = {}
    if internal:
        cube_commands = [e for e in events if e[0] == 1]
        markers = {}
        for gen, core, timestamp in receipt["core_prefix"]:
            markers.setdefault(gen, {})[core] = timestamp
        for gen, cores in markers.items():
            assert set(cores) == set(range(24))
            event = cube_commands[gen - 1]
            assert event[1] == 1
            assert all(event[3] <= t <= event[4] for t in cores.values())
            prefix_ready[event[6]] = max(cores.values())
    assert len(events) <= (4 + 3 * parts) * 48
    origin = min(e[3] for e in events)
    stages = [[], []]
    wave_ids = {}
    slot_wave = [0, 0]
    costs = {}
    for engine, kind, slot, begin, end, rows, seq, part in events:
        assert 0 <= slot < 2 and end >= begin
        name = names[engine, kind]
        if parts == 2 and ((engine == 1 and not internal) or kind == 3):
            name += f".part{part}"
        stages[slot].append((engine, kind, begin, end, part, seq))
        wave_ids[seq] = (slot, slot_wave[slot])
        if (engine, kind) == (0, 4):
            slot_wave[slot] += 1
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
    wave_service_us, math_chain_us = [], []
    for sequence in stages:
        wave = []
        for event in sequence:
            wave.append(event)
            if event[:2] != (0, 4):
                continue
            pulls = [e for e in wave if e[:2] == (0, 1)]
            packs = [e for e in wave if e[:2] == (0, 2)]
            assert 1 <= len(pulls) <= 2 and len(packs) == 1
            pack = packs[0]
            assert max(e[3] for e in pulls) <= pack[2]
            assert len(wave) == len(pulls) + 2 + (parts + 2 if internal else 3 * parts)
            for part in range(parts):

                def unique(engine, kind):
                    found = [
                        e
                        for e in wave
                        if e[:2] == (engine, kind)
                        and e[4] == (0 if internal and engine == 1 else part)
                    ]
                    assert len(found) == 1, wave
                    return found[0]

                up, act, down = unique(1, 1), unique(0, 3), unique(1, 2)
                assert pack[3] <= up[2]
                if internal and part == 0:
                    assert prefix_ready[up[5]] <= act[2]
                else:
                    assert up[3] <= act[2]
                assert act[3] <= down[2] <= down[3] <= event[2]
            wave_service_us.append((event[3] - pack[2]) / 50)
            cube_events = [e for e in wave if e[0] == 1]
            math_chain_us.append(
                (max(e[3] for e in cube_events) - min(e[2] for e in cube_events)) / 50
            )
            wave = []
        assert not wave, wave
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
        preparation, compute, activation, up_compute = [], [], [], []
        by_wave = {}
        work_by_event = {}
        for engine, gen, core, begin, end in receipt["core_work"]:
            e = commands[engine, gen]
            assert e[3] <= begin <= end <= e[4], (e, begin, end)
            counts[engine, gen] = counts.get((engine, gen), 0) + 1
            work_by_event.setdefault(e[6], []).append((begin, end))
            wave_work = by_wave.setdefault(wave_ids[e[6]], {"up": [], "act": []})
            if engine == 1 and e[1] == 1:
                wave_work["up"].append((begin, end))
            elif engine == 0 and e[1] == 3:
                wave_work["act"].append((begin, end))
            if engine == 1:
                compute.append((begin, end))
                if e[1] == 1:
                    up_compute.append((begin, end))
            elif e[1] in (1, 2):
                preparation.append((begin, end))
            elif e[1] == 3:
                activation.append((begin, end))
            core_display.append(
                dict(
                    name=names[engine, e[1]]
                    + (
                        f".part{e[7]}"
                        if parts == 2 and ((engine == 1 and not internal) or e[1] == 3)
                        else ""
                    ),
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
    activation_overlap = None
    within_wave_overlap = None
    if "core_work" in receipt:
        activation_overlap = (
            sum(
                max(0, min(v[1], c[1]) - max(v[0], c[0]))
                for v in merge_intervals(activation)
                for c in merge_intervals(up_compute)
            )
            / 50
        )
        within_wave_overlap = (
            sum(
                max(0, min(v[1], c[1]) - max(v[0], c[0]))
                for work in by_wave.values()
                for v in merge_intervals(work["act"])
                for c in merge_intervals(work["up"])
            )
            / 50
        )
    publication_delay, remaining_up = [], []
    if internal:
        for seq, ready in prefix_ready.items():
            act = next(
                e
                for e in events
                if wave_ids[e[6]] == wave_ids[seq]
                and e[0] == 0
                and e[1] == 3
                and e[7] == 0
            )
            first_consumer = min(b for b, end in work_by_event[act[6]])
            assert first_consumer >= ready
            publication_delay.append((first_consumer - ready) / 50)
            remaining_up.append(
                (max(end for b, end in work_by_event[seq]) - ready) / 50
            )
    summaries.append(
        dict(
            server=server,
            segmented=parts == 2,
            internal_pipeline=internal,
            prefix_to_activation_median_us=(
                statistics.median(publication_delay) if publication_delay else None
            ),
            up_remaining_after_prefix_median_us=(
                statistics.median(remaining_up) if remaining_up else None
            ),
            tail_experts=receipt.get("tail_experts", 0),
            median_pack_to_return_us=statistics.median(wave_service_us),
            median_math_chain_us=statistics.median(math_chain_us),
            activation_up_core_overlap_us=activation_overlap,
            same_wave_activation_up_core_overlap_us=within_wave_overlap,
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
