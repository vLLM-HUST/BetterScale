"""Attribute gaps between measured Cube routines, not MMAD hardware idle.

Readiness is the coordinator-observed prerequisite of the NEXT executed command.
This is not a counterfactual scheduler simulation or proof no partial tiles exist.
"""

import argparse
import json
import statistics
from pathlib import Path


def audit(path):
    receipt = json.loads(path.read_text())
    assert receipt["internal_pipeline"] and receipt["early_down"]
    events = receipt["events"]
    cube = [e for e in events if e[0] == 1]
    cores = {}
    for engine, gen, core, begin, end in receipt["core_work"]:
        if engine == 1:
            cores.setdefault(gen, {})[core] = (begin, end)
    assert len(cores) == len(cube)
    for gen, work in cores.items():
        assert set(work) == set(range(24))
        assert all(
            cube[gen - 1][3] <= b <= e <= cube[gen - 1][4] for b, e in work.values()
        )
    readiness = {}
    slots = [[], []]
    waves = []
    for event in events:
        slots[event[2]].append(event)
        if event[:2] != [0, 4]:
            continue
        wave = slots[event[2]]
        up = next(e for e in wave if e[:2] == [1, 1])
        down = next(e for e in wave if e[:2] == [1, 2])
        pack = next(e for e in wave if e[:2] == [0, 2])
        prefix = next(e for e in wave if e[1] == 3 and e[7] == 0)
        readiness[up[6]] = pack[4]
        readiness[down[6]] = max(up[4], prefix[4])
        waves.append(wave)
        slots[event[2]] = []
    assert not any(slots)
    gaps = {}
    for i in range(1, len(cube)):
        prev, nxt = cube[i - 1 : i + 1]
        end = max(e for b, e in cores[i].values())
        start = min(b for b, e in cores[i + 1].values())
        assert start >= end
        ready = readiness[nxt[6]]
        assert ready <= nxt[3] <= start
        before = max(0, min(start, ready) - end)
        after = start - end - before
        label = f"{'up' if prev[1] == 1 else 'down'}->{'up' if nxt[1] == 1 else 'down'}"
        gaps.setdefault(label, []).append(
            dict(
                gap_us=(start - end) / 50,
                before_next_ready_us=before / 50,
                after_next_ready_us=after / 50,
            )
        )
    summary = {}
    for label, records in gaps.items():
        summary[label] = dict(
            count=len(records),
            median_gap_us=statistics.median(r["gap_us"] for r in records),
            total_gap_us=sum(r["gap_us"] for r in records),
            total_before_next_ready_us=sum(r["before_next_ready_us"] for r in records),
            total_after_next_ready_us=sum(r["after_next_ready_us"] for r in records),
        )
    paired_turnaround = None
    # Only a sequential fully paired cohort gives this simple adjacent-wave
    # decomposition. Heterogeneous waves can overlap and must not use it.
    if receipt["same_graph_cross_source_waves"] == receipt["waves"]:
        rows = []
        for prev, nxt in zip(waves, waves[1:]):
            down = next(e for e in prev if e[:2] == [1, 2])
            send = next(e for e in prev if e[:2] == [0, 4])
            pull = next(e for e in nxt if e[:2] == [0, 1])
            pack = next(e for e in nxt if e[:2] == [0, 2])
            up = next(e for e in nxt if e[:2] == [1, 1])
            times = [down[4], send[4], pull[3], pack[4], up[3]]
            assert times == sorted(times)
            rows.append([(b - a) / 50 for a, b in zip(times, times[1:])])
        paired_turnaround = {
            name: statistics.median(r[i] for r in rows)
            for i, name in enumerate(
                [
                    "down_to_send_done_us",
                    "send_done_to_next_pull_us",
                    "next_pull_to_pack_done_us",
                    "pack_done_to_up_command_us",
                ]
            )
        }
    return dict(
        source=str(path),
        transitions=summary,
        paired_turnaround_medians=paired_turnaround,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    result = [
        audit(p)
        for root in args.runs
        for p in sorted((root / "run/measurements").glob("expert*.json"))
    ]
    assert result
    print(json.dumps(result, indent=2))
