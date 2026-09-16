"""Lower-bound ready expert work using real pack completion and safe phase joins.

Only the deterministic balanced burst fixture is admitted: reconstruct its routes
and validate every recorded destination/provenance. Never interpret a Cube routine
start as a hardware MMAD issue or a call return as completed GEMM output.
"""

import argparse
import collections
import gzip
import json
import math
import statistics
from pathlib import Path


def merged(intervals):
    out = []
    for begin, end in sorted(intervals):
        if end <= begin:
            continue
        if out and begin <= out[-1][1]:
            out[-1][1] = max(end, out[-1][1])
        else:
            out.append([begin, end])
    return out


def coverage(intervals, gaps):
    return (
        sum(
            max(0, min(b, y) - max(a, x)) for a, b in merged(intervals) for x, y in gaps
        )
        / 50
    )


def audit(root, server, timeline):
    receipt = json.loads((root / f"run/measurements/expert{server}.json").read_text())
    assert receipt["early_down"] and receipt["internal_pipeline"]
    assert not receipt["tail_experts"] and not receipt["resident_moves"]
    clients = [
        json.loads((root / f"run/measurements/client{c}.json").read_text())
        for c in range(2)
    ]
    events = receipt["events"]
    vector = [e for e in events if e[0] == 0]
    cube = [e for e in events if e[0] == 1]
    marks = collections.defaultdict(dict)
    for gen, row, timestamp, c, sourcegen, route, layer, worker in receipt[
        "pack_row_ready"
    ]:
        e = vector[gen - 1]
        assert e[1] == 2 and e[3] <= timestamp <= e[4]
        assert row not in marks[e[6]]
        marks[e[6]][row] = (timestamp, c, sourcegen, route, layer, worker)
    core = collections.defaultdict(dict)
    for engine, gen, cid, begin, end in receipt["core_work"]:
        if engine == 1:
            core[cube[gen - 1][6]][cid] = (begin, end)
    assert all(set(v) == set(range(24)) for v in core.values())
    spans = [
        (min(b for b, e in core[c[6]].values()), max(e for b, e in core[c[6]].values()))
        for c in cube
    ]
    gaps = [(a[1], b[0]) for a, b in zip(spans, spans[1:])]
    assert all(b >= a for a, b in gaps)
    origin = min(e[3] for e in events)
    slots = [[], []]
    results, ready_up, ready_down = [], [], []
    waveid = 0
    for event in events:
        slots[event[2]].append(event)
        if event[:2] != [0, 4]:
            continue
        w = slots[event[2]]
        trace = receipt["trace"][waveid]
        assert trace[3] == event[2]
        expected = []
        for c in range(2):
            gen = trace[c]
            if not gen:
                continue
            fixture = clients[c][gen - 1]
            assert fixture["pattern"] == "balanced" and fixture["repeat"] == gen - 1
            layer = fixture["layer"]
            for route in range(fixture["rows_per_source"] * 8):
                expert = (route + c * 8 + (gen - 1) * 8) % 128
                if expert // 64 == server:
                    expected.append((layer * 64 + expert % 64, c, gen, route, layer))
        expected.sort()
        pack = next(e for e in w if e[:2] == [0, 2])
        up = next(e for e in w if e[:2] == [1, 1])
        down = next(e for e in w if e[:2] == [1, 2])
        acts = {e[7]: e for e in w if e[1] == 3}
        start_up = min(b for b, e in core[up[6]].values())
        start_down = min(b for b, e in core[down[6]].values())
        assert len(expected) == trace[2] == len(marks[pack[6]])
        grouped = collections.defaultdict(list)
        for row, (g, c, gen, route, layer) in enumerate(expected):
            t, mc, mg, mr, ml, worker = marks[pack[6]][row]
            assert (c, gen, route, layer) == (mc, mg, mr, ml)
            assert worker == route % 16
            assert t <= start_up
            grouped[g].append(t)
        count, boundary = 0, None
        for g, times in sorted(grouped.items()):
            count += len(times)
            if boundary is None and count >= (len(expected) + 1) // 2:
                boundary = g
        inputs = []
        for g, times in sorted(grouped.items()):
            input_ready = max(times)
            # Whole-expert split: completed activation segment is a safe upper
            # bound on when each member expert's down inputs became ready.
            act_ready = acts[int(g > boundary)][4]
            inputs.append(input_ready)
            ready_up.append((input_ready, start_up))
            if act_ready < start_down:
                ready_down.append((act_ready, start_down))
            for name, a, b in [
                ("up inputs ready", input_ready, start_up),
                ("down inputs proven ready", act_ready, start_down),
            ]:
                if a < b:
                    timeline.append(
                        dict(
                            name=name,
                            ph="X",
                            pid=server,
                            tid=g,
                            ts=(a - origin) / 50,
                            dur=(b - a) / 50,
                            args=dict(
                                wave=waveid,
                                expert_group=g,
                                routed_rows=len(times),
                                scope="ready until earliest core routine start; not MMAD scheduling",
                            ),
                        )
                    )
        inputs.sort()
        results.append(
            dict(
                wave=waveid,
                experts=len(grouped),
                routed_rows=len(expected),
                up_tiles=sum(math.ceil(len(t) / 128) * 6 for t in grouped.values()),
                first_ready_lead_us=(start_up - inputs[0]) / 50,
                half_ready_lead_us=(start_up - inputs[(len(inputs) - 1) // 2]) / 50,
                all_ready_lead_us=(start_up - inputs[-1]) / 50,
                pack_duration_us=(pack[4] - pack[3]) / 50,
                prefix_down_ready_while_up_us=max(0, up[4] - acts[0][4]) / 50,
            )
        )
        slots[event[2]] = []
        waveid += 1
    assert waveid == receipt["waves"]
    for i, (begin, end) in enumerate(spans):
        timeline.append(
            dict(
                name="Cube routine team envelope",
                ph="X",
                pid=server,
                tid=200,
                ts=(begin - origin) / 50,
                dur=(end - begin) / 50,
                args=dict(command=i + 1, scope="includes waits; not MMAD utilization"),
            )
        )
    return dict(
        server=server,
        waves=results,
        gap_total_us=sum(b - a for a, b in gaps) / 50,
        gap_with_ready_up_us=coverage(ready_up, gaps),
        gap_with_proven_ready_down_us=coverage(ready_down, gaps),
        gap_with_either_ready_us=coverage(ready_up + ready_down, gaps),
        medians={
            k: statistics.median(r[k] for r in results)
            for k in [
                "experts",
                "routed_rows",
                "up_tiles",
                "first_ready_lead_us",
                "half_ready_lead_us",
                "all_ready_lead_us",
                "pack_duration_us",
                "prefix_down_ready_while_up_us",
            ]
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    timeline = []
    results = [audit(args.run, server, timeline) for server in range(2)]
    out = args.run / "analysis"
    out.mkdir(exist_ok=True)
    (out / "expert-ready.json").write_text(json.dumps(results, indent=2))
    with gzip.open(out / "expert-ready-relative.json.gz", "wt") as f:
        json.dump(dict(traceEvents=timeline), f)
    print(
        json.dumps(
            [{k: v for k, v in r.items() if k != "waves"} for r in results], indent=2
        )
    )
