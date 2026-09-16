"""Account for preparation/continuation on existing coordinator receipts.

No cross-device clock fit and no claim that every interval is removable idle.
Unlike pack->return, this includes the first FETCH of a coalesced wave.
"""

import argparse
import json
from pathlib import Path
import statistics


def analyze(receipt):
    slots = [[], []]
    waves = []
    for event in receipt["events"]:
        engine, kind, slot, begin, end, *_ = event
        assert slot in (0, 1) and begin <= end
        slots[slot].append(event)
        if (engine, kind) != (0, 4):
            continue
        batch = slots[slot]
        pulls = [e for e in batch if e[:2] == [0, 1]]
        packs = [e for e in batch if e[:2] == [0, 2]]
        ups = [e for e in batch if e[:2] == [1, 1]]
        downs = [e for e in batch if e[:2] == [1, 2]]
        assert pulls and len(packs) == len(ups) == len(downs) == 1
        first = min(e[3] for e in pulls)
        fetch_end = max(e[4] for e in pulls)
        pack, up, down = packs[0], ups[0], downs[0]
        assert first <= fetch_end <= pack[3] <= up[3]
        assert up[4] <= down[3] <= down[4] <= end
        waves.append(
            dict(
                first_fetch=first,
                returned=end,
                fetch_to_return_us=(end - first) / 50,
                post_fetch_preparation_us=(pack[3] - fetch_end) / 50,
                first_fetch_to_up_us=(up[3] - first) / 50,
                up_down_gap_us=(down[3] - up[4]) / 50,
                return_tail_us=(end - down[4]) / 50,
            )
        )
        slots[slot] = []
    assert all(not slot for slot in slots)
    waves.sort(key=lambda w: w["first_fetch"])
    result = {
        "waves": len(waves),
        "medians": {
            key: statistics.median(w[key] for w in waves)
            for key in waves[0]
            if key.endswith("_us")
        },
    }
    gaps = [(b["first_fetch"] - a["returned"]) / 50 for a, b in zip(waves, waves[1:])]
    result["next_fetch_signed_gap_median_us"] = statistics.median(gaps)
    result["overlapping_preparation_transitions"] = sum(g < 0 for g in gaps)
    # A negative gap is overlap, not an invalid timestamp or a negative latency.
    result["scope"] = "server-local coordinator intervals; not client or wire latency"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            [
                analyze(
                    json.loads(
                        (args.run / f"run/measurements/expert{i}.json").read_text()
                    )
                )
                for i in (0, 1)
            ],
            indent=2,
        )
    )
