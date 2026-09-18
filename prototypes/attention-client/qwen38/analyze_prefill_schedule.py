"""Count useful prefill rows from monotone encoded cursors, not output tokens."""

import argparse
import json
from pathlib import Path


def summarize(receipt, bucket_rows):
    previous = [0] * len(receipt["sessions"])
    lengths = []
    for wave in receipt["waves"]:
        delta = [b - a for a, b in zip(previous, wave["contexts"])]
        previous = wave["contexts"]
        if wave["phase"] == "prefill":
            if min(delta) < 0:
                raise ValueError("This analyzer requires retained monotone history")
            if "prefill_lengths" in wave and delta != wave["prefill_lengths"]:
                raise ValueError("Cursor and declared input rows disagree")
            lengths.append(delta)
    useful = sum(map(sum, lengths))
    assert useful == sum(s["prefill_tokens"] for s in receipt["sessions"])
    return dict(
        source=receipt["source"],
        prefill_waves=len(lengths),
        useful_rows=useful,
        executed_bucket_rows=bucket_rows * len(lengths),
        utilization=useful / (bucket_rows * len(lengths)),
        single_request_waves=sum(sum(x > 0 for x in row) == 1 for row in lengths),
        full_waves=sum(sum(row) == bucket_rows for row in lengths),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("roles", type=Path)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--sources", type=int, required=True)
    p.add_argument("--bucket-rows", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = [
        summarize(
            json.loads((a.roles / f"attention{i*a.tp_size}.json").read_text()),
            a.bucket_rows,
        )
        for i in range(a.sources)
    ]
    totals = {
        key: sum(r[key] for r in rows)
        for key in (
            "prefill_waves",
            "useful_rows",
            "executed_bucket_rows",
            "single_request_waves",
            "full_waves",
        )
    }
    totals["utilization"] = totals["useful_rows"] / totals["executed_bucket_rows"]
    a.output.write_text(
        json.dumps(
            dict(
                scope="retained trace scheduler row accounting, not operator FLOPs",
                sources=rows,
                total=totals,
            ),
            indent=2,
        )
        + "\n"
    )
