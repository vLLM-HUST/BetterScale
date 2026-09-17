"""Summarize matched retained-prefix receipts, counting TP groups only once."""

import argparse
import json
import math
from pathlib import Path


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[max(0, math.ceil(q * len(values)) - 1)]


def summarize(directory, groups):
    rows = [
        json.loads((directory / f"attention{i * 2}.json").read_text())
        for i in range(groups)
    ]
    peers = [
        json.loads((directory / f"attention{i * 2 + 1}.json").read_text())
        for i in range(groups)
    ]
    if not all(
        r["status"] == "PASS" and r["prefix_first_page_exact"] and r["tp_output_exact"]
        for r in rows + peers
    ):
        raise ValueError("failed runtime gate")
    for a, b in zip(rows, peers):
        if a["sessions"] != b["sessions"]:
            # Timestamp jitter is expected; workload and accounting are not.
            fields = (
                "trace_id",
                "completed_turns",
                "output_tokens",
                "prefill_tokens",
                "prefix_reused_tokens",
                "final_encoded_context",
            )
            if [[s[k] for k in fields] for s in a["sessions"]] != [
                [s[k] for k in fields] for s in b["sessions"]
            ]:
                raise ValueError("TP accounting disagreement")
    sessions = sorted(
        [s for r in rows for s in r["sessions"]], key=lambda s: s["trace_id"]
    )
    elapsed = max(r["seconds"] for r in rows)
    outputs = sum(s["output_tokens"] for s in sessions)
    intervals = [x for r in rows for x in r["output_intervals_seconds"]]
    ttft = [e["ttft_seconds"] for s in sessions for e in s["events"]]
    return dict(
        groups=groups,
        mtp_tokens=rows[0]["mtp_tokens"],
        sessions=sessions,
        max_source_seconds=elapsed,
        committed_tokens=outputs,
        committed_tokens_per_second=outputs / elapsed,
        prefill_tokens=sum(s["prefill_tokens"] for s in sessions),
        reused_prefix_tokens=sum(s["prefix_reused_tokens"] for s in sessions),
        ttft_p50_seconds=quantile(ttft, 0.5),
        ttft_p95_seconds=quantile(ttft, 0.95),
        output_interval_p99_seconds=quantile(intervals, 0.99),
        output_interval_max_seconds=max(intervals, default=0),
        max_allocated_gib=max(r["memory"]["peak_allocated"] for r in rows + peers)
        / 2**30,
        max_reserved_gib=max(r["memory"]["reserved"] for r in rows + peers) / 2**30,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("separated", type=Path)
    p.add_argument("colocated", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    sep = summarize(a.separated, 2)
    col = summarize(a.colocated, 4)
    signature = lambda d: [
        (
            s["trace_id"],
            s["completed_turns"],
            s["output_tokens"],
            s["prefill_tokens"],
            s["final_encoded_context"],
        )
        for s in d["sessions"]
    ]
    assert signature(sep) == signature(col), "not matched work"
    assert sep["mtp_tokens"] == col["mtp_tokens"]
    result = dict(
        scope="single-run four-session two-turn SWE-derived pilot; full recorded input/output lengths; not complete trajectories, quality or unmodified vLLM",
        time_scope="maximum source duration after common host warmup rendezvous; not precisely timestamped global makespan",
        separated=sep,
        colocated=col,
        speed_ratio=col["max_source_seconds"] / sep["max_source_seconds"],
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: (
                    {x: y for x, y in v.items() if x != "sessions"}
                    if isinstance(v, dict)
                    else v
                )
                for k, v in result.items()
            },
            indent=2,
        )
    )
