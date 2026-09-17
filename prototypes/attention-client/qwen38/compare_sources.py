"""Compare warm target decode windows, preserving long pauses and card counts."""

import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(directory):
    clients = [
        json.loads(p.read_text()) for p in sorted(directory.glob("attention*.json"))
    ]
    owners = [json.loads(p.read_text()) for p in sorted(directory.glob("expert*.json"))]
    assert all(c["status"] == "PASS" and c["aligned_steady_start"] for c in clients)
    assert len(owners) == 4 and all(o["status"] == "pass" for o in owners)
    leaders = [c for c in clients if c["calls"] is not None]
    expected = [
        next((c["calls"] for c in leaders if c["source"] == i), 0) for i in range(2)
    ]
    assert all(o["completed_counts"] == expected for o in owners)
    starts, ends, rows = [], [], []
    for client in leaders:
        offset = 0 if "wave_output_tokens" in client else 2
        times = client["wave_seconds"][offset:]
        token_count = (
            sum(client["wave_output_tokens"])
            if "wave_output_tokens" in client
            else len(times) * client.get("batch_size", 1)
        )
        starts.append(client["wave_started_seconds"][offset])
        ends.append(client["wave_started_seconds"][-1] + times[-1])
        rows.append(
            dict(
                source=client["source"],
                steps=len(times),
                batch_size=client.get("batch_size", 1),
                output_tokens=token_count,
                mtp_tokens=client.get("mtp_tokens", 0),
                prompt_width=client.get("prompt_width", 3),
                median_ms=statistics.median(times) * 1000,
                p95_ms=sorted(times)[math.ceil(0.95 * len(times)) - 1] * 1000,
                max_ms=max(times) * 1000,
                mean_ms=statistics.mean(times) * 1000,
            )
        )
    duration = max(ends) - min(starts)
    return dict(
        capsule_roles=str(directory),
        attention_sources=len(leaders),
        deferred_steady_gc=all(c.get("deferred_steady_gc", False) for c in clients),
        physical_cards=len(clients) + 4,
        steady_seconds=duration,
        output_tokens=sum(r["output_tokens"] for r in rows),
        output_tokens_per_second=sum(r["output_tokens"] for r in rows) / duration,
        sources=rows,
        graph_shadow_relative_l2=[c.get("graph_shadow_relative_l2") for c in clients],
        graph_shadow_exact=[c.get("graph_shadow_exact") for c in clients],
        full_run_paired_waves=[sum(o["completed_counts"]) - o["waves"] for o in owners],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directories", type=Path, nargs="+")
    args = p.parse_args()
    print(
        json.dumps(
            dict(
                scope="hw0 same E4 pool, one vs two TP2 sources, synthetic short-context target decode; not equal-card colocated baseline or online throughput",
                timing="warm target decode replays/source; excludes prefill and capture, includes all observed pauses; host wall time",
                runs=[summarize(d) for d in args.directories],
            ),
            indent=2,
        )
    )
