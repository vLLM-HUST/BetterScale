"""Read the captured oracle and remote stages without double-counting task wrappers."""

import sqlite3
import json
import statistics
import pathlib
import argparse

parser = argparse.ArgumentParser(
    description="Diagnostic stage accounting, not a fair graph speedup benchmark."
)
parser.add_argument("root", type=pathlib.Path)
root = parser.parse_args().root.resolve()
parallel = json.loads((root / "run/measurements/expert0.json").read_text()).get(
    "parallel_transport", False
)
report = {
    "scope": "profiled two-layer dummy correctness fixture; native eager vs device graph; no speedup claim",
    "clients": [],
}
for rank in range(2):
    c = sqlite3.connect(root / f"analysis/attention{rank}.db")
    c.row_factory = sqlite3.Row
    es = [
        dict(r)
        for r in c.execute(
            "select label,start_ns,end_ns,dur_us,stream_id from traceloom_event where source_table='TASK' order by start_ns"
        )
    ]
    refstream = next(e["stream_id"] for e in es if e["label"] == "GroupedMatmul")
    ref = [e for e in es if e["stream_id"] == refstream]
    checks = json.loads((root / f"run/measurements/attention{rank}.json").read_text())[
        "checks"
    ]
    rows = [x["rows"] for x in checks[-6:] for _ in range(2)]
    # Each native expert stage ends at the unique unpermute.
    ends = [i for i, e in enumerate(ref) if e["label"] == "MoeTokenUnpermute"][-12:]
    assert len(ends) == len(rows) == 12
    assert all(not x["preparing"] for x in checks[-6:])
    native = []
    for end, n in zip(ends, rows):
        gate = max(i for i in range(end) if ref[i]["label"] == "MoeGatingTopK")
        start = gate - 1
        while ref[start]["label"] != "MatMulV2":
            start -= 1
        events = ref[start : end + 1]
        selected = [
            e
            for e in events
            if e["label"]
            in (
                "MatMulV2",
                "MoeGatingTopK",
                "MoeInitRoutingV3",
                "GroupedMatmul",
                "SwiGlu",
                "Abs",
                "MoeTokenUnpermute",
            )
        ]
        assert sum(e["label"] == "GroupedMatmul" for e in selected) == 2
        native.append(
            dict(
                rows=n,
                span_us=(ref[end]["end_ns"] - ref[start]["start_ns"]) / 1000,
                kernels_us=sum(e["dur_us"] for e in selected),
                gemm_us=sum(
                    e["dur_us"] for e in selected if e["label"] == "GroupedMatmul"
                ),
            )
        )
    remote = []
    clients = [e for e in es if e["label"] == "neural_client"][-12:]
    assert len(clients) == 12
    for cl, n in zip(clients, rows):
        ss = [e for e in es if e["stream_id"] == cl["stream_id"]]
        idx = next(i for i, e in enumerate(ss) if e["start_ns"] == cl["start_ns"])
        start = idx - 1
        while ss[start]["label"] != "MatMulV2":
            start -= 1
        end = idx + 1
        terminal = "MoeTokenUnpermute" if parallel else "ReduceSum"
        while ss[end]["label"] != terminal:
            end += 1
        if not parallel:
            assert ss[end + 1]["label"] == "Cast"
            end += 1
        remote.append(
            dict(
                rows=n,
                span_us=(ss[end]["end_ns"] - ss[start]["start_ns"]) / 1000,
                client_us=sum(
                    e["dur_us"]
                    for e in ss[idx : end + 1]
                    if e["label"]
                    in ("neural_client", "neural_collect", "neural_retire")
                ),
            )
        )
    aggregates = {}
    for n in sorted(set(rows)):
        a = [x for x in native if x["rows"] == n]
        b = [x for x in remote if x["rows"] == n]
        aggregates[n] = dict(
            samples=len(a),
            native_kernel_median_us=statistics.median(x["kernels_us"] for x in a),
            native_gmm_median_us=statistics.median(x["gemm_us"] for x in a),
            native_eager_span_median_us=statistics.median(x["span_us"] for x in a),
            remote_graph_span_median_us=statistics.median(x["span_us"] for x in b),
            remote_wait_transfer_median_us=statistics.median(x["client_us"] for x in b),
        )
    report["clients"].append(
        dict(rank=rank, by_rows=aggregates, native=native, remote=remote)
    )
# Server cycles have device-written source generations; use those identities,
# not nearest timestamps, to bind single-source cost to its actual row count.
rowmaps = {}
for rank in (0, 1):
    record = json.loads((root / f"run/measurements/attention{rank}.json").read_text())
    rowmaps[rank] = {
        e["generation"]: e["rows"] for e in record["events"] if e["event"] == "submit"
    }
report["servers"] = []
for rank in (0, 1):
    connection = sqlite3.connect(root / f"analysis/expert{rank}.db")
    connection.row_factory = sqlite3.Row
    events = [
        dict(e)
        for e in connection.execute(
            "select label,dur_us from traceloom_event where source_table='TASK' order by start_ns"
        )
    ]
    starts = [i for i, e in enumerate(events) if e["label"] == "neural_prepare"]
    traces = json.loads((root / f"run/measurements/expert{rank}.json").read_text())[
        "trace"
    ]
    assert len(starts) == len(traces) == 48
    buckets = {}
    for wave, (start, trace) in enumerate(zip(starts, traces)):
        generations = [(c, trace[c + 2]) for c in (0, 1) if trace[c + 2]]
        if len(generations) != 1 or generations[0][1] < 13:
            continue
        client, generation = generations[0]
        rows = rowmaps[client][generation]
        stop = starts[wave + 1] if wave + 1 < len(starts) else len(events)
        totals = {}
        for event in events[start:stop]:
            label = event["label"]
            if label in (
                "neural_prepare",
                "neural_pack",
                "neural_scatter",
                "neural_complete",
                "GroupedMatmul",
                "SwiGlu",
            ):
                totals[label] = totals.get(label, 0) + event["dur_us"]
        bucket = buckets.setdefault(rows, {})
        for label, value in totals.items():
            bucket.setdefault(label, []).append(value)
    report["servers"].append(
        dict(
            rank=rank,
            by_rows={
                rows: {
                    label: statistics.median(values) for label, values in bucket.items()
                }
                for rows, bucket in buckets.items()
            },
            warning="prepare includes idle polling; single-source sealed waves only",
        )
    )
print(json.dumps(report, indent=2))
(root / "analysis/expert-stage-comparison.json").write_text(
    json.dumps(report, indent=2)
)
