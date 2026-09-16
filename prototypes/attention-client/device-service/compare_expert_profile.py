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
        while ss[end]["label"] != "ReduceSum":
            end += 1
        assert ss[end + 1]["label"] == "Cast"
        end += 1
        remote.append(
            dict(
                rows=n,
                span_us=(ss[end]["end_ns"] - ss[start]["start_ns"]) / 1000,
                client_us=cl["dur_us"],
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
print(json.dumps(report, indent=2))
(root / "analysis/expert-stage-comparison.json").write_text(
    json.dumps(report, indent=2)
)
