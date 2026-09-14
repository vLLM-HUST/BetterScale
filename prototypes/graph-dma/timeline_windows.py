"""Read-only rank-local candidate windows; not measured spare bandwidth.

DSV4 target-body delimiters: 43 routed MoEs, two HcPre/HcPost per layer.
Do not sum overlapping kernels or include host scheduling bubbles as bandwidth.
"""

import bisect
import json
import sqlite3
from pathlib import Path
from statistics import median

ROOT = Path("/workspace/strengthen-dsv4")
ARMS = [
    (
        "tp_prefill",
        ROOT / "runs/hw3-split-046/analysis",
        49,
        "TP8, target bucket4128, QA/QB516 local rows; 3 waves",
    ),
    (
        "tp_decode",
        ROOT / "runs/hw3-split-046/analysis",
        48,
        "TP8, target24 query rows (4 K5 seats); 5 waves",
    ),
    (
        "dp_prefill",
        Path(
            "/workspace/strengthen-dsv4-dp-full/runs/hw3-dp8-065/engine/profileskew/analysis"
        ),
        49,
        "TP1/DP8 EP8, fixed1026 rows/rank, skew workload; padding is included",
    ),
    (
        "dp_decode",
        ROOT / "runs/hw3-dp8-052/profiledecode/analysis",
        48,
        "TP1/DP8 EP8, target6 rows/rank (one K5 seat); native retained run052, not current release",
    ),
]


def union(xs):
    out = []
    for a, b in sorted(xs):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def minus(xs, ys):
    ys = union(ys)
    starts = [a for a, b in ys]
    out = []
    for a, b in union(xs):
        pos = a
        i = max(0, bisect.bisect_right(starts, a) - 1)
        while i < len(ys) and ys[i][0] < b:
            lo, hi = ys[i]
            if lo > pos:
                out.append([pos, min(lo, b)])
            pos = max(pos, hi)
            if pos >= b:
                break
            i += 1
        if pos < b:
            out.append([pos, b])
    return out


def duration(xs):
    return sum(b - a for a, b in xs) / 1e6


def quantile(xs, p):
    ss = sorted(xs)
    return ss[round((len(ss) - 1) * p)]


def analyze(path, mid):
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = c.execute("""select t.startNs,t.endNs,s.value,t.modelId,sh.value
    from TASK t join COMPUTE_TASK_INFO x using(globalTaskId)
    join STRING_IDS s on s.id=x.opType join STRING_IDS sh on sh.id=x.inputShapes
    order by t.startNs""").fetchall()
    selected = [r for r in rows if r[3] == mid]
    pre = [r for r in selected if r[2] == "HcPre"]
    post = [r for r in selected if r[2] == "HcPost"]
    gmm = [r for r in selected if r[2] == "GroupedMatmulSwigluQuant"]
    assert len(pre) == len(post) == 2 * len(gmm) and len(gmm) % 43 == 0
    copies = c.execute(
        """select t.startNs,t.endNs from TASK t join STRING_IDS s on s.id=t.taskType
    where s.value='MEMCPY_ASYNC' order by t.startNs"""
    ).fetchall()
    # Use actual device compute/communication task envelopes, not HOST HCCL spans.
    comm = [r[:2] for r in rows if r[2].lower().startswith("hcom")]
    rs = [r[0] for r in rows]
    layers = []
    for i in range(len(gmm)):
        lo, hi = pre[2 * i][0], post[2 * i + 1][1]
        local = rows[bisect.bisect_left(rs, lo) : bisect.bisect_left(rs, hi)]
        mm = [r[:2] for r in local if "matmul" in r[2].lower()]
        dense = [
            r[:2]
            for r in local
            if "matmul" in r[2].lower() and "grouped" not in r[2].lower()
        ]
        others = [r[:2] for r in local if "matmul" not in r[2].lower()]
        transports = [
            (max(a, lo), min(b, hi)) for a, b in comm + copies if a < hi and b > lo
        ]
        dc = minus(dense, transports)
        quiet = minus(dense, others + transports)
        broad = minus(mm, transports)
        layers.append(
            dict(
                wave=i // 43,
                layer=i % 43,
                body_ms=(hi - lo) / 1e6,
                dense_no_transport_ms=duration(dc),
                dense_alone_ms=duration(quiet),
                all_gemm_no_transport_ms=duration(broad),
                all_gemm_union_ms=duration(union(mm)),
                max_dense_contiguous_ms=max([b - a for a, b in dc], default=0) / 1e6,
                transport_ms=duration(union(transports)),
            )
        )
    wave_rows = []
    for j in range(len(gmm) // 43):
        lr = layers[j * 43 : (j + 1) * 43]
        # Full body span includes inter-layer transitions, unlike sum(layer spans).
        wave_rows.append(
            dict(
                wave=j,
                body_ms=(post[(j + 1) * 86 - 1][1] - pre[j * 86][0]) / 1e6,
                **{
                    k: sum(r[k] for r in lr)
                    for k in [
                        "dense_no_transport_ms",
                        "dense_alone_ms",
                        "all_gemm_no_transport_ms",
                        "all_gemm_union_ms",
                        "transport_ms",
                    ]
                },
            )
        )
    return dict(
        source=str(path),
        waves=len(wave_rows),
        wave_medians={
            k: median(r[k] for r in wave_rows) for k in wave_rows[0] if k != "wave"
        },
        layer_summary={
            k: dict(
                p10=quantile([r[k] for r in layers], 0.1),
                p50=median(r[k] for r in layers),
                p90=quantile([r[k] for r in layers], 0.9),
            )
            for k in [
                "body_ms",
                "dense_no_transport_ms",
                "dense_alone_ms",
                "all_gemm_no_transport_ms",
                "max_dense_contiguous_ms",
            ]
        },
        wave_rows=wave_rows,
    )


def main():
    out = []
    for name, folder, mid, scope in ARMS:
        ranks = []
        for rank in range(8):
            result = analyze(folder / f"rank{rank}.db", mid)
            ranks.append(dict(rank=rank, **result))
        summary = {
            k: dict(
                min=min(r["wave_medians"][k] for r in ranks),
                median=median(r["wave_medians"][k] for r in ranks),
                max=max(r["wave_medians"][k] for r in ranks),
            )
            for k in ranks[0]["wave_medians"]
        }
        # 22 GB/s is a single-card local projection onto historical hw3 windows.
        # Not a measured eight-card NUMA/shared-bandwidth guarantee.
        estimates = {
            k: dict(
                wave_mib_per_rank=summary[k]["median"] * 22e6 / 2**20,
                mean_layer_mib_per_rank=summary[k]["median"] / 43 * 22e6 / 2**20,
            )
            for k in [
                "dense_alone_ms",
                "dense_no_transport_ms",
                "all_gemm_no_transport_ms",
            ]
        }
        out.append(
            dict(
                arm=name,
                scope=scope,
                target_model_id=mid,
                summary_ms=summary,
                projection_at_22GBps=estimates,
                ranks=ranks,
            )
        )
        print(name, json.dumps(summary), json.dumps(estimates), flush=True)
    Path(__file__).with_name("timeline-windows.json").write_text(
        json.dumps(
            dict(
                caveat="rank-local target-only candidate windows, not measured free bandwidth; no draft/embedding/head/bubbles; historical hw3 timelines x local single-card bandwidth; excludes hcom device-task and memcpy envelopes, not PMU-certified idle hardware",
                arms=out,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
