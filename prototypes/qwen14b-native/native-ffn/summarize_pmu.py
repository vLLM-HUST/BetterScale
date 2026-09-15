"""Summarize actual per-core PMU counters, never interpret them as a timeline."""

import csv, json, statistics, sys, html, zipfile
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
summary = {}
panels = {}
tables = []
keys = [
    "aic_time(us)",
    "aic_cube_time(us)",
    "aic_mte2_time(us)",
    "aic_scalar_wait_id9_time(us)",
    "aiv_time(us)",
    "aiv_vec_time(us)",
    "aiv_mte2_time(us)",
    "aiv_mte3_time(us)",
    "aiv_scalar_wait_id8_time(us)",
    "aic_read_hit_rate(%)",
    "aiv_read_hit_rate(%)",
]
with zipfile.ZipFile(out / "per-core-counters.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for mode in ["pair", "full", "matmul", "swiglu"]:
        summary[mode] = {}
        allrows = []
        for name in [
            "OpBasicInfo",
            "PipeUtilization",
            "ArithmeticUtilization",
            "Memory",
            "MemoryL0",
            "MemoryUB",
            "L2Cache",
            "ResourceConflictRatio",
        ]:
            p = next((root / mode).rglob(name + ".csv"))
            z.write(p, f"{mode}/{name}.csv")
            rows = list(csv.DictReader(p.open()))
            if name == "OpBasicInfo":
                summary[mode]["basic"] = rows[0]
            if name == "PipeUtilization":
                panels[mode] = rows
            for k in keys:
                vals = [float(r[k]) for r in rows if r.get(k) not in [None, "", "NA"]]
                if vals:
                    summary[mode][k] = dict(
                        n=len(vals),
                        min=min(vals),
                        median=statistics.median(vals),
                        max=max(vals),
                    )
            head = list(rows[0])
            body = "".join(
                "<tr>"
                + "".join("<td>" + html.escape(r.get(k, "")) + "</td>" for k in head)
                + "</tr>"
                for r in rows
            )
            tables.append(
                f'<details><summary>{mode} / {name}</summary><div class="scroll"><table><thead><tr>'
                + "".join("<th>" + html.escape(k) + "</th>" for k in head)
                + f"</tr></thead><tbody>{body}</tbody></table></div></details>"
            )
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
svg = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="1140" height="550" viewBox="0 0 1140 550">',
    '<rect width="1140" height="550" fill="#f8fafc"/>',
    '<g font-family="sans-serif" fill="#182535">',
    '<text x="25" y="28" font-size="19">4K BF16 gate/up: per-core hardware counters</text>',
    '<text x="25" y="50" font-size="13">Purple: Cube active time. Blue: MTE2 active time. Independent counters, NOT a timeline; do not add.</text>',
]
for col, mode in enumerate(["matmul", "pair", "full"]):
    x = 25 + col * 375
    svg.append(f'<text x="{x}" y="82" font-size="17">{mode}</text>')
    rows = [r for r in panels[mode] if r["aic_time(us)"] != "NA"]
    for j, r in enumerate(rows):
        y = 103 + j * 17
        svg.append(f'<text x="{x}" y="{y+9}" font-size="10">C{r["block_id"]}</text>')
        for offset, k, color in [
            (0, "aic_cube_time(us)", "#8b5cf6"),
            (6, "aic_mte2_time(us)", "#0891b2"),
        ]:
            width = float(r[k]) / 5000 * 300
            svg.append(
                f'<rect x="{x+30}" y="{y+offset}" width="{width:.2f}" height="5" fill="{color}"><title>{html.escape(k)}: {r[k]} us</title></rect>'
            )
    for tick in range(6):
        svg.append(f'<text x="{x+30+tick*60}" y="530" font-size="10">{tick}ms</text>')
svg.append("</g></svg>")
(out / "cube-memory-counters.svg").write_text("\n".join(svg))
(out / "per-core-report.html").write_text(
    """<!doctype html><meta charset="utf-8"><title>Qwen FFN per-core PMU</title><style>body{font:15px system-ui;margin:24px;color:#182535}summary{padding:10px;cursor:pointer;background:#eef2ff;margin-top:8px}.scroll{overflow:auto;max-height:650px}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px;white-space:nowrap}th{position:sticky;top:0;background:#eef2ff}</style><h1>Qwen 4K FFN: per-core PMU</h1><p>Local910B2 physical7, CANN9.0.1, BF16/NZ, M4096 K5120 N27648. Each profile targets one standalone leaf kernel. Down GEMM excluded.</p><p><b>TimelineDetail failed:</b> kernel context/argument dump failed. These are real-board PMU counters, not instruction timelines. Counters overlap and must not be added. Kernel replay/cache conditions differ from whole-FFN graph throughput tests.</p><img width="1140" src="cube-memory-counters.svg"><p>Individual rows for all24 Cube and48 Vector lanes are below. Missing fields are not zero.</p>"""
    + "".join(tables)
)
print(
    json.dumps(
        {
            m: {k: round(v["median"], 3) for k, v in d.items() if k != "basic"}
            for m, d in summary.items()
        },
        indent=2,
    )
)
