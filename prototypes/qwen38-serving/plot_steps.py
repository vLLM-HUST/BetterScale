"""Paper-style static figure from verified step-summary.json, no invented points."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("capsule", type=Path)
    a = p.parse_args()
    data = json.loads((a.capsule / "step-summary.json").read_text())
    assert not data["missing_cohorts"], data["missing_cohorts"]
    points = data["points"]
    assert all(set(p["summary"]) == {"baseline", "candidate"} for p in points)
    assert all(s["cohorts"] == 4 for p in points for s in p["summary"].values())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    colors = dict(baseline="#566579", candidate="#007f73")

    def curves(ax, selected, xkey, suffix="", style="-"):
        selected.sort(key=lambda p: p["case"][xkey])
        x = [p["case"][xkey] for p in selected]
        for arm, color in colors.items():
            stats = [p["summary"][arm]["period_ms"] for p in selected]
            y = [s["mean"] for s in stats]
            ax.plot(x, y, marker="o", linestyle=style, color=color,
                    label=("Native" if arm == "baseline" else "BetterScale") + suffix)
            ax.fill_between(x, [s["low"] for s in stats], [s["high"] for s in stats],
                            color=color, alpha=.10)
        ax.set_ylabel("Step period (ms)")
        ax.grid(axis="y", alpha=.18)

    ax = axes[0, 0]
    for context, style in ((1024, "-"), (4096, "--")):
        curves(ax, [p for p in points if p["case"]["kind"] == "decode" and p["case"]["context"] == context],
               "batch", f" / initial {context // 1024}K", style)
    ax.set(title="A  Occupied decode", xlabel="Actual active requests", xticks=[1, 2, 4, 8])
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    curves(ax, [p for p in points if p["case"]["kind"] == "prefill" and p["case"].get("prefix", 0) == 0], "context")
    ax.set(title="B  Cold-prefix prefill", xlabel="Actual scheduled tokens", xticks=[128, 512, 1024, 1536])
    ax.legend(frameon=False, fontsize=8)
    ax.text(.98, .04, "1536 = first chunk of a 2048-token prompt.", transform=ax.transAxes, fontsize=8, ha="right")

    ax = axes[1, 0]
    for batch, style in ((1, "-"), (4, "--")):
        curves(ax, [p for p in points if p["case"]["kind"] == "mixed" and p["case"]["batch"] == batch],
               "joining", f" / {batch} decode", style)
    ax.set(title="C  Mixed prefill + occupied decode", xlabel="Joining prefill tokens (+ 1 or 4 decode tokens)", xticks=[128, 512, 1024, 1536])
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    point = next(p for p in points if p["case"] == dict(kind="decode", batch=8, context=4096))
    for arm, color in colors.items():
        values = sorted(x for cohort in point["arms"][arm] for x in cohort["between_forward_samples_ms"])
        assert values
        ax.step(values, [(i + 1) / len(values) for i in range(len(values))], where="post", color=color,
                label=("Native" if arm == "baseline" else "BetterScale") + f" (n={len(values)})")
    ax.set(title="D  Between-forward interval: 8 requests / initial 4K", xlabel="Forward end to next forward start (ms)", ylabel="Empirical cumulative fraction", ylim=(0, 1.02))
    ax.grid(alpha=.18)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Graph / asynchronous execution — Qwen3.8-27B, TP2", fontsize=17, fontweight="bold", y=.99)
    fig.text(.5, .947, "Same Ascend 910B2 pair · no MTP · APC + AIV enabled · cold shape cohorts", ha="center", fontsize=10)
    fig.text(.06, .028, "ABBA; 4 measured cohorts/arm/point after shape warmup. Bands: observed cohort range, not confidence intervals.\n"
             "Unprofiled external device events. Forward intervals include useful work and waits, not just idle. SWE service results are separate.", fontsize=8, color="#4a5564")
    fig.tight_layout(rect=(0, .065, 1, .93), h_pad=2.5)
    for extension in ("svg", "png", "pdf"):
        fig.savefig(a.capsule / f"qwen-step-efficiency.{extension}", dpi=180)


if __name__ == "__main__":
    main()
