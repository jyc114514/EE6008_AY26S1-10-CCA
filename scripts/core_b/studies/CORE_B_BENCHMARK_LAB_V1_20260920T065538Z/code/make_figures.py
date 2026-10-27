"""Create truthful baseline/progress figures from locked CPU evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import BENCH_ROOT, atomic_write_json, read_csv, read_json


def save_pair(fig: plt.Figure, stem: str, caption: str, population: str, uncertainty: str) -> dict[str, object]:
    png = BENCH_ROOT / "figures" / f"{stem}.png"
    pdf = BENCH_ROOT / "figures" / f"{stem}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"figure_id": stem, "png": str(png), "pdf": str(pdf), "caption": caption, "population": population, "uncertainty": uncertainty, "status": "PASS"}


def main() -> int:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 120})
    manifest = []
    query = read_csv(BENCH_ROOT / "d1/D1_QUERY_TABLE.csv")
    dispersion = np.asarray([float(row["dispersion_h60"]) for row in query])
    distance = np.asarray([float(row["neighbor_distance_k20"]) for row in query])
    thresholds = read_json(BENCH_ROOT / "d1/D1_SUMMARY.json")["thresholds"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    axes[0].hist(dispersion, bins=35, color="#3568a8", alpha=0.85)
    axes[0].axvline(float(thresholds["q25_h60"]), color="#d97706", linestyle="--", label="Q25")
    axes[0].axvline(float(thresholds["q75_h60"]), color="#b91c1c", linestyle="--", label="Q75")
    axes[0].set(title="D1 train-neighborhood dispersion", xlabel="A (standardized residual MAD)", ylabel="dev windows")
    axes[0].legend(frameon=False)
    axes[1].hist(distance, bins=35, color="#4b8c5a", alpha=0.85)
    axes[1].axvline(float(thresholds["support_q95_distance"]), color="#b91c1c", linestyle="--", label="train Q95")
    axes[1].set(title="D1 support distance", xlabel="20th-neighbor state MSE", ylabel="dev windows")
    axes[1].legend(frameon=False)
    manifest.append(save_pair(fig, "FIG_D1_SUPPORT_DISPERSION", "Locked state-only neighbor bank dispersion and support distance; no learned video result shown.", "V3 primary train bank and dev windows; 603 dev episodes", "No CI; descriptive distributions"))

    stats = read_csv(BENCH_ROOT / "d1/D1_BOOTSTRAP_CLUSTER_STATS.csv")
    selected = [row for row in stats if row["horizon_steps"] == "60" and row["stratum"] in {"low_supported", "high_supported"}]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    bodies = ["Hand", "Arm", "Full28"]
    x = np.arange(len(bodies))
    width = 0.36
    for offset, stratum, color in [(-width / 2, "low_supported", "#d97706"), (width / 2, "high_supported", "#3568a8")]:
        rows = {row["body"]: row for row in selected if row["stratum"] == stratum}
        values = [float(rows[body]["estimate"]) if body in rows else np.nan for body in bodies]
        lows = [float(rows[body]["estimate"]) - float(rows[body]["ci_low"]) if body in rows else 0.0 for body in bodies]
        highs = [float(rows[body]["ci_high"]) - float(rows[body]["estimate"]) if body in rows else 0.0 for body in bodies]
        ax.bar(x + offset, values, width, yerr=np.vstack([lows, highs]), capsize=3, color=color, label=stratum.replace("_", " "))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, bodies)
    ax.set_ylabel("Persistence MAE − NN median MAE")
    ax.set_title("D1 neighbor baseline gain; 2 s horizon")
    ax.legend(frameon=False)
    manifest.append(save_pair(fig, "FIG_D1_NN_GAIN_CLUSTER_BOOTSTRAP", "CPU neighbor-median versus persistence standardized-MAE gain; positive favors the neighbor baseline. Learned V3 fusion/state rows are pending.", "603 dev episodes; low/high supported strata", "10,000 episode bootstrap percentile CI"))

    events = read_csv(BENCH_ROOT / "d2/D2_EVENT_REGISTRY.csv")
    groups = ["Hand", "Arm"]
    counts = [sum(row["group"] == group for row in events) for group in groups]
    dense = read_csv(BENCH_ROOT / "d2/D2_DENSE_BASELINE_RESULTS.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    axes[0].bar(groups, counts, color=["#7c3aed", "#0f766e"])
    axes[0].set(title="Dev command-change event registry", ylabel="events")
    labels = [f"{'P' if row['model'] == 'PERSISTENCE' else 'S'} H{row['horizon_steps']}" for row in dense]
    axes[1].bar(labels, [int(row["misses"]) for row in dense], color="#64748b")
    axes[1].set(title="Dense baseline missed events", ylabel="misses")
    axes[1].tick_params(axis="x", labelsize=8)
    manifest.append(save_pair(fig, "FIG_D2_EVENT_SUPPORT_BASELINES", "D2 command-change event support and fixed-panel baseline misses; learned-model event forecasts are not yet available.", "588 dev event records; 118-episode dense panel", "Counts only; no inferential CI"))

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    stages = ["V3 training", "D1 CPU", "D2 CPU\nbaselines", "D3 grid\nStage0", "D3 grid\nfull", "D3 36\ntraining"]
    values = [36, 1, 1, 1, 0, 0]
    colors = ["#3568a8", "#4b8c5a", "#4b8c5a", "#4b8c5a", "#d97706", "#d97706"]
    ax.bar(stages, values, color=colors)
    ax.set_ylim(0, 38)
    ax.set_ylabel("accepted units / 36 where applicable")
    ax.set_title("Independent benchmark progress at snapshot")
    for i, value in enumerate(values):
        ax.text(i, value + 0.5, str(value), ha="center", va="bottom")
    manifest.append(save_pair(fig, "FIG_BENCHMARK_PROGRESS", "Execution progress; D3 full cache is running and D3 training has not started, so zeros are not scientific scores.", "Benchmark state snapshot", "No uncertainty"))
    atomic_write_json(BENCH_ROOT / "figures/FIGURE_MANIFEST.json", {"status": "PASS", "figures": manifest, "final_test_access": 0})
    print(json.dumps({"status": "PASS", "figures": len(manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
