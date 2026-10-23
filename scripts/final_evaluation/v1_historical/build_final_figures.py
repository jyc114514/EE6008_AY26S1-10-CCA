"""Build reproducible Core A dev/final and Core B dev-only figures."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_ROOT = Path(__file__).resolve().parents[1]
PROJECT = OUT_ROOT.parents[2]
DEV_ROOT = OUT_ROOT / "replay/dev_replay/attempt_06_score_retry_02"
FIG_ROOT = OUT_ROOT / "figures"
DATA_ROOT = FIG_ROOT / "figure_data"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty figure data: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def build_core_a() -> None:
    dev = read_csv(DEV_ROOT / "CORE_A_DEV_REPLAY_PER_SEED.csv")
    final_path = OUT_ROOT / "results/CORE_A_FINAL_PER_SEED.csv"
    final = read_csv(final_path)
    data = [{"split": "development", **row} for row in dev] + [{"split": "final_test", **row} for row in final]
    write_csv(DATA_ROOT / "CORE_A_MATCHED_DEV_FINAL.csv", data)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    metrics = (("top1_accuracy", "Top-1 accuracy"), ("macro_f1", "Macro-F1"))
    condition_colors = {"A_DYNAMIC": "#007C78", "A_STATIC": "#C15A46"}
    for row_index, (split, support) in enumerate((("development", 617), ("final_test", 602))):
        split_rows = [row for row in data if row["split"] == split]
        for col_index, (metric, label) in enumerate(metrics):
            ax = axes[row_index, col_index]
            for seed in (20260825, 20260826, 20260827):
                dynamic = next(float(row[metric]) for row in split_rows if row["program"] == "A_DYNAMIC" and int(row["seed"]) == seed)
                static = next(float(row[metric]) for row in split_rows if row["program"] == "A_STATIC" and int(row["seed"]) == seed)
                ax.plot([0, 1], [dynamic, static], color="#8A939B", linewidth=1, alpha=0.75, marker="o", markersize=4, zorder=2)
            for x, program in enumerate(("A_DYNAMIC", "A_STATIC")):
                values = np.asarray([float(row[metric]) for row in split_rows if row["program"] == program], dtype=np.float64)
                ax.errorbar(x, values.mean(), yerr=values.std(ddof=1), fmt="o", color=condition_colors[program], ecolor=condition_colors[program], capsize=4, markersize=8, linewidth=2, zorder=3)
                for seed_index, value in enumerate(values):
                    ax.scatter(x + (-0.04 + seed_index * 0.04), value, color=condition_colors[program], edgecolor="white", linewidth=0.5, s=35, zorder=4)
            ax.set_xticks([0, 1], ["Dynamic", "Static"])
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel(label)
            ax.set_title(f"{'Development' if split == 'development' else 'Final'} • n={support}", loc="left", fontsize=12, weight="bold")
            ax.grid(axis="y", color="#D8DEE4", linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Core A matched checkpoints", fontsize=16, weight="bold")
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_ROOT / "CORE_A_MATCHED_DEV_FINAL.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIG_ROOT / "CORE_A_MATCHED_DEV_FINAL.svg", bbox_inches="tight")
    plt.close(fig)
    (FIG_ROOT / "CORE_A_MATCHED_DEV_FINAL.caption.md").write_text(
        "Development validation (617 episodes) and the v3 primary final partition (602 episodes) are shown in separate rows. Grey segments pair dynamic and separately trained static checkpoints within each of the three fixed seeds; colored markers show the across-seed mean and sample SD. The two populations are descriptive panels and are not compared as a paired sample.\n",
        encoding="utf-8",
    )


def build_core_b_dev() -> None:
    summary = read_csv(DEV_ROOT / "CORE_B_DEV_REPLAY_SUMMARY.csv")
    write_csv(DATA_ROOT / "CORE_B_DEV_ONLY.csv", summary)
    programs = [row["program"] for row in summary]
    labels = {"A0": "A0", "AD": "AD", "V0": "V0", "VD": "VD", "S_OPT": "S-OPT", "MIX0": "MIX0", "MIXD": "MIXD", "PERSISTENCE": "Persistence"}
    metrics = (("position_mae", "Position MAE"), ("safe_fd_l1", "Safe-FD L1"), ("seam_safe_fd_l1", "Seam Safe-FD L1"))
    colors = {"V0": "#007C78", "VD": "#E09F3E", "A0": "#597081", "AD": "#8A7CA8", "S_OPT": "#3B6F8F", "MIX0": "#2D9A8F", "MIXD": "#61B7A9", "PERSISTENCE": "#C15A46"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    x = np.arange(len(programs))
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        means = np.asarray([float(row[f"{metric}_mean"]) for row in summary])
        sds = np.asarray([0.0 if row[f"{metric}_sample_sd"] == "" else float(row[f"{metric}_sample_sd"]) for row in summary])
        ax.bar(x, means, color=[colors[p] for p in programs], width=0.72, edgecolor="white", linewidth=0.7)
        for i, (mean, sd) in enumerate(zip(means, sds, strict=True)):
            if sd > 0:
                ax.errorbar(i, mean, yerr=sd, color="#28323C", capsize=3, linewidth=1.2)
        ax.set_xticks(x, [labels[p] for p in programs], rotation=35, ha="right")
        ax.set_title(title, loc="left", fontsize=12, weight="bold")
        ax.set_ylabel("Train-normalized error (lower is better)")
        ax.grid(axis="y", color="#D8DEE4", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Core B development replay only • 47 episodes / 185 anchors", fontsize=15, weight="bold")
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_ROOT / "CORE_B_DEV_ONLY.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIG_ROOT / "CORE_B_DEV_ONLY.svg", bbox_inches="tight")
    plt.close(fig)
    (FIG_ROOT / "CORE_B_DEV_ONLY.caption.md").write_text(
        "Development-only task-macro results across five tasks. Bars show the mean across three fixed training seeds; error bars show sample SD, omitted for the single Persistence reference. V0 is the primary single-model reference and VD is the secondary dropout condition. Core B final inference remains closed because the fresh GPU3 encoder feature bridge did not meet the registered 1e-5 tolerance against the accepted cache.\n",
        encoding="utf-8",
    )


def main() -> int:
    build_core_a()
    build_core_b_dev()
    print("built CORE_A_MATCHED_DEV_FINAL and CORE_B_DEV_ONLY figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
