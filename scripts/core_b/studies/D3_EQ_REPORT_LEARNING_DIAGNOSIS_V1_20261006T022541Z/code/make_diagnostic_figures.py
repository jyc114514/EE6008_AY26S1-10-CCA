#!/usr/bin/env python3
"""Generate provenance-bound train/dev, seed and fixed-trajectory figures."""
from __future__ import annotations
import os

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
EQ = PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
OUT = PROJECT / "reports/benchmarks/D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1_20261006T022541Z"
FIG = OUT / "figures"
CONFIGS = ("TS00-D", "TS10-D", "TS01-D", "TS11-D", "TS11-S", "STATE-CTRL")
SPLITS = ("train", "dev")
COLORS = {"train": "#0072B2", "dev": "#D55E00"}
SEED_COLORS = {20261001: "#0072B2", 20261002: "#D55E00", 20261003: "#009E73"}
REGION_MARKERS = {"action[0:14]": "o", "action[14:28]": "s", "action[0:28]": "D"}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(fig, name: str) -> dict[str, str]:
    png = FIG / f"{name}.png"
    pdf = FIG / f"{name}.pdf"
    fig.savefig(png, dpi=240, facecolor="white", metadata={"Title": name, "Creator": "D3-EQ local analysis"})
    fig.savefig(pdf, facecolor="white", metadata={"Title": name, "Creator": "D3-EQ local analysis"})
    plt.close(fig)
    return {"png": str(png), "png_sha256": sha(png), "pdf": str(pdf), "pdf_sha256": sha(pdf)}


def plot_train_dev() -> dict[str, str]:
    rows = load_rows(OUT / "TRAIN_DEV_SIX_CONFIG_SUMMARY.csv")
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), layout="constrained", sharex="col")
    x = np.arange(len(CONFIGS), dtype=float)
    metrics = [
        ("Persistence skill: standardized MAE gain", "gain_vs_persistence_standardized_mae", 0.0),
        ("Future motion fidelity: TV(pred) / TV(GT)", "tv_ratio_group_weighted_means", 1.0),
        ("Active-step direction accuracy", "direction_accuracy_active", None),
    ]
    for col, horizon in enumerate((30, 60)):
        for r_i, (title, field, reference) in enumerate(metrics):
            ax = axes[r_i, col]
            for split in SPLITS:
                vals, low, high = [], [], []
                for config in CONFIGS:
                    matches = [q for q in rows if q["split"] == split and q["config"] == config
                               and int(q["horizon_steps"]) == horizon and q["region_raw"] == "action_full28"]
                    if len(matches) != 1:
                        raise AssertionError(f"summary row count {split}/{config}/H{horizon}: {len(matches)}")
                    row = matches[0]
                    mean_value = float(row[f"{field}_seed_mean"])
                    min_value = float(row[f"{field}_seed_min"])
                    max_value = float(row[f"{field}_seed_max"])
                    vals.append(mean_value); low.append(mean_value - min_value); high.append(max_value - mean_value)
                offset = -0.10 if split == "train" else 0.10
                marker = "o" if split == "train" else "s"
                ax.errorbar(x + offset, vals, yerr=np.vstack((low, high)), fmt=marker,
                            color=COLORS[split], ecolor=COLORS[split], capsize=2,
                            linewidth=1.2, markersize=4.5, label=split.capitalize())
            if reference is not None:
                ax.axhline(reference, color="#555555", linestyle="--", linewidth=0.8)
            ax.set_title(f"H{horizon} — {title}")
            ax.grid(axis="y", color="#dddddd", linewidth=0.6)
            ax.set_ylabel("Mean across 3 seeds; whiskers = seed min–max")
            if r_i == 0 and col == 0:
                ax.legend(frameon=False, ncol=2, loc="best")
            if r_i == 2:
                ax.set_xticks(x, CONFIGS, rotation=25, ha="right")
    fig.suptitle("D3-EQ train/dev learning and motion fidelity — Full28 raw action vector", fontsize=15)
    fig.text(0.5, -0.01,
             "Each point is a source-group-equal score averaged across the three fixed seeds; whiskers are seed ranges, not confidence intervals. "
             "Dev uses 603 source groups; train uses 2,712. Direction is conditional on train-frozen active steps.",
             ha="center", va="top", fontsize=9)
    return save(fig, "FIG_TRAIN_DEV_LEARNING_DIAGNOSIS_V1")


def plot_seed_sensitivity() -> dict[str, str]:
    rows = load_rows(OUT / "SEED_SENSITIVITY_PRIMARY.csv")
    contrasts = ("TS11D_vs_TS10D", "TS11D_vs_TS01D", "TS11D_vs_TS11S", "TS11D_vs_STATE")
    labels = ("TS11-D − TS10-D\n(add space)", "TS11-D − TS01-D\n(add time)",
              "TS11-D − TS11-S\n(dynamic − static)", "TS11-D − STATE\n(add video)")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), layout="constrained", sharey=True)
    seed_cols = ("seed_20261001", "seed_20261002", "seed_20261003")
    seed_ids = (20261001, 20261002, 20261003)
    for ax, horizon in zip(axes, (30, 60)):
        for x_i, contrast in enumerate(contrasts):
            for region_i, region in enumerate(("action[0:14]", "action[14:28]", "action[0:28]")):
                row = next(q for q in rows if q["contrast"] == contrast and int(q["horizon_steps"]) == horizon and q["body_raw_slice"] == region)
                base_x = x_i + (region_i - 1) * 0.19
                mean_value = float(row["fixed_seed_mean"])
                lo, hi = float(row["conditional_group_ci_low"]), float(row["conditional_group_ci_high"])
                ax.errorbar(base_x, mean_value, yerr=[[mean_value - lo], [hi - mean_value]],
                            fmt=REGION_MARKERS[region], color="#222222", ecolor="#444444",
                            capsize=3, markersize=5, linewidth=1.1, zorder=3)
                for seed_i, (seed_id, col) in enumerate(zip(seed_ids, seed_cols)):
                    jitter = (seed_i - 1) * 0.035
                    ax.scatter(base_x + jitter, float(row[col]), s=27, marker=REGION_MARKERS[region],
                               color=SEED_COLORS[seed_id], edgecolor="white", linewidth=0.35, zorder=4)
        ax.axhline(0, color="#333333", linewidth=0.9)
        ax.set_xticks(range(4), labels, fontsize=8)
        ax.set_title(f"H{horizon}")
        ax.set_ylabel("Candidate − control standardized MAE\n(negative favors TS11-D)")
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    seed_handles = [plt.Line2D([], [], marker="o", color=SEED_COLORS[s], linestyle="None", label=str(s)) for s in seed_ids]
    region_handles = [plt.Line2D([], [], marker=m, color="#222222", linestyle="None", label=lab)
                      for lab, m in (("action[0:14]", "o"), ("action[14:28]", "s"), ("action[0:28]", "D"))]
    axes[0].legend(handles=seed_handles + region_handles, frameon=False, fontsize=8,
                   ncol=2, loc="lower left", title="Seed points / raw slice")
    fig.suptitle("Primary contrasts: fixed-seed points and conditional source-group intervals", fontsize=14)
    fig.text(0.5, -0.02,
             "Black interval = locked 95% source-group bootstrap CI for the fixed-three-seed mean; it is conditional on those trained models. "
             "Colored dots are seed estimates; labels are raw action slices because Hand/Arm mapping conflicts across source code.",
             ha="center", va="top", fontsize=8.5)
    return save(fig, "FIG_PRIMARY_SEED_SENSITIVITY_V1")


def plot_fixed_trajectories() -> dict[str, str]:
    selection_path = EQ / "protocol/D3_EQ_TRAJECTORY_SELECTION.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "LOCKED_BEFORE_PREDICTION_OUTCOMES" or selection.get("prediction_outcomes_opened") is not False:
        raise AssertionError("trajectory selection is not outcome-blind locked input")
    reps = selection["representatives"]
    archive = np.load(V3 / "data/V3_MAIN_INPUTS.npz", allow_pickle=False)
    ids = archive["anchor_id"].astype("U96")
    id_index = {sample_id: i for i, sample_id in enumerate(ids.tolist())}
    history = archive["history_native"].astype(np.float32, copy=False)
    times = archive["history_selected_timestamps"].astype(np.float32, copy=False)
    targets = archive["targets_native"].astype(np.float32, copy=False)
    anchors = archive["anchor_command_native"].astype(np.float32, copy=False)
    fig, axes = plt.subplots(3, 2, figsize=(14, 9), layout="constrained", sharex=False)
    linecolors = {0: "#0072B2", 14: "#D55E00"}
    for row_i, rep in enumerate(reps):
        sample_id = rep["anchor_id"]
        if sample_id not in id_index:
            raise AssertionError(f"fixed selected anchor absent from locked inputs: {sample_id}")
        index = id_index[sample_id]
        t_hist = times[index] - float(times[index, -1])
        hist0 = history[index, :, 43]
        hist14 = history[index, :, 57]
        anchor = anchors[index]
        for col_i, horizon in enumerate((30, 60)):
            ax = axes[row_i, col_i]
            run_id = f"D3EQ-TS11-D-H{horizon}-S20261001"
            pred_path = EQ / "d3/predictions" / f"{run_id}.npz"
            with np.load(pred_path, allow_pickle=False) as pred_archive:
                pred_ids = pred_archive["anchor_id"].astype("U96")
                order = int(np.flatnonzero(pred_ids == sample_id)[0])
                pred = pred_archive["predictions_native"][order]
            y = targets[index, :horizon]
            future_t = np.arange(1, horizon + 1, dtype=np.float32) / 30.0
            for dim, past in ((0, hist0), (14, hist14)):
                color = linecolors[dim]
                ax.plot(t_hist, past, color=color, alpha=0.28, linewidth=1.0, linestyle=":")
                ax.plot(future_t, y[:, dim], color=color, linewidth=1.5, linestyle="-",
                        label=f"GT action[{dim}]" if row_i == 0 and col_i == 0 else None)
                ax.plot(future_t, pred[:, dim], color=color, linewidth=1.5, linestyle="--",
                        label=f"TS11-D action[{dim}]" if row_i == 0 and col_i == 0 else None)
                ax.scatter([0.0], [anchor[dim]], color=color, marker="o", s=18, zorder=5)
            ax.axvline(0.0, color="#444444", linewidth=0.8)
            ax.set_title(f"{sample_id} | ep={rep['episode_index']} task={rep['original_task_index']} | H{horizon}", fontsize=8.5)
            ax.set_xlabel("Seconds relative to anchor (past < 0, future > 0)")
            ax.set_ylabel("Source-native command coordinate")
            ax.grid(color="#e2e2e2", linewidth=0.5)
            if row_i == 0 and col_i == 0:
                ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")
    fig.suptitle("Outcome-blind fixed-anchor trajectories: past history, anchor, GT and TS11-D", fontsize=14)
    fig.text(0.5, -0.015,
             "Selected IDs are the three anchors locked before prediction outcomes. Dotted segments are observed history; solid is GT; dashed is TS11-D (seed 20261001); colored circle marks the native last-command anchor. "
             "Action-vector region semantics are unresolved; coordinates are not physical joint or Cartesian units.",
             ha="center", va="top", fontsize=8.5)
    return save(fig, "FIG_FIXED_TRAJECTORIES_HISTORY_ANCHOR_V1")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "train_dev": plot_train_dev(),
        "seed_sensitivity": plot_seed_sensitivity(),
        "fixed_trajectories": plot_fixed_trajectories(),
    }
    manifest = {
        "status": "generated_pending_agent_visual_review",
        "publisher_compliance": "not assessed; no venue specified",
        "accessibility_certification": False,
        "figures": artifacts,
        "sources": {
            "train_dev_summary": {"path": str(OUT / "TRAIN_DEV_SIX_CONFIG_SUMMARY.csv"), "sha256": sha(OUT / "TRAIN_DEV_SIX_CONFIG_SUMMARY.csv")},
            "seed_sensitivity": {"path": str(OUT / "SEED_SENSITIVITY_PRIMARY.csv"), "sha256": sha(OUT / "SEED_SENSITIVITY_PRIMARY.csv")},
            "trajectory_selection": {"path": str(selection_path if "selection_path" in locals() else EQ / "protocol/D3_EQ_TRAJECTORY_SELECTION.json"),
                                     "sha256": sha(EQ / "protocol/D3_EQ_TRAJECTORY_SELECTION.json")},
        },
        "sample_trajectory_ids": [x["anchor_id"] for x in json.loads((EQ / "protocol/D3_EQ_TRAJECTORY_SELECTION.json").read_text())["representatives"]],
        "coordinate_note": "source-native action vector coordinates; not physical joint or Cartesian quantities",
        "final_test_access": 0,
    }
    (OUT / "FIGURE_MANIFEST_V1.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "figure_count": len(artifacts)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
