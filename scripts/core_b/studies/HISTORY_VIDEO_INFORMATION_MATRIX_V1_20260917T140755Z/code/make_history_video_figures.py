#!/usr/bin/env python3
"""Create six development-audit figures for the History×Video study."""

from __future__ import annotations
import os

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
INPUT = ROOT / "inputs/HISTORY_VIDEO_INPUTS.npz"
MOTION_STRATA = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z/MOTION_STRATA_MANIFEST.csv"
REPRESENTATIVE = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z/data/V2_TRAJECTORY_REPRESENTATIVE_EPISODES.csv"
sys.path.insert(0, str(ROOT / "code"))
from run_dev_inference_and_analysis import metric_arrays  # noqa: E402


COLORS = {"blue": "#0072B2", "orange": "#E69F00", "vermilion": "#D55E00", "green": "#009E73", "purple": "#CC79A7", "gray": "#666666", "black": "#111111", "light_blue": "#56B4E9"}
CELL_ORDER = ["HI-S-LAST", "HI-S-HIST", "HI-F-LAST", "HI-F-HIST"]
MODEL_ORDER = ["PERSISTENCE", "FIXED_HYBRID", *CELL_ORDER, "SLOPE_BASELINE"]
MODEL_LABELS = {"PERSISTENCE": "Persistence", "FIXED_HYBRID": "Fixed hybrid", "HI-S-LAST": "S-LAST", "HI-S-HIST": "S-HIST", "HI-F-LAST": "F-LAST", "HI-F-HIST": "F-HIST", "SLOPE_BASELINE": "Slope"}
MODEL_COLORS = {"PERSISTENCE": COLORS["gray"], "FIXED_HYBRID": COLORS["black"], "HI-S-LAST": COLORS["blue"], "HI-S-HIST": COLORS["light_blue"], "HI-F-LAST": COLORS["green"], "HI-F-HIST": COLORS["orange"], "SLOPE_BASELINE": COLORS["vermilion"]}
STEPS = 30


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(ROOT / "figures" / f"{stem}.png", dpi=300, facecolor="white")
    fig.savefig(ROOT / "figures" / f"{stem}.pdf", dpi=300, facecolor="white")
    plt.close(fig)


def caption(stem: str, body: str) -> None:
    path = ROOT / "figure_manifests" / f"{stem}.caption.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.strip() + "\n", encoding="utf-8")


def load_all() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    with np.load(INPUT, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    with np.load(REPO / "reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/runs/E-F-SP-G/predictions_best.npz", allow_pickle=False) as arm_archive:
        locked = {key: arm_archive[key].copy() for key in arm_archive.files}
    target_native = data["targets_native"][data["split_code"] == 1].astype(np.float32)
    target_std = data["targets_standardized"][data["split_code"] == 1].astype(np.float32)
    persistence_native = data["persistence_native"][data["split_code"] == 1].astype(np.float32)
    persistence_std = data["persistence_standardized"][data["split_code"] == 1].astype(np.float32)
    arm_native = locked["predictions_native"][:, :, 14:].astype(np.float32)
    arm_std = locked["predictions_standardized"][:, :, 14:].astype(np.float32)
    dev = data["split_code"] == 1
    anchor = persistence_native[:, 0, :14]
    native: dict[str, np.ndarray] = {"PERSISTENCE": persistence_native, "FIXED_HYBRID": np.concatenate((persistence_native[:, :, :14], arm_native), axis=2)}
    standardized: dict[str, np.ndarray] = {"PERSISTENCE": persistence_std, "FIXED_HYBRID": np.concatenate((persistence_std[:, :, :14], arm_std), axis=2)}
    for cell in CELL_ORDER:
        with np.load(ROOT / "runs" / cell / "formal/predictions_last.npz", allow_pickle=False) as archive:
            native[cell] = archive["predictions_full_native"].copy()
            standardized[cell] = archive["predictions_full_standardized"].copy()
    with np.load(ROOT / "analysis/SLOPE_BASELINE_DEV.npz", allow_pickle=False) as archive:
        native["SLOPE_BASELINE"] = archive["predictions_native"].copy()
        standardized["SLOPE_BASELINE"] = archive["predictions_standardized"].copy()
    metrics = {name: metric_arrays(native[name], standardized[name], target_native, target_std, anchor) for name in native}
    return data, {"target_native": target_native, "target_std": target_std, "persistence_native": persistence_native, "persistence_std": persistence_std, "arm_native": arm_native, "arm_std": arm_std}, {name: {key: value for key, value in metrics[name].items()} for name in metrics}


def main() -> int:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    data, arrays, metrics = load_all()
    dev_mask = data["split_code"] == 1
    dev_ids = data["episode_index"][dev_mask].astype(np.int64)
    target_native = arrays["target_native"]
    persistence_native = arrays["persistence_native"]

    # 1. Input timeline and available history length.
    position = 0
    selected_ts = data["history_selected_timestamps"][dev_mask][position]
    query_ts = data["history_query_timestamps"][dev_mask][position]
    target_ts = data["target_timestamps"][dev_mask][position]
    t0 = float(selected_ts[-1])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), layout="constrained")
    axes[0].vlines(selected_ts - t0, 0.0, 1.0, color=COLORS["blue"], linewidth=1.0, alpha=0.8, label="selected past row")
    axes[0].scatter(query_ts - t0, np.full(32, 1.04), color=COLORS["light_blue"], marker="|", s=35, label="query time")
    axes[0].axvline(0, color=COLORS["black"], linestyle="--", label="t0 = frame 119")
    axes[0].axvline(target_ts[0] - t0, color=COLORS["vermilion"], linestyle=":", label="first future target")
    axes[0].set(xlabel="Relative time to t0 (s)", yticks=[], title=f"Past/future timestamp seam (dev ID {int(dev_ids[position])})")
    axes[0].legend(frameon=False, loc="lower left")
    spans = data["history_selected_timestamps"][dev_mask][:, -1] - data["history_selected_timestamps"][dev_mask][:, 0]
    axes[1].hist(spans, bins=10, color=COLORS["blue"], edgecolor=COLORS["black"])
    axes[1].axvline(float(np.mean(spans)), color=COLORS["vermilion"], linestyle="--", label=f"mean={np.mean(spans):.4f}s")
    axes[1].set(xlabel="Actual selected-history span (s)", ylabel="Episodes", title="Available history span across 614 dev episodes")
    axes[1].legend(frameon=False)
    save(fig, "FIG01_INPUT_TIMELINE_HISTORY_LENGTH")
    write_csv(ROOT / "figure_data/FIG01_INPUT_TIMELINE_HISTORY_LENGTH.csv", [{"episode_index": int(dev_ids[position]), "series": "selected_past", "relative_seconds": float(value - t0)} for value in selected_ts] + [{"episode_index": int(dev_ids[position]), "series": "query_time", "relative_seconds": float(value - t0)} for value in query_ts] + [{"episode_index": int(dev_ids[position]), "series": "future_target", "relative_seconds": float(value - t0)} for value in target_ts] + [{"episode_index": int(episode), "series": "actual_selected_span", "relative_seconds": float(span)} for episode, span in zip(dev_ids, spans)])
    caption("FIG01_INPUT_TIMELINE_HISTORY_LENGTH", """# Input timestamp timeline and available history

The left panel shows the fixed 32 query times, the latest legal past Parquet row selected at or before each query, the anchor frame 119, and the first future target for one fixed development episode. The right panel reports actual selected-history spans for all 614 development episodes. No future row is used as a history input.""")

    # 2. Four-cell level and FD comparison.
    main_rows = {row["model"]: row for row in read_csv(ROOT / "analysis/MAIN_RESULTS.csv")}
    order = ["PERSISTENCE", "FIXED_HYBRID", *CELL_ORDER, "SLOPE_BASELINE"]
    labels = [MODEL_LABELS[name] for name in order]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), layout="constrained")
    x = np.arange(len(order))
    axes[0].bar(x, [float(main_rows[name]["hand_mae_std"]) for name in order], color=[MODEL_COLORS[name] for name in order], edgecolor=COLORS["black"])
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Hand standardized MAE", title="Level accuracy (lower is better)")
    axes[0].tick_params(axis="x", rotation=30)
    pfd = float(main_rows["PERSISTENCE"]["hand_fd_native"])
    axes[1].bar(x, [float(main_rows[name]["hand_fd_native"]) for name in order], color=[MODEL_COLORS[name] for name in order], edgecolor=COLORS["black"])
    axes[1].axhline(pfd, color=COLORS["gray"], linestyle="--", label="Persistence")
    axes[1].axhline(1.01 * pfd, color=COLORS["vermilion"], linestyle=":", label="+1% margin")
    axes[1].set(xticks=x, xticklabels=labels, ylabel="Hand native FD error", title="Temporal fidelity (lower is better)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(frameon=False)
    save(fig, "FIG02_FOUR_CELL_LEVEL_FD")
    write_csv(ROOT / "figure_data/FIG02_FOUR_CELL_LEVEL_FD.csv", [{"model": name, "label": MODEL_LABELS[name], "hand_mae_std": float(main_rows[name]["hand_mae_std"]), "hand_fd_native": float(main_rows[name]["hand_fd_native"])} for name in order])
    caption("FIG02_FOUR_CELL_LEVEL_FD", """# Four-cell level and FD comparison

Hand standardized full-horizon MAE and native 29-step future-to-future FD error are shown for Persistence, the fixed Arm-learned/Hand-Persistence hybrid, the four new cells, and the no-fit slope diagnostic. The +1% line is a project operating margin, not a physical law.""")

    # 3. Four fixed matched effects with bootstrap intervals.
    pair_rows = [row for row in read_csv(ROOT / "analysis/PAIRED_EFFECTS.csv") if row["family"] == "four_fixed_contrasts_co_primary"]
    contrasts = ["S-HIST_vs_S-LAST", "F-HIST_vs_F-LAST", "F-HIST_vs_S-HIST", "F-LAST_vs_S-LAST"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), layout="constrained")
    for axis, metric_name, title, color in ((axes[0], "hand_mae_std", "Hand level effect", COLORS["blue"]), (axes[1], "hand_fd_native", "Hand FD effect", COLORS["vermilion"])):
        rows = {row["contrast"]: row for row in pair_rows if row["metric"] == metric_name}
        values = np.asarray([float(rows[name]["effect_control_minus_candidate"]) for name in contrasts])
        lower = values - np.asarray([float(rows[name]["ci_low"]) for name in contrasts])
        upper = np.asarray([float(rows[name]["ci_high"]) for name in contrasts]) - values
        axis.errorbar(np.arange(4), values, yerr=np.vstack((lower, upper)), fmt="o", color=color, ecolor=COLORS["black"], capsize=3, linewidth=1.2)
        axis.axhline(0, color=COLORS["gray"], linewidth=1)
        axis.set(xticks=np.arange(4), xticklabels=["S hist\nvs last", "F hist\nvs last", "F hist\nvs S hist", "F last\nvs S last"], ylabel="Control error − candidate error", title=title)
        for index, name in enumerate(contrasts):
            axis.text(index, float(rows[name]["ci_high"]) + (abs(values).max() * 0.05 if values.size else 0), f"Holm {float(rows[name]['holm_p']):.3g}", ha="center", va="bottom", fontsize=6.5, rotation=45)
    save(fig, "FIG03_MATCHED_EFFECTS")
    write_csv(ROOT / "figure_data/FIG03_MATCHED_EFFECTS.csv", pair_rows)
    caption("FIG03_MATCHED_EFFECTS", """# Matched History×Video effects

Each point is a paired episode effect (control error minus candidate error) with a 95% percentile bootstrap interval from 10,000 episode resamples. The eight co-primary tests are the four pre-registered contrasts crossed with Hand level MAE and native FD; Holm-adjusted sign-flip p-values are annotated. Positive values favor the candidate.""")

    # 4. Horizon, joint, task and motion error structure.
    temporal_rows = read_csv(ROOT / "analysis/TEMPORAL_METRICS.csv")
    temporal = {(row["model"], row["horizon"]): float(row["hand_mae_std"]) for row in temporal_rows}
    per_joint = read_csv(ROOT / "analysis/PER_JOINT_METRICS.csv")
    joint = {(row["model"], int(row["hand_joint"])): float(row["hand_mae_std"]) for row in per_joint}
    per_task = read_csv(ROOT / "analysis/PER_TASK_METRICS.csv")
    task_p = {int(row["task_index"]): float(row["hand_mae_std"]) for row in per_task if row["model"] == "PERSISTENCE"}
    task_f = {int(row["task_index"]): float(row["hand_mae_std"]) for row in per_task if row["model"] == "HI-F-HIST"}
    motion_rows = read_csv(ROOT / "analysis/MOTION_STRATA_RESULTS.csv")
    motion = {(row["model"], row["motion_stratum"]): float(row["hand_mae_std"]) for row in motion_rows}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), layout="constrained")
    horizons = ["early", "middle", "late", "endpoint"]
    for name in ("PERSISTENCE", *CELL_ORDER):
        axes[0, 0].plot(horizons, [temporal[(name, horizon)] for horizon in horizons], marker="o", color=MODEL_COLORS[name], label=MODEL_LABELS[name])
    axes[0, 0].set(ylabel="Hand standardized MAE", title="Horizon profile")
    axes[0, 0].legend(frameon=False, ncol=2)
    joint_gain = np.asarray([joint[("PERSISTENCE", index)] - joint[("HI-F-HIST", index)] for index in range(14)])[None, :]
    limit = max(abs(float(joint_gain.min())), abs(float(joint_gain.max())), 1e-6)
    image = axes[0, 1].imshow(joint_gain, aspect="auto", cmap="RdBu", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
    axes[0, 1].set(yticks=[0], yticklabels=["F-HIST"], xticks=np.arange(14), xticklabels=[str(i) for i in range(14)], xlabel="Hand joint", title="Per-joint gain vs Persistence")
    for index, value in enumerate(joint_gain[0]):
        axes[0, 1].text(index, 0, f"{value:.3f}", ha="center", va="center", fontsize=6, color=COLORS["black"])
    fig.colorbar(image, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Persistence − F-HIST")
    task_ids = sorted(task_p)
    task_gain = np.asarray([task_p[index] - task_f[index] for index in task_ids])
    task_n = {int(row["task_index"]): int(row["N_episode"]) for row in per_task if row["model"] == "HI-F-HIST"}
    sizes = np.asarray([task_n[index] for index in task_ids])
    axes[1, 0].scatter(task_ids, task_gain, c=sizes, cmap="viridis", s=20 + 5 * sizes, edgecolor=COLORS["black"], linewidth=0.25)
    axes[1, 0].axhline(0, color=COLORS["gray"], linewidth=1)
    axes[1, 0].set(xlabel="Task index", ylabel="Persistence − F-HIST MAE", title="All development tasks")
    for index, gain in zip(task_ids, task_gain):
        if gain == task_gain.min() or gain == task_gain.max():
            axes[1, 0].annotate(str(index), (index, gain), xytext=(3, 3), textcoords="offset points", fontsize=6)
    strata = ["low", "mid", "high"]
    x = np.arange(3)
    axes[1, 1].bar(x - 0.18, [motion[("PERSISTENCE", s)] for s in strata], width=0.36, color=COLORS["gray"], label="Persistence")
    axes[1, 1].bar(x + 0.18, [motion[("HI-F-HIST", s)] for s in strata], width=0.36, color=COLORS["orange"], label="F-HIST")
    axes[1, 1].set(xticks=x, xticklabels=strata, ylabel="Hand standardized MAE", title="Fixed old motion strata")
    axes[1, 1].legend(frameon=False)
    save(fig, "FIG04_HORIZON_JOINT_TASK_MOTION")
    write_csv(ROOT / "figure_data/FIG04_HORIZON_JOINT_TASK_MOTION.csv", [{"panel": "horizon", "model": name, "horizon": horizon, "hand_mae_std": temporal[(name, horizon)]} for name in ("PERSISTENCE", *CELL_ORDER) for horizon in horizons] + [{"panel": "joint", "joint": index, "gain": float(joint_gain[0, index])} for index in range(14)] + [{"panel": "task", "task_index": index, "N_episode": task_n[index], "gain": float(gain)} for index, gain in zip(task_ids, task_gain)] + [{"panel": "motion", "model": name, "stratum": stratum, "hand_mae_std": motion[(name, stratum)]} for name in ("PERSISTENCE", "HI-F-HIST") for stratum in strata])
    caption("FIG04_HORIZON_JOINT_TASK_MOTION", """# Error structure by horizon, joint, task and motion stratum

The four panels retain the full declared development views: horizon bins are early/middle/late/endpoint; all 14 Hand joints are shown; every task is plotted with its support size encoded by point size; and the fixed old low/mid/high motion strata are compared without redefining thresholds.""")

    # 5. Fixed representative trajectories.
    representative_rows = read_csv(REPRESENTATIVE)
    unique_reps: list[tuple[int, str]] = []
    for row in representative_rows:
        item = (int(row["episode_id"]), row["motion_stratum"])
        if item not in unique_reps:
            unique_reps.append(item)
    chosen_reps: list[tuple[int, str]] = []
    for stratum in ("low", "mid", "high"):
        chosen_reps.append(next(item for item in unique_reps if item[1] == stratum))
    id_to_position = {int(episode): index for index, episode in enumerate(dev_ids.tolist())}
    slope = np.load(ROOT / "analysis/SLOPE_BASELINE_DEV.npz", allow_pickle=False)["predictions_native"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9), layout="constrained", sharey=True)
    trajectory_rows: list[dict[str, object]] = []
    for axis, (episode, stratum) in zip(axes, chosen_reps):
        index = id_to_position[episode]
        series = [("target", target_native[index, :, 0], COLORS["black"], "-", 2.0), ("Persistence", persistence_native[index, :, 0], COLORS["gray"], "--", 1.2), ("Slope", slope[index, :, 0], COLORS["vermilion"], ":", 1.2)]
        for cell in CELL_ORDER:
            series.append((MODEL_LABELS[cell], np.load(ROOT / "runs" / cell / "formal/predictions_last.npz", allow_pickle=False)["predictions_hand_native"][index, :, 0], MODEL_COLORS[cell], "-" if cell.endswith("HIST") else "-.", 1.0))
        for name, values, color, linestyle, linewidth in series:
            axis.plot(np.arange(STEPS), values, color=color, linestyle=linestyle, linewidth=linewidth, label=name)
            for time_index, value in enumerate(values):
                trajectory_rows.append({"episode_id": episode, "motion_stratum": stratum, "time_index": time_index, "hand_joint": 0, "series": name, "native_value": float(value)})
        axis.set_title(f"{stratum}; episode {episode}")
        axis.set_xlabel("Future step")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Native command value (unit unverified)")
    axes[-1].legend(frameon=False, fontsize=6.5, loc="best")
    fig.suptitle("Fixed representative episode trajectories (Hand joint 0)", fontsize=11)
    save(fig, "FIG05_FIXED_REPRESENTATIVE_TRAJECTORIES")
    write_csv(ROOT / "figure_data/FIG05_FIXED_REPRESENTATIVE_TRAJECTORIES.csv", trajectory_rows)
    caption("FIG05_FIXED_REPRESENTATIVE_TRAJECTORIES", """# Fixed representative episode trajectories

One episode from each fixed old low/mid/high motion stratum is shown. The episode identities come from the pre-existing ground-truth motion manifest, not from the new model errors. Lines include target, Persistence, the no-fit slope diagnostic, and all four new Hand predictions. Native units remain unverified.""")

    # 6. Failure cloud and aggregate Pareto view.
    p_metric = metrics["PERSISTENCE"]
    f_metric = metrics["HI-F-HIST"]
    motion_map = {int(row["episode_id"]): row["motion_stratum"] for row in read_csv(MOTION_STRATA)}
    strata_values = np.asarray([motion_map[int(episode)] for episode in dev_ids], dtype=object)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), layout="constrained")
    for stratum, color in (("low", COLORS["green"]), ("mid", COLORS["blue"]), ("high", COLORS["vermilion"])):
        mask = strata_values == stratum
        axes[0].scatter(p_metric["hand_mae_std"][mask], f_metric["hand_mae_std"][mask], s=15, alpha=0.55, color=color, label=stratum)
    lo = min(float(p_metric["hand_mae_std"].min()), float(f_metric["hand_mae_std"].min()))
    hi = max(float(p_metric["hand_mae_std"].max()), float(f_metric["hand_mae_std"].max()))
    axes[0].plot([lo, hi], [lo, hi], color=COLORS["gray"], linestyle="--", label="equal error")
    axes[0].set(xlabel="Persistence episode Hand MAE", ylabel="F-HIST episode Hand MAE", title="All 614 episodes: failure cloud")
    axes[0].legend(frameon=False)
    pareto_order = ["PERSISTENCE", *CELL_ORDER, "SLOPE_BASELINE"]
    for name in pareto_order:
        row = main_rows[name]
        x_value = float(row["hand_mae_std"])
        y_value = float(row["hand_fd_native"])
        axes[1].scatter(x_value, y_value, s=65, color=MODEL_COLORS[name], edgecolor=COLORS["black"], label=MODEL_LABELS[name])
        if name == "SLOPE_BASELINE":
            axes[1].annotate("Slope", (x_value, y_value), xytext=(5, 4), textcoords="offset points", fontsize=7)
    axes[1].set(xlabel="Aggregate Hand standardized MAE", ylabel="Aggregate Hand native FD error", title="Aggregate lower-left Pareto view")
    axes[1].legend(frameon=False, fontsize=7, loc="best")
    save(fig, "FIG06_FAILURE_CLOUD_PARETO")
    write_csv(ROOT / "figure_data/FIG06_FAILURE_CLOUD_PARETO.csv", [{"panel": "failure_cloud", "episode_id": int(episode), "motion_stratum": stratum, "persistence_hand_mae_std": float(p), "f_hist_hand_mae_std": float(f)} for episode, stratum, p, f in zip(dev_ids, strata_values, p_metric["hand_mae_std"], f_metric["hand_mae_std"])] + [{"panel": "pareto", "model": name, "hand_mae_std": float(main_rows[name]["hand_mae_std"]), "hand_fd_native": float(main_rows[name]["hand_fd_native"])} for name in pareto_order])
    caption("FIG06_FAILURE_CLOUD_PARETO", """# Failure cloud and aggregate Pareto view

The left panel retains all 614 development episodes and compares Persistence with F-HIST; colors identify the fixed old motion strata. The right panel is an aggregate lower-left-is-better view of level MAE versus native FD. It is descriptive and does not select a model by a post-hoc objective.""")

    manifest_rows = []
    for index in range(1, 7):
        stem = ["FIG01_INPUT_TIMELINE_HISTORY_LENGTH", "FIG02_FOUR_CELL_LEVEL_FD", "FIG03_MATCHED_EFFECTS", "FIG04_HORIZON_JOINT_TASK_MOTION", "FIG05_FIXED_REPRESENTATIVE_TRAJECTORIES", "FIG06_FAILURE_CLOUD_PARETO"][index - 1]
        manifest_rows.append({"figure_id": index, "stem": stem, "png": str(ROOT / "figures" / f"{stem}.png"), "pdf": str(ROOT / "figures" / f"{stem}.pdf"), "data_csv": str(ROOT / "figure_data" / f"{stem}.csv"), "caption": str(ROOT / "figure_manifests" / f"{stem}.caption.md"), "audience": "internal development audit", "final_test_access": 0})
    write_csv(ROOT / "figure_manifests/FIGURE_MANIFEST.csv", manifest_rows)
    (ROOT / "figure_manifests/FIGURE_MANIFEST.json").write_text(json.dumps({"status": "PASS_WITH_CAVEAT", "figure_count": 6, "rows": manifest_rows, "visual_qa_required": True, "final_test_access": 0}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "figures": 6, "final_test_access": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
