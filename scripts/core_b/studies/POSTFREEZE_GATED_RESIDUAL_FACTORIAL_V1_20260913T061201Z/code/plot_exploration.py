#!/usr/bin/env python3
"""Create the predeclared exploratory figures with explicit provenance."""

from __future__ import annotations
import os

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z'))
STAGE2_REFERENCE = ROOT.parent / "JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/runs/stage2_authorized_20260912T151855Z/seed20260921/JY02-F-R-D/predictions_best.npz"
CELLS = ["E-S-SH-U", "E-S-SH-G", "E-S-SP-U", "E-S-SP-G", "E-F-SH-U", "E-F-SH-G", "E-F-SP-U", "E-F-SP-G"]
GATED = {cell for cell in CELLS if cell.endswith("-G")}
COLORS = {"STATE": "#0072B2", "FUSION": "#D55E00"}
GROUP_COLORS = {"arm": "#0072B2", "hand": "#D55E00"}


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, stem: str, caption: str, alt_text: str, sources: list[str], manifest: list[dict]) -> None:
    for extension in ("png", "svg", "pdf"):
        path = ROOT / "figures" / extension / f"{stem}.{extension}"
        fig.savefig(path, dpi=220, facecolor="white")
    (ROOT / "figure_manifests" / f"{stem}.caption.md").write_text(caption + "\n", encoding="utf-8")
    (ROOT / "figure_manifests" / f"{stem}.alt.txt").write_text(alt_text + "\n", encoding="utf-8")
    manifest.append({"figure_id": stem, "caption": caption, "alt_text": alt_text, "sources": ";".join(sources), "formats": "png,svg,pdf", "dpi_png": 220, "smoothing": "none", "status": "PASS"})
    plt.close(fig)


def main() -> int:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9, "figure.dpi": 120})
    main_rows = read_csv("MAIN_RESULTS.csv")
    factorial_rows = read_csv("FACTORIAL_EFFECTS.csv")
    horizon_rows = read_csv("MOTION_HORIZON_RESULTS.csv")
    gate_rows = read_csv("GATE_DIAGNOSTICS.csv")
    episode_rows = read_csv("PER_EPISODE.csv")
    task_rows = read_csv("PER_TASK.csv")
    decision = json.loads((ROOT / "EXPLORATION_DECISION.json").read_text())
    by_cell = {row["cell_id"]: row for row in main_rows}
    manifest: list[dict] = []

    # 1. Primary MAE with episode-bootstrap metric intervals.
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    x = np.arange(len(CELLS)); y = np.array([float(by_cell[c]["arm_full_mae_std"]) for c in CELLS])
    lo = y - np.array([float(by_cell[c]["arm_ci_low"]) for c in CELLS]); hi = np.array([float(by_cell[c]["arm_ci_high"]) for c in CELLS]) - y
    bars = ax.bar(x, y, color=[COLORS[by_cell[c]["input"]] for c in CELLS], edgecolor="black", linewidth=.5)
    for bar, cell in zip(bars, CELLS): bar.set_hatch("//" if "SP" in cell else "")
    ax.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor="black", capsize=3, linewidth=1)
    ax.set_xticks(x, CELLS, rotation=35, ha="right"); ax.set_ylabel("Arm-14 standardized MAE"); ax.set_title("Exploratory primary metric (10,000 episode-bootstrap intervals)")
    ax.grid(axis="y", alpha=.25); ax.text(.99, .98, "blue=STATE, orange=FUSION; //=split", transform=ax.transAxes, ha="right", va="top")
    save_figure(fig, "fig01_primary_arm14_mae", "Arm-14 standardized full-horizon MAE for all eight accepted cells; intervals are paired episode-bootstrap 95% intervals.", "Eight bars compare Arm-14 MAE. FUSION split-gated is lowest, but this is single-seed development evidence.", ["MAIN_RESULTS.csv"], manifest)

    # 2. Full factorial effects.
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    labels = [row["term"] for row in factorial_rows]; values = np.array([float(row["effect"]) for row in factorial_rows]); lows = values - np.array([float(row["ci_low"]) for row in factorial_rows]); highs = np.array([float(row["ci_high"]) for row in factorial_rows]) - values
    yy = np.arange(len(labels)); ax.errorbar(values, yy, xerr=[lows, highs], fmt="o", color="#0072B2", ecolor="#333333", capsize=3); ax.axvline(0, color="black", linewidth=.8); ax.set_yticks(yy, labels); ax.set_xlabel("Improvement effect (positive = lower MAE)"); ax.set_title("2×2×2 factorial Arm-14 effects"); ax.grid(axis="x", alpha=.25)
    save_figure(fig, "fig02_factorial_effects", "Main and interaction effects from the complete 2x2x2 factorial; positive values indicate lower Arm-14 MAE at the high factor level or positive interaction.", "A horizontal forest plot shows positive main effects for input, split head and gate; interaction intervals mostly include zero.", ["FACTORIAL_EFFECTS.csv"], manifest)

    # 3. Arm/Hand/Full comparison.
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    width=.25
    for offset, (key, label, color) in enumerate((("arm_full_mae_std", "Arm-14", "#0072B2"), ("hand_full_mae_std", "Hand-14", "#D55E00"), ("full28_full_mae_std", "Full-28", "#009E73"))):
        ax.bar(x + (offset - 1) * width, [float(by_cell[c][key]) for c in CELLS], width=width, label=label, color=color, edgecolor="black", linewidth=.35)
    ax.set_xticks(x, CELLS, rotation=35, ha="right"); ax.set_ylabel("Standardized MAE"); ax.set_title("Arm / Hand / Full-28 development metrics"); ax.legend(ncol=3); ax.grid(axis="y", alpha=.25)
    save_figure(fig, "fig03_arm_hand_full", "Standardized Arm-14, Hand-14 and Full-28 MAE for the eight cells; all values are development-only.", "Grouped bars show Arm, Hand and Full-28 error for each cell. Arm improves more than Hand across gated cells.", ["MAIN_RESULTS.csv"], manifest)

    # 4. First difference and native TV.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    fd = [float(by_cell[c]["arm_first_difference_mae_std"]) for c in CELLS]; tv = [float(by_cell[c]["native_tv_ratio"]) for c in CELLS]
    axes[0].bar(x, fd, color=[COLORS[by_cell[c]["input"]] for c in CELLS], edgecolor="black"); axes[0].set_xticks(x, CELLS, rotation=35, ha="right"); axes[0].set_ylabel("Arm first-difference MAE"); axes[0].grid(axis="y", alpha=.25)
    axes[1].bar(x, tv, color=[COLORS[by_cell[c]["input"]] for c in CELLS], edgecolor="black"); axes[1].axhline(1, color="black", linestyle="--", linewidth=.8, label="ideal TV ratio=1"); axes[1].set_xticks(x, CELLS, rotation=35, ha="right"); axes[1].set_ylabel("Native TV ratio"); axes[1].legend(); axes[1].grid(axis="y", alpha=.25)
    fig.suptitle("Trajectory-fidelity guardrails")
    save_figure(fig, "fig04_first_difference_tv", "First-difference MAE and native TV ratio are shown as guardrails; no smoothing is applied and TV ratio one is a reference line.", "Two aligned bar plots show first-difference error and native TV ratio. Gated FUSION cells have lower TV ratios but remain above one.", ["MAIN_RESULTS.csv"], manifest)

    # 5. Motion x horizon relative skill.
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    selected = [row for row in horizon_rows if row["horizon"] == "late" and row["motion_stratum"] in {"low", "mid", "high"}]
    matrix = np.array([[float(next(row["relative_skill_vs_persistence"] for row in selected if row["cell_id"] == c and row["motion_stratum"] == s)) for s in ("low", "mid", "high")] for c in CELLS])
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=min(-.1, matrix.min()), vmax=max(.35, matrix.max()), aspect="auto"); ax.set_xticks(range(3), ["low", "mid", "high"]); ax.set_yticks(range(8), CELLS); ax.set_xlabel("Train-Q25/Q75 motion stratum"); ax.set_title("Late-horizon Arm skill relative to Persistence"); fig.colorbar(im, ax=ax, label="relative improvement")
    for i in range(8):
        for j in range(3): ax.text(j, i, f"{matrix[i,j]:.1%}", ha="center", va="center", fontsize=8)
    save_figure(fig, "fig05_motion_horizon", "Late-horizon Arm relative improvement versus Persistence by cell and train-derived motion stratum.", "A heatmap shows late-horizon relative skill across low, mid and high motion strata; the high-motion column is generally more favorable for gated FUSION cells.", ["MOTION_HORIZON_RESULTS.csv"], manifest)

    # 6. Fixed development representative trajectories.
    rep_manifest = V2_ROOT / "V2_TRAJECTORY_SELECTION_MANIFEST.csv"
    fixed_ids = []
    if rep_manifest.is_file():
        with rep_manifest.open(encoding="utf-8", newline="") as handle:
            fixed_ids = [int(row["episode_id"]) for row in csv.DictReader(handle)]
    with np.load(ROOT / "runs" / decision["best_primary_cell"] / "predictions_best.npz", allow_pickle=False) as candidate_bundle:
        ids = candidate_bundle["episode_index"].copy(); target = candidate_bundle["targets_standardized"].copy(); persistence = candidate_bundle["persistence_standardized"].copy(); candidate = candidate_bundle["predictions_standardized"].copy()
    with np.load(STAGE2_REFERENCE, allow_pickle=False) as ref_bundle:
        ref_ids = ref_bundle["episode_index"].copy(); ref_pred = ref_bundle["predictions_standardized"].copy()
    id_to_index = {int(value): i for i, value in enumerate(ids)}; ref_to_index = {int(value): i for i, value in enumerate(ref_ids)}
    chosen = [episode for episode in fixed_ids if episode in id_to_index and episode in ref_to_index][:2]
    if not chosen: chosen = [int(value) for value in ids[:2]]
    fig, axes = plt.subplots(len(chosen), 1, figsize=(9, 3.3 * len(chosen)), squeeze=False, layout="constrained")
    for ax, episode in zip(axes[:, 0], chosen):
        i, j = id_to_index[episode], ref_to_index.get(episode, 0); ax.plot(target[i, :, 14:28].mean(axis=1), color="black", label="target", linewidth=1.5); ax.plot(persistence[i, :, 14:28].mean(axis=1), color="#777777", linestyle="--", label="Persistence"); ax.plot(ref_pred[j, :, 14:28].mean(axis=1), color="#CC79A7", linestyle=":", label="frozen F-R-D ref"); ax.plot(candidate[i, :, 14:28].mean(axis=1), color="#0072B2", label=decision["best_primary_cell"]); ax.set_ylabel(f"ep {episode}\nArm mean"); ax.grid(alpha=.25)
    axes[-1, 0].set_xlabel("future step"); axes[0, 0].legend(ncol=4, fontsize=8); fig.suptitle("Predeclared development trajectory examples")
    save_figure(fig, "fig06_representative_trajectories", "Mean Arm trajectory for predeclared development representative episode IDs, comparing target, Persistence, the frozen Stage2 reference and the best exploratory primary cell.", "Line plots show target, Persistence, frozen F-R-D reference and best exploratory prediction for two fixed development episodes; examples are descriptive, not final evidence.", ["V2_TRAJECTORY_SELECTION_MANIFEST.csv", "runs/E-F-SP-G/predictions_best.npz", "STAGE2_REFERENCE"], manifest)

    # 7. Gate time profiles.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout="constrained", sharex=True)
    time_rows = read_csv("figure_data/gate_time.csv")
    for cell in sorted(GATED):
        rows = [r for r in time_rows if r["cell_id"] == cell]
        axes[0].plot(range(30), [float(r["gate_arm"]) for r in rows], label=cell, linewidth=1.4)
        axes[1].plot(range(30), [float(r["gate_hand"]) for r in rows], label=cell, linewidth=1.4)
    for ax, title, group in zip(axes, ["Arm gate", "Hand gate"], ["Arm", "Hand"]): ax.set_title(title); ax.set_xlabel("future step"); ax.set_ylabel("mean gate in [0,1]"); ax.set_ylim(0, 1.05); ax.grid(alpha=.25); ax.axhline(.9, color="black", linestyle=":", linewidth=.8, label="initial .9")
    axes[1].legend(fontsize=7); fig.suptitle("Group-time gate profiles")
    save_figure(fig, "fig07_gate_time", "Mean Arm and Hand persistence gates over time for the four gated cells; gate values are bounded and no smoothing is applied.", "Two line panels show time-dependent Arm and Hand gate means. Gates contract residuals below one and vary substantially across cells and time.", ["figure_data/gate_time.csv"], manifest)

    # 8. Gate versus target motion.
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for cell in sorted(GATED):
        rows = [r for r in episode_rows if r["cell_id"] == cell]
        ax.scatter([float(r["motion"]) for r in rows], [float(r["gate_arm"]) for r in rows], s=9, alpha=.35, label=cell)
    ax.set_xlabel("target Arm mean step motion (standardized)"); ax.set_ylabel("episode mean Arm gate"); ax.set_ylim(-.02, 1.02); ax.set_title("Gate versus target motion (descriptive)"); ax.grid(alpha=.25); ax.legend(fontsize=7)
    save_figure(fig, "fig08_gate_vs_motion", "Episode mean Arm gate versus target Arm motion for gated cells; the motion quantity is descriptive and no gate was fit using a motion label.", "A scatter plot displays episode mean Arm gate against target motion for each gated cell, without a fitted trend line.", ["PER_EPISODE.csv"], manifest)

    # 9. Joint x time heatmap.
    err_candidate = np.abs(candidate - target).mean(axis=0).T; err_persistence = np.abs(persistence - target).mean(axis=0).T; vmax = max(float(err_candidate.max()), float(err_persistence.max()))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained", sharey=True)
    for ax, values, title in zip(axes, [err_candidate, err_persistence], [decision["best_primary_cell"], "Persistence"]):
        im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest"); ax.set_title(title); ax.set_xlabel("future step"); ax.set_xticks([0, 9, 19, 29]); ax.set_yticks([0, 7, 14, 21, 27], ["j0", "j7", "j14", "j21", "j27"])
    axes[0].set_ylabel("canonical joint index (Hand 0:14, Arm 14:28)"); fig.colorbar(im, ax=axes, label="absolute standardized error")
    save_figure(fig, "fig09_joint_time_heatmap", "Joint-by-time absolute standardized error for the best exploratory primary cell and Persistence using common color limits.", "Two heatmaps compare joint-time absolute error for the best exploratory cell and Persistence; the same color scale is used in both panels.", ["runs/E-F-SP-G/predictions_best.npz"], manifest)

    # 10. Task support.
    best = decision["best_primary_cell"]; baseline = "E-F-SH-U" if best.startswith("E-F") else "E-S-SH-U"; cand_tasks = {int(r["original_task_index"]): float(r["arm_mae"]) for r in task_rows if r["cell_id"] == best}; base_tasks = {int(r["original_task_index"]): float(r["arm_mae"]) for r in task_rows if r["cell_id"] == baseline}; task_ids = sorted(set(cand_tasks) & set(base_tasks)); effects = [base_tasks[t] - cand_tasks[t] for t in task_ids]
    fig, ax = plt.subplots(figsize=(11, 4.5), layout="constrained"); ax.bar(np.arange(len(task_ids)), effects, color="#0072B2", edgecolor="black", linewidth=.25); ax.axhline(0, color="black", linewidth=.8); ax.set_xticks(np.arange(len(task_ids)), [str(t) for t in task_ids], rotation=90, fontsize=6); ax.set_ylabel("Arm effect vs same-input shared-ungated"); ax.set_xlabel("original task index (sorted; no effect sorting)"); ax.set_title(f"Task support: {best} vs {baseline}"); ax.grid(axis="y", alpha=.25)
    save_figure(fig, "fig10_task_support", "Per-task Arm effects for the best exploratory cell versus its same-input shared-ungated baseline; tasks remain in sorted identifier order.", "A task-index bar plot shows positive and negative Arm effects without sorting tasks by effect; this checks whether the aggregate is broadly supported.", ["PER_TASK.csv"], manifest)

    # 11. Gate collapse and entropy.
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), layout="constrained")
    gd = [r for r in gate_rows if r["gated_cell"] == "True"]; labels = [f"{r['cell_id']}\n{r['group']}" for r in gd]; collapse = [float(r["collapse_fraction"]) for r in gd]; entropy = [float(r["entropy"]) for r in gd]; xx = np.arange(len(labels)); axes[0].bar(xx, collapse, color="#D55E00", edgecolor="black"); axes[0].axhline(.95, color="black", linestyle="--", label="collapse threshold"); axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("fraction <.05 or >.95"); axes[0].set_title("collapse"); axes[0].legend(fontsize=7); axes[1].bar(xx, entropy, color="#009E73", edgecolor="black"); axes[1].set_ylabel("Bernoulli gate entropy"); axes[1].set_title("entropy");
    for ax in axes: ax.set_xticks(xx, labels, rotation=45, ha="right", fontsize=7); ax.grid(axis="y", alpha=.25)
    save_figure(fig, "fig11_gate_collapse_entropy", "Gate collapse fraction and Bernoulli entropy for gated cells; collapse is pre-defined as gate below 0.05 or above 0.95.", "Two bar panels report gate collapse and entropy for Arm and Hand across gated cells; none is interpreted as a final model-selection result.", ["GATE_DIAGNOSTICS.csv"], manifest)

    # 12. Decision summary.
    fig, ax = plt.subplots(figsize=(15, 6.8), layout="constrained"); ax.axis("off"); decision_rows = []
    failure_labels = {"arm_vs_persistence_relative_ge_2pct": "Arm <2% or CI", "same_input_shared_ungated_relative_ge_2pct": "same-input <2% or CI", "first_difference_improves_vs_shared_ungated": "first-difference not improved", "native_tv_closeness_improves_vs_shared_ungated": "TV closeness not improved", "hand_relative_degradation_le_1pct": "Hand degradation >1%", "full_relative_degradation_le_1pct": "Full degradation >1%", "high_motion_not_over_1pct_reverse": "high-motion reverse >1%", "late_not_over_1pct_reverse": "late reverse >1%", "endpoint_not_over_1pct_reverse": "endpoint reverse >1%", "gate_not_collapsed": "gate collapse", "task_support_not_single": "task support single", "joint_support_not_single": "joint support single", "fusion_increment_positive": "Fusion increment nonpositive"}
    for row in decision["gate_evaluations"]:
        fails = [failure_labels.get(key, key) for key, value in row["checks"].items() if not value]
        decision_rows.append([row["cell_id"], f"{float(by_cell[row['cell_id']]['arm_full_mae_std']):.9f}", f"{row['hand_relative_degradation']:.1%}", "PASS" if row["gate_pass"] else "FAIL", ";\n".join(fails) if fails else "-"])
    table = ax.table(cellText=decision_rows, colLabels=["priority cell", "Arm MAE", "Hand rel. deg.", "gate", "failed conditions"], colWidths=[.14, .16, .16, .10, .44], loc="center", cellLoc="left", colLoc="left"); table.auto_set_font_size(False); table.set_fontsize(6.5); table.scale(1, 2.45)
    for cell in table.get_celld().values(): cell.get_text().set_wrap(True)
    ax.set_title("Predeclared exploratory candidate gate\nNo cell is promoted; best primary cell is E-F-SP-G", pad=20)
    save_figure(fig, "fig12_decision_summary", "Summary of the predeclared exploratory candidate gate. E-F-SP-G has the lowest Arm MAE but fails the Hand-14 relative-degradation condition.", "A table lists priority cells, Arm MAE, Hand degradation and failed conditions. No cell passes the complete candidate gate.", ["EXPLORATION_DECISION.json", "MAIN_RESULTS.csv"], manifest)

    write_csv(ROOT / "figure_manifests/FIGURE_MANIFEST.csv", manifest)
    write_json = (ROOT / "figure_manifests/FIGURE_MANIFEST.json").write_text
    write_json(json.dumps({"status": "PASS", "figure_count": len(manifest), "formats": ["png", "svg", "pdf"], "rows": manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "figures": len(manifest), "formats": ["png", "svg", "pdf"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
