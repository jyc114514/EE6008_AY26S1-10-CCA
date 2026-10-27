#!/usr/bin/env python3
"""Create six recovery figures from the locked CPU analysis artifacts."""

from __future__ import annotations
import os

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
RECOVERY = PROJECT / "research_archive/CORE_B_B_INFERENCE_RECOVERY_20260919T061409Z"
ANALYSIS = RECOVERY / "analysis"
FIGURES = RECOVERY / "figures"
DATA = RECOVERY / "figure_data"
MANIFESTS = RECOVERY / "figure_manifests"
HISTORY_ROOT = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/HISTORY_VIDEO_INFORMATION_MATRIX_V1_20260917T140755Z"
CELLS = ["FT-D-FRZ", "FT-S-FRZ", "FT-D-LORA4", "FT-S-LORA4", "FT-D-DENSE4", "FT-S-DENSE4"]
COLORS = {"FT-D-FRZ": "#6b7280", "FT-S-FRZ": "#9ca3af", "FT-D-LORA4": "#2563eb", "FT-S-LORA4": "#60a5fa", "FT-D-DENSE4": "#b91c1c", "FT-S-DENSE4": "#f87171"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle: return list(csv.DictReader(handle))


def save_pair(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=220)
    fig.savefig(FIGURES / f"{stem}.pdf")
    plt.close(fig)


def caption(stem: str, text: str) -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    (MANIFESTS / f"{stem}.caption.md").write_text(text.strip() + "\n", encoding="utf-8")


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True); MANIFESTS.mkdir(parents=True, exist_ok=True)
    result_rows = read_csv(ANALYSIS / "SIX_CELL_RESULTS.csv")
    by_model = {row["model"]: row for row in result_rows}
    paired = read_csv(ANALYSIS / "PAIRED_EFFECTS.csv")
    interactions = read_csv(ANALYSIS / "VIDEO_ADAPTATION_INTERACTIONS.csv")
    gates = read_csv(ANALYSIS / "CANDIDATE_GATE_DETAIL.csv")
    figures: list[dict[str, str]] = []

    # 1. Six-cell level/temporal trade-off.
    fig, ax = plt.subplots(figsize=(7.0, 5.2), layout="constrained")
    for cell in CELLS:
        row = by_model[cell]
        ax.scatter(float(row["hand_mae_std"]), 100 * float(row["native_fd_relative_vs_persistence"]), s=75, color=COLORS[cell], marker="o" if cell.startswith("FT-D") else "s", label=cell)
    ax.scatter(float(by_model["PERSISTENCE"]["hand_mae_std"]), 0, color="#111827", marker="x", s=85, label="Persistence")
    ax.axhline(1.0, color="#b91c1c", linestyle="--", linewidth=1, label="+1% FD margin")
    ax.set(xlabel="Hand standardized MAE (lower is better)", ylabel="Native Hand FD change vs Persistence (%)", title="Independent B: six-cell development Pareto view")
    ax.grid(alpha=0.25); ax.legend(fontsize=8, ncol=2)
    save_pair(fig, "FIG01_SIX_CELL_MAE_FD_PARETO")
    np.savetxt(DATA / "FIG01_SIX_CELL_MAE_FD_PARETO.csv", np.asarray([[c, by_model[c]["hand_mae_std"], by_model[c]["native_fd_relative_vs_persistence"]] for c in CELLS], dtype=object), fmt="%s", delimiter=",", header="cell,hand_mae_std,native_fd_relative_vs_persistence", comments="")
    caption("FIG01_SIX_CELL_MAE_FD_PARETO", "All six fixed-last development exports on the shared 614 canonical development episodes. Lower-left is favorable. The dashed line is the pre-locked +1% native FD operating margin. This is one B screen seed, not a final-test result.")
    figures.append({"stem": "FIG01_SIX_CELL_MAE_FD_PARETO", "description": "Six-cell Hand MAE versus native FD trade-off"})

    # 2. Registered paired adaptation contrasts.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), layout="constrained")
    for axis, metric, title in ((axes[0], "hand_mae_std", "Hand MAE"), (axes[1], "hand_fd_native", "Native Hand FD")):
        selected = [row for row in paired if row["metric"] == metric]
        labels = [row["contrast"] for row in selected]; x = np.arange(len(labels)); values = np.asarray([float(row["effect_control_minus_candidate"]) for row in selected]); lows = np.asarray([float(row["ci_low"]) for row in selected]); highs = np.asarray([float(row["ci_high"]) for row in selected])
        axis.errorbar(x, values, yerr=[values - lows, highs - values], fmt="o", color="#1d4ed8", capsize=4)
        axis.axhline(0, color="#555", linewidth=0.8); axis.set_xticks(x, labels, rotation=55, ha="right"); axis.set_title(title + "; 16-item Holm family"); axis.set_ylabel("control − candidate"); axis.grid(axis="y", alpha=0.25)
    save_pair(fig, "FIG02_REGISTERED_PAIRED_EFFECTS")
    (DATA / "FIG02_REGISTERED_PAIRED_EFFECTS.csv").write_text("\n".join(",".join(str(row.get(key, "")) for key in ("contrast", "metric", "effect_control_minus_candidate", "ci_low", "ci_high", "holm_p")) for row in paired) + "\n", encoding="utf-8")
    caption("FIG02_REGISTERED_PAIRED_EFFECTS", "Pre-registered eight contrasts crossed with the two co-primary metrics. Intervals are paired 10,000-episode development bootstrap intervals; Holm adjustment is reported in the source CSV. Positive effects favor the candidate.")
    figures.append({"stem": "FIG02_REGISTERED_PAIRED_EFFECTS", "description": "Registered paired effects with bootstrap intervals"})

    # 3. Dynamic/static adaptation interactions.
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8), layout="constrained")
    for axis, metric in zip(axes, ("hand_mae_std", "hand_fd_native")):
        selected = [row for row in interactions if row["metric"] == metric]; x = np.arange(len(selected)); v = np.asarray([float(row["effect_dynamic_gain_minus_static_gain"]) for row in selected]); lo = np.asarray([float(row["ci_low"]) for row in selected]); hi = np.asarray([float(row["ci_high"]) for row in selected])
        axis.errorbar(x, v, yerr=[v - lo, hi - v], fmt="o", color="#7c3aed", capsize=4); axis.axhline(0, color="#555", linewidth=0.8); axis.set_xticks(x, [row["adaptation"] for row in selected]); axis.set_title(metric + " interaction"); axis.set_ylabel("dynamic gain − static gain"); axis.grid(axis="y", alpha=0.25)
    save_pair(fig, "FIG03_VIDEO_ADAPTATION_INTERACTION")
    (DATA / "FIG03_VIDEO_ADAPTATION_INTERACTION.csv").write_text("\n".join(",".join(str(row.get(key, "")) for key in ("adaptation", "metric", "effect_dynamic_gain_minus_static_gain", "ci_low", "ci_high", "p_signflip")) for row in interactions) + "\n", encoding="utf-8")
    caption("FIG03_VIDEO_ADAPTATION_INTERACTION", "The registered descriptive interaction compares adaptation gain in dynamic versus static video. It is reported separately and is not interpreted as a causal dynamics claim from one development seed.")
    figures.append({"stem": "FIG03_VIDEO_ADAPTATION_INTERACTION", "description": "Dynamic/static adaptation interaction"})

    # 4. Horizon and motion guardrails.
    labels = ["early", "middle", "late", "endpoint", "low-motion", "mid-motion", "high-motion"]
    gate_names = ["horizon_early", "horizon_middle", "horizon_late", "horizon_endpoint", "motion_low", "motion_mid", "motion_high"]
    fig, ax = plt.subplots(figsize=(11, 5.0), layout="constrained")
    x = np.arange(len(labels)); width = 0.18
    candidate_cells = ["FT-D-LORA4", "FT-D-DENSE4", "FT-S-LORA4", "FT-S-DENSE4"]
    for i, cell in enumerate(candidate_cells):
        values = []
        for prefix in gate_names:
            matches = [row for row in gates if row["cell"] == cell and row["gate"].startswith(prefix)]
            values.append(100 * float(matches[0]["value"]) if matches else np.nan)
        ax.bar(x + (i - 1.5) * width, values, width, color=COLORS[cell], label=cell)
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1, label="+1% margin"); ax.axhline(0, color="#555", linewidth=0.8); ax.set_xticks(x, labels, rotation=30, ha="right"); ax.set_ylabel("Relative change vs Persistence (%)"); ax.set_title("Guardrails: horizon and motion strata"); ax.grid(axis="y", alpha=0.25); ax.legend(fontsize=8, ncol=2)
    save_pair(fig, "FIG04_HORIZON_MOTION_GUARDRAILS")
    (DATA / "FIG04_HORIZON_MOTION_GUARDRAILS.csv").write_text("\n".join(",".join([cell, label, str(next((row["value"] for row in gates if row["cell"] == cell and row["gate"].startswith(prefix)), "nan"))]) for cell in candidate_cells for label, prefix in zip(labels, gate_names)) + "\n", encoding="utf-8")
    caption("FIG04_HORIZON_MOTION_GUARDRAILS", "Relative Hand standardized MAE changes by horizon and the locked low/mid/high motion strata. Positive values are adverse; the dashed line is the +1% operating margin. Low-motion N is 172 under the inherited definition.")
    figures.append({"stem": "FIG04_HORIZON_MOTION_GUARDRAILS", "description": "Horizon and motion guardrails"})

    # 5. Fixed representative command trajectories.
    with np.load(HISTORY_ROOT / "inputs/HISTORY_VIDEO_INPUTS.npz", allow_pickle=False) as source_data:
        target = source_data["targets_native"][source_data["split_code"] == 1, :, 0].astype(np.float32)
        persistence = source_data["persistence_native"][source_data["split_code"] == 1, :, 0].astype(np.float32)
        ids = source_data["episode_index"][source_data["split_code"] == 1].astype(np.int64)
    with np.load(RECOVERY / "predictions/FT-D-DENSE4/shard_000.npz", allow_pickle=False) as pred:
        candidate = pred["predictions_hand_native"][:, :, 0].astype(np.float32)
    positions = [0, 204, 409, 613]; steps = np.arange(1, 31)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True, layout="constrained")
    trajectory_rows: list[list[object]] = []
    for axis, position in zip(axes.flat, positions):
        axis.plot(steps, target[position], color="#111827", linewidth=1.8, label="GT"); axis.plot(steps, persistence[position], color="#6b7280", linestyle="--", label="Persistence"); axis.plot(steps, candidate[position], color="#b91c1c", linewidth=1.4, label="FT-D-DENSE4"); axis.set_title(f"dev row {position}; episode {int(ids[position])}"); axis.grid(alpha=0.25); axis.set_xlabel("future step"); axis.set_ylabel("command joint 0")
        for step, truth, base, value in zip(steps, target[position], persistence[position], candidate[position]): trajectory_rows.append([position, int(ids[position]), int(step), float(truth), float(base), float(value)])
    axes[0, 0].legend(fontsize=8); save_pair(fig, "FIG05_FIXED_REPRESENTATIVE_TRAJECTORIES")
    with (DATA / "FIG05_FIXED_REPRESENTATIVE_TRAJECTORIES.csv").open("w", encoding="utf-8") as handle:
        handle.write("dev_row,episode_index,future_step,ground_truth,persistence,FT-D-DENSE4\n"); handle.write("\n".join(",".join(map(str, row)) for row in trajectory_rows) + "\n")
    caption("FIG05_FIXED_REPRESENTATIVE_TRAJECTORIES", "Four deterministic development episodes and Hand command joint 0. Positions [0, 204, 409, 613] were fixed before inspecting recovery outcomes. These are command trajectories; physical units and FK interpretation remain unverified.")
    figures.append({"stem": "FIG05_FIXED_REPRESENTATIVE_TRAJECTORIES", "description": "Fixed representative command trajectories"})

    # 6. Resource and recovery QA cost.
    resource = read_csv(ANALYSIS / "RESOURCE_COSTS.csv")
    failed_count = sum(1 for row in resource if row["kind"] == "replay" and row["status"] == "oom_exhausted")
    groups = [("prior B training", sum(float(row["wall_gpu_hours"]) for row in resource if row["kind"] == "prior_training")), ("six dev inference", sum(float(row["wall_gpu_hours"]) for row in resource if row["kind"] == "inference" and row["status"] == "complete")), ("successful replay", sum(float(row["wall_gpu_hours"]) for row in resource if row["kind"] == "replay" and row["status"] == "complete")), (f"failed replay (n={failed_count}; time n/r)", np.nan)]
    plot_values = [0.0 if not np.isfinite(value) else value for _, value in groups]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained"); ax.bar(np.arange(len(groups)), plot_values, color=["#374151", "#2563eb", "#059669", "#dc2626"]); ax.set_xticks(np.arange(len(groups)), [name for name, _ in groups], rotation=20, ha="right"); ax.set_ylabel("GPU-hours"); ax.set_title("Core B B branch: prior training and recovery compute"); ax.grid(axis="y", alpha=0.25)
    for i, (_, value) in enumerate(groups): ax.text(i, 0.05 if not np.isfinite(value) else value, "duration not recorded" if not np.isfinite(value) else f"{value:.3f}", ha="center", va="bottom", fontsize=8, rotation=90 if not np.isfinite(value) else 0)
    save_pair(fig, "FIG06_RESOURCE_AND_RECOVERY_COST")
    (DATA / "FIG06_RESOURCE_AND_RECOVERY_COST.csv").write_text("group,gpu_hours\n" + "\n".join(f"{name},{value}" for name, value in groups) + "\n", encoding="utf-8")
    caption("FIG06_RESOURCE_AND_RECOVERY_COST", "GPU-hour accounting separates the already completed 10-epoch B training from recovery replay and fixed development inference. Four early replay OOM attempts are retained; their watcher records did not contain a reliable completion timestamp, so their duration is explicitly marked not recorded rather than treated as zero. No training was repeated.")
    figures.append({"stem": "FIG06_RESOURCE_AND_RECOVERY_COST", "description": "Resource and recovery cost"})

    manifest = {"status": "PASS", "figure_count": len(figures), "figures": figures, "png_pdf": True, "final_test_access": 0}
    (MANIFESTS / "FIGURE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (MANIFESTS / "FIGURE_MANIFEST.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stem", "description"]); writer.writeheader(); writer.writerows(figures)
    print(json.dumps({"status": "PASS", "figure_count": len(figures), "output": str(FIGURES), "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
