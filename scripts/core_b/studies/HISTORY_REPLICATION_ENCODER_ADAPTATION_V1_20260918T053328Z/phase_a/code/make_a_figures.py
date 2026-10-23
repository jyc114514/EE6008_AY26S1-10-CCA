#!/usr/bin/env python3
"""Create the locked Phase-A development figures and machine-readable data."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
FIGURES = ROOT / "figures"
FIGURE_DATA = ROOT / "figure_data"
MANIFESTS = ROOT / "figure_manifests"
SEEDS = [20260940, 20260941, 20260942]
CELLS = ["HI-S-LAST", "HI-S-HIST", "HI-F-LAST", "HI-F-HIST"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_pair(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def caption(stem: str, text: str) -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    (MANIFESTS / f"{stem}.caption.md").write_text(text.strip() + "\n", encoding="utf-8")


def main() -> int:
    FIGURE_DATA.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    per_seed = read_csv(ANALYSIS / "A_PER_SEED_METRICS.csv")
    rows = [row for row in per_seed if row["model"] in CELLS]
    by_key = {(int(row["seed"]), row["model"]): row for row in rows}

    # FIG01: all four cells across all fresh seeds.
    x = np.arange(len(CELLS))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    colors = {20260940: "#2563eb", 20260941: "#dc2626", 20260942: "#059669"}
    for seed in SEEDS:
        axes[0].plot(x, [float(by_key[(seed, cell)]["hand_mae_std"]) for cell in CELLS], "o-", color=colors[seed], label=str(seed))
        axes[1].plot(x, [100 * (float(by_key[(seed, cell)]["hand_fd_native"]) / 0.018316442146897316 - 1) for cell in CELLS], "o-", color=colors[seed], label=str(seed))
    axes[0].set_ylabel("Hand standardized MAE (lower is better)")
    axes[1].set_ylabel("Hand native FD change vs Persistence (%)")
    for axis in axes:
        axis.set_xticks(x, CELLS, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.axhline(0, color="#888", linewidth=0.7)
    axes[1].axhline(1.0, color="#b91c1c", linestyle="--", linewidth=1, label="+1% gate")
    axes[0].legend(title="fresh seed", fontsize=8)
    fig.suptitle("Phase A: four matched cells across fresh seeds")
    save_pair(fig, "FIG01_A_FOUR_CELL_SEED")
    np.savetxt(FIGURE_DATA / "FIG01_A_FOUR_CELL_SEED.csv", np.asarray([[seed, cell, float(by_key[(seed, cell)]["hand_mae_std"]), float(by_key[(seed, cell)]["hand_fd_native"])] for seed in SEEDS for cell in CELLS], dtype=object), fmt="%s", delimiter=",", header="seed,cell,hand_mae_std,hand_fd_native", comments="")
    caption("FIG01_A_FOUR_CELL_SEED", "Fresh-seed Phase-A results for the four pre-registered History×Video cells. Each point is a fixed-last checkpoint evaluated on the same 614 development episodes. FD is shown relative to the fixed Persistence native FD; the dashed line is the pre-locked +1% operating margin.")

    # FIG02: Hand level versus temporal fidelity Pareto view.
    fig, ax = plt.subplots(figsize=(6.6, 5.0), constrained_layout=True)
    for seed in SEEDS:
        for cell in CELLS:
            row = by_key[(seed, cell)]
            ax.scatter(float(row["hand_mae_std"]), 100 * (float(row["hand_fd_native"]) / 0.018316442146897316 - 1), color=colors[seed], marker={"HI-S-LAST": "o", "HI-S-HIST": "s", "HI-F-LAST": "^", "HI-F-HIST": "D"}[cell], s=55)
    ax.axhline(1.0, color="#b91c1c", linestyle="--", linewidth=1)
    ax.set_xlabel("Hand standardized MAE")
    ax.set_ylabel("Native FD change vs Persistence (%)")
    ax.grid(alpha=0.25)
    handles = [plt.Line2D([], [], color=colors[seed], marker="o", linestyle="", label=str(seed)) for seed in SEEDS]
    handles += [plt.Line2D([], [], color="#333", marker=marker, linestyle="", label=cell) for cell, marker in {"HI-S-LAST": "o", "HI-S-HIST": "s", "HI-F-LAST": "^", "HI-F-HIST": "D"}.items()]
    ax.legend(handles=handles, fontsize=8, ncol=2, title="color=seed; marker=cell")
    ax.set_title("Phase A Hand MAE–temporal-fidelity trade-off")
    save_pair(fig, "FIG02_A_HAND_FD_PARETO")
    np.savetxt(FIGURE_DATA / "FIG02_A_HAND_FD_PARETO.csv", np.asarray([[seed, cell, float(by_key[(seed, cell)]["hand_mae_std"]), float(by_key[(seed, cell)]["hand_fd_native"])] for seed in SEEDS for cell in CELLS], dtype=object), fmt="%s", delimiter=",", header="seed,cell,hand_mae_std,hand_fd_native", comments="")
    caption("FIG02_A_HAND_FD_PARETO", "Each point is one fresh-seed/cell result. Lower-left is favorable. The dashed line marks the +1% native FD margin relative to Persistence; this is a development diagnostic, not a final-test result.")

    # FIG03: registered contrasts and separately registered interaction.
    contrasts = read_csv(ANALYSIS / "A_FIXED_CONTRASTS.csv")
    interaction = json.loads((ANALYSIS / "A_HISTORY_VIDEO_INTERACTION.json").read_text(encoding="utf-8"))["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for axis, metric, title in ((axes[0, 0], "hand_mae_std", "Hand MAE"), (axes[0, 1], "hand_fd_native", "Native FD")):
        selected = [row for row in contrasts if row["metric"] == metric]
        labels = [row["contrast"] for row in selected]
        effects = [float(row["effect_control_minus_candidate"]) for row in selected]
        lows = [float(row["ci_low"]) for row in selected]
        highs = [float(row["ci_high"]) for row in selected]
        axis.errorbar(np.arange(len(labels)), effects, yerr=[np.asarray(effects) - np.asarray(lows), np.asarray(highs) - np.asarray(effects)], fmt="o", color="#1d4ed8", capsize=4)
        axis.axhline(0, color="#666", linewidth=0.8)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
        axis.set_title(title + " fixed contrasts")
        axis.set_ylabel("control − candidate")
        axis.grid(axis="y", alpha=0.25)
    labels = [row["metric"] for row in interaction]
    effects = [float(row["effect_control_minus_candidate"]) for row in interaction]
    lows = [float(row["ci_low"]) for row in interaction]
    highs = [float(row["ci_high"]) for row in interaction]
    axes[1, 0].errorbar(np.arange(len(labels)), effects, yerr=[np.asarray(effects) - np.asarray(lows), np.asarray(highs) - np.asarray(effects)], fmt="o", color="#7c3aed", capsize=4)
    axes[1, 0].axhline(0, color="#666", linewidth=0.8)
    axes[1, 0].set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    axes[1, 0].set_title("History×Video interaction")
    axes[1, 0].set_ylabel("(F-LAST−F-HIST) − (S-LAST−S-HIST)")
    axes[1, 0].grid(axis="y", alpha=0.25)
    decision = json.loads((ANALYSIS / "A_CANDIDATE_DECISION.json").read_text(encoding="utf-8"))
    fd = decision["native_fd_vs_persistence"]["aggregate"]
    low_motion = next(row for row in read_csv(ANALYSIS / "A_GUARDRAILS.csv") if row["guardrail"] == "motion_low_hand_mae_vs_persistence")
    axes[1, 1].bar(["Native FD\nrelative", "Low-motion\nMAE relative"], [100 * float(fd["relative_change_candidate_vs_control"]), 100 * float(low_motion["relative_change_candidate_vs_control"])], color=["#f97316", "#dc2626"])
    axes[1, 1].axhline(1.0, color="#111", linestyle="--", linewidth=1)
    axes[1, 1].set_ylabel("Relative change (%)")
    axes[1, 1].set_title("Gate-limiting guardrails")
    axes[1, 1].grid(axis="y", alpha=0.25)
    fig.suptitle("Phase A matched effects and gate diagnostics")
    save_pair(fig, "FIG03_A_MATCHED_EFFECTS_INTERACTION")
    (FIGURE_DATA / "FIG03_A_MATCHED_EFFECTS_INTERACTION.json").write_text(json.dumps({"contrasts": contrasts, "interaction": interaction, "gate_limited": {"native_fd": fd, "low_motion": low_motion}}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    caption("FIG03_A_MATCHED_EFFECTS_INTERACTION", "The upper panels show the four registered matched contrasts for the two co-primary metrics, with crossed seed×episode 95% intervals. The lower-left panel reports the pre-defined interaction separately. The lower-right panel shows the two observed gate-limiting relative changes; positive FD/low-motion values are adverse.")

    # FIG04: deterministic representative command trajectories.  The rule is
    # fixed before plotting: seed 20260940, dev row positions [0,204,409,613],
    # Hand joint 0, no outcome-based selection.
    selected_positions = [0, 204, 409, 613]
    seed = SEEDS[0]
    run_dir = ROOT / f"runs/HI-F-HIST/seed{seed}/formal"
    pred = np.load(run_dir / "predictions_last.npz", allow_pickle=False)
    target = pred["targets_native"][:, :, 0]
    persistence = pred["persistence_native"][:, :, 0]
    candidate = pred["predictions_hand_native"][:, :, 0]
    ids = pred["episode_index"]
    steps = np.arange(1, 31)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), sharex=True, sharey=True, constrained_layout=True)
    traj_rows: list[list[object]] = []
    for axis, position in zip(axes.flat, selected_positions):
        axis.plot(steps, target[position], color="#111827", linewidth=1.8, label="GT")
        axis.plot(steps, persistence[position], color="#6b7280", linestyle="--", label="Persistence")
        axis.plot(steps, candidate[position], color="#2563eb", linewidth=1.4, label="HI-F-HIST")
        axis.set_title(f"dev row {position}; episode {int(ids[position])}")
        axis.grid(alpha=0.25)
        axis.set_xlabel("future step")
        axis.set_ylabel("command joint 0")
        for step, truth, base, value in zip(steps, target[position], persistence[position], candidate[position]):
            traj_rows.append([seed, position, int(ids[position]), int(step), float(truth), float(base), float(value)])
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Phase A fixed representative trajectories (Hand joint 0)")
    save_pair(fig, "FIG04_A_REPRESENTATIVE_TRAJECTORIES")
    np.savetxt(FIGURE_DATA / "FIG04_A_REPRESENTATIVE_TRAJECTORIES.csv", np.asarray(traj_rows, dtype=object), fmt="%s", delimiter=",", header="seed,dev_row,episode_index,future_step,ground_truth,persistence,hi_f_hist", comments="")
    caption("FIG04_A_REPRESENTATIVE_TRAJECTORIES", "Four deterministic development episodes are shown for Hand command joint 0. The selection rule is fixed-seed 20260940 and dev row positions [0, 204, 409, 613], chosen without looking at outcomes. These are command trajectories; physical units and FK interpretation remain unverified.")

    manifest = {
        "status": "PASS",
        "figures": [
            {"stem": "FIG01_A_FOUR_CELL_SEED", "png": "figures/FIG01_A_FOUR_CELL_SEED.png", "pdf": "figures/FIG01_A_FOUR_CELL_SEED.pdf"},
            {"stem": "FIG02_A_HAND_FD_PARETO", "png": "figures/FIG02_A_HAND_FD_PARETO.png", "pdf": "figures/FIG02_A_HAND_FD_PARETO.pdf"},
            {"stem": "FIG03_A_MATCHED_EFFECTS_INTERACTION", "png": "figures/FIG03_A_MATCHED_EFFECTS_INTERACTION.png", "pdf": "figures/FIG03_A_MATCHED_EFFECTS_INTERACTION.pdf"},
            {"stem": "FIG04_A_REPRESENTATIVE_TRAJECTORIES", "png": "figures/FIG04_A_REPRESENTATIVE_TRAJECTORIES.png", "pdf": "figures/FIG04_A_REPRESENTATIVE_TRAJECTORIES.pdf"},
        ],
        "visual_qa_required": True,
        "final_test_access": 0,
    }
    (MANIFESTS / "FIGURE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (MANIFESTS / "FIGURE_MANIFEST.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stem", "png", "pdf"])
        writer.writeheader()
        writer.writerows(manifest["figures"])
    print(json.dumps({"status": "PASS", "figure_count": len(manifest["figures"]), "output": str(FIGURES)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
