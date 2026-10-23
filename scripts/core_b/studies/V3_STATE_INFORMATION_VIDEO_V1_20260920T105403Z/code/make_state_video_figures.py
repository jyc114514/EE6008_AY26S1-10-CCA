#!/usr/bin/env python3
"""Create six compact development figures from the harvested CSV/NPZ evidence."""

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
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
V3 = REPO / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
EXP = REPO / "reports/core_b/V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z"
CELLS = tuple(f"{state}-{video}" for state in ("FULL", "OBS", "ANCHOR") for video in ("N", "D", "S"))
SEEDS = (20260953, 20260954, 20260955)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(EXP / f"figures/{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(EXP / f"figures/{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_caption(stem: str, text: str) -> None:
    (EXP / f"figure_manifests/{stem}.caption.md").write_text(text + "\n", encoding="utf-8")


def main() -> int:
    (EXP / "figures").mkdir(parents=True, exist_ok=True)
    (EXP / "figure_data").mkdir(parents=True, exist_ok=True)
    (EXP / "figure_manifests").mkdir(parents=True, exist_ok=True)

    # FIG01: exact information mask contract, not a claim about physical necessity.
    mask_rows = []
    for state in ("FULL", "OBS", "ANCHOR"):
        for index in range(71):
            kept = 1 if state == "FULL" or (state == "OBS" and index < 43) else 0
            mask_rows.append({"state_level": state, "feature_index": index, "kept_history_value": kept, "feature_block": "observation" if index < 43 else "action"})
    with (EXP / "figure_data/FIG01_STATE_MASK.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mask_rows[0])); writer.writeheader(); writer.writerows(mask_rows)
    matrix = np.asarray([[row["kept_history_value"] for row in mask_rows if row["state_level"] == state] for state in ("FULL", "OBS", "ANCHOR")])
    fig, ax = plt.subplots(figsize=(10, 2.6))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(3), ["FULL", "OBS", "ANCHOR"]); ax.set_xlabel("71-d history feature index (0–42 observation, 43–70 action)"); ax.set_title("State-information masks applied after normalization")
    ax.axvline(42.5, color="black", linewidth=0.8); fig.colorbar(im, ax=ax, label="history value retained")
    save_figure(fig, "FIG01_STATE_INFORMATION_MASKS")
    write_caption("FIG01_STATE_INFORMATION_MASKS", "The three declared state-information conditions. OBS removes only the 28 action/history columns; ANCHOR removes the history branch but retains the independent last-command anchor token. This is an input contract, not evidence that removed information is physically unnecessary.")

    # FIG02: primary aggregate MAE by horizon.
    aggregate = rows(EXP / "analysis/AGGREGATE_RESULTS.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, horizon in zip(axes, (30, 60)):
        subset = [r for r in aggregate if int(r["horizon_steps"]) == horizon]
        x = np.arange(len(CELLS)); width = 0.25
        for offset, group, label, color in ((-width, "arm_mae_std", "Arm", "#3264a8"), (0, "hand_mae_std", "Hand", "#d7792b"), (width, "full_mae_std", "Full", "#4b8b5a")):
            ax.bar(x + offset, [float(r[group + "_mean"]) for r in subset], width, label=label, color=color)
        ax.set_title(f"{horizon/30:.0f}s horizon"); ax.set_xticks(x, [r["cell"] for r in subset], rotation=45, ha="right"); ax.set_ylabel("standardized MAE"); ax.grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    save_figure(fig, "FIG02_ARM_HAND_FULL_BY_HORIZON")
    write_caption("FIG02_ARM_HAND_FULL_BY_HORIZON", "Episode-equal primary development MAE for all nine state/video cells. Bars show the three-seed mean; seed variation is retained in the CSV tables rather than hidden by selecting a best seed.")

    # FIG03: D versus N and D versus S video effects.
    effects = rows(EXP / "analysis/PRIMARY_VIDEO_EFFECTS.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, horizon in zip(axes, (30, 60)):
        subset = [r for r in effects if int(r["horizon_steps"]) == horizon]
        labels = [f"{r['state_level']}\n{r['group'].split('_')[0]}" for r in subset if r["control"].endswith("-N")]
        x = np.arange(len(labels)); width = 0.36
        for offset, suffix, color, label in ((-width/2, "_vs_N", "#3478b9", "D vs N"), (width/2, "_vs_S", "#a44a9d", "D vs S")):
            selected = [r for r in subset if r["control"].endswith("-N" if suffix == "_vs_N" else "-S")]
            ax.errorbar(x + offset, [float(r["effect"]) for r in selected], yerr=[[float(r["effect"]) - float(r["ci_low"]) for r in selected], [float(r["ci_high"]) - float(r["effect"]) for r in selected]], fmt="o", capsize=3, color=color, label=label)
        ax.axhline(0, color="black", linewidth=.8); ax.set_title(f"{horizon/30:.0f}s horizon"); ax.set_xticks(x, labels); ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("positive effect = video candidate lower MAE"); axes[0].legend(frameon=False, fontsize=8)
    save_figure(fig, "FIG03_VIDEO_GAIN_BY_STATE")
    write_caption("FIG03_VIDEO_GAIN_BY_STATE", "Primary video contrasts. D vs N measures video relative to normalized no-video input; D vs S measures dynamic temporal tokens relative to static repeated-last-frame tokens. Confidence intervals are crossed seed-by-episode bootstrap intervals.")

    # FIG04: information interactions.
    interactions = rows(EXP / "analysis/INFORMATION_INTERACTIONS.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    labels = [f"{r['horizon_steps']}\n{r['group'].split('_')[0]}\n{r['high_state']}−{r['low_state']}" for r in interactions]
    x = np.arange(len(labels)); effect = [float(r["effect"]) for r in interactions]
    ax.errorbar(x, effect, yerr=[[e - float(r["ci_low"]) for e, r in zip(effect, interactions)], [float(r["ci_high"]) - e for e, r in zip(effect, interactions)]], fmt="o", capsize=3, color="#7a3e9d")
    ax.axhline(0, color="black", linewidth=.8); ax.set_xticks(x, labels); ax.set_ylabel("change in video gain (positive = larger after state removal)"); ax.set_title("Information-level interaction contrasts"); ax.grid(axis="y", alpha=.25)
    save_figure(fig, "FIG04_INFORMATION_INTERACTIONS")
    write_caption("FIG04_INFORMATION_INTERACTIONS", "Interaction contrasts compare video gain after reducing state information. A positive value is consistent with a larger conditional video gain, but does not establish that command history is physically redundant in the complete-state task.")

    # FIG05: per-seed full skill heatmaps.
    seed_rows = rows(EXP / "analysis/RESULTS_BY_SEED.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, horizon in zip(axes, (30, 60)):
        values = np.asarray([[float(next(r["full_skill_vs_persistence"] for r in seed_rows if int(r["seed"]) == seed and int(r["horizon_steps"]) == horizon and r["cell"] == cell)) for cell in CELLS] for seed in SEEDS])
        im = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=max(.25, float(values.max())))
        ax.set_title(f"{horizon/30:.0f}s full skill vs Persistence"); ax.set_xticks(range(len(CELLS)), CELLS, rotation=45, ha="right"); ax.set_yticks(range(3), [str(s) for s in SEEDS])
        for i in range(3):
            for j in range(len(CELLS)): ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="1 − model MAE / Persistence MAE")
    save_figure(fig, "FIG05_PER_SEED_HORIZON_SKILL")
    write_caption("FIG05_PER_SEED_HORIZON_SKILL", "Full standardized-MAE skill relative to the Persistence baseline for every fresh seed and every cell. This visualization keeps seed variability visible and does not rank or remove cells based on the first seed.")

    # FIG06: fixed primary episode trajectories, direct native output.
    with np.load(V3 / "data/V3_MAIN_INPUTS.npz", allow_pickle=False) as bundle:
        data = {key: bundle[key].copy() for key in bundle.files}
    dev = np.flatnonzero(data["split_code"] == 1)
    row = int(dev[0]); target = data["targets_native"][row]
    methods = ["PERSISTENCE", "FULL-N", "FULL-D", "OBS-D", "ANCHOR-D"]
    colors = {"PERSISTENCE": "black", "FULL-N": "#777777", "FULL-D": "#3478b9", "OBS-D": "#d7792b", "ANCHOR-D": "#7a3e9d"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), sharex='col')
    for col, horizon in enumerate((30, 60)):
        t = np.arange(1, horizon + 1) / 30.0
        for row_ax, joint, title in ((axes[0, col], 0, "Arm joint 0"), (axes[1, col], 14, "Hand joint 0")):
            row_ax.plot(t, target[:horizon, joint], color="green", linewidth=2, label="GT")
            for method in methods:
                if method == "PERSISTENCE":
                    value = np.repeat(data["anchor_command_native"][row, joint], horizon)
                else:
                    run_id = f"SIV-{method}-H{horizon}-S20260953"
                    with np.load(EXP / f"predictions/{run_id}/primary.npz", allow_pickle=False) as bundle: value = bundle["predictions_native"][0, :, joint]
                row_ax.plot(t, value, color=colors[method], linewidth=1, alpha=.9, label=method)
            row_ax.set_title(f"{horizon/30:.0f}s: {title}"); row_ax.grid(alpha=.25)
    axes[1, 0].set_xlabel("future seconds"); axes[1, 1].set_xlabel("future seconds"); axes[0, 0].set_ylabel("native command"); axes[1, 0].set_ylabel("native command"); axes[0, 0].legend(frameon=False, fontsize=7, ncol=3)
    save_figure(fig, "FIG06_FIXED_EPISODE_TRAJECTORIES")
    write_caption("FIG06_FIXED_EPISODE_TRAJECTORIES", "One predeclared primary development window, shown for two joints and both horizons. The plot is qualitative trajectory context; all claims use the episode-equal numeric tables and confidence intervals.")

    manifest = {"status": "PASS", "figures": ["FIG01_STATE_INFORMATION_MASKS", "FIG02_ARM_HAND_FULL_BY_HORIZON", "FIG03_VIDEO_GAIN_BY_STATE", "FIG04_INFORMATION_INTERACTIONS", "FIG05_PER_SEED_HORIZON_SKILL", "FIG06_FIXED_EPISODE_TRAJECTORIES"], "png_pdf_each": True, "final_test_access": 0}
    (EXP / "figure_manifests/FIGURE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
