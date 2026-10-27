#!/usr/bin/env python3
"""Make the 12 locked Frozen attribution figure bundles from CPU summaries."""

from __future__ import annotations
import os

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[0].parent
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V2 = PROJECT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in values:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def save(name: str, fig: plt.Figure, data: list[dict[str, object]], caption: str, alt: str) -> None:
    write_csv(ROOT / "figure_data" / f"{name}.csv", data)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(ROOT / "figures" / extension / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    (ROOT / "figure_manifests" / f"{name}.caption.md").write_text(f"# {name}\n\n{caption}\n\n**中文 alt text：** {alt}\n", encoding="utf-8")


def bundle_path(cell: str) -> Path:
    if cell.startswith("VF-"):
        return ROOT / "predictions/B3-V-D-F" / cell / "predictions.npz"
    if cell in {"FF-R1", "FF-R2", "FF-R3"}:
        return ROOT / "predictions/B3-FUS-D-F_FEATURE_RETRY_20260912T070723Z" / cell / "predictions.npz"
    return ROOT / "predictions/B3-FUS-D-F" / cell / "predictions.npz"


def load_cell(cell: str) -> dict[str, np.ndarray]:
    with np.load(bundle_path(cell), allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def main() -> int:
    main_rows = rows(ROOT / "MAIN_RESULTS.csv")
    effect_rows = rows(ROOT / "PAIRED_EFFECTS.csv")
    time_rows = rows(ROOT / "PER_TIME.csv")
    task_rows = rows(ROOT / "PER_TASK.csv")
    manifest_rows = rows(V2 / "V2_TRAJECTORY_SELECTION_MANIFEST.csv")
    figures: list[dict[str, object]] = []

    # 1. Fusion effect forest.
    fusion_cells = ["FF-P1", "FF-P2", "FF-P3", "FF-P4", "FF-R1", "FF-R2", "FF-R3"]
    selected = [r for r in effect_rows if r["cell"] in fusion_cells and r["metric"] == "arm_full"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(selected))
    values = [float(r["damage"]) for r in selected]
    lo = [values[i] - float(selected[i]["damage_ci_low"]) for i in range(len(selected))]
    hi = [float(selected[i]["damage_ci_high"]) - values[i] for i in range(len(selected))]
    ax.errorbar(values, y, xerr=[lo, hi], fmt="o", color="#0072B2", capsize=3)
    ax.axvline(0, color="black", linewidth=1); ax.set_yticks(y, [r["cell"] for r in selected]); ax.set_xlabel("Arm-14 MAE damage vs matched (positive=worse)"); ax.set_title("Fusion Frozen video controls")
    data = [{"cell": r["cell"], "damage": r["damage"], "ci_low": r["damage_ci_low"], "ci_high": r["damage_ci_high"]} for r in selected]
    save("fig01_fusion_effects_forest", fig, data, "Paired episode-bootstrap damage for each Fusion control versus chronological matched.", "Fusion control damage with shared episode bootstrap intervals."); figures.append({"figure": "fig01_fusion_effects_forest", "status": "PASS"})

    # 2. Temporal effect forest.
    selected = [r for r in effect_rows if r["cell"] in ["VF-P1", "VF-P2", "VF-P3", "VF-P4", "FF-P1", "FF-P2", "FF-P3", "FF-P4"] and r["metric"] == "arm_full"]
    fig, ax = plt.subplots(figsize=(8, 5)); y = np.arange(len(selected)); values = [float(r["damage"]) for r in selected]; lo = [values[i] - float(selected[i]["damage_ci_low"]) for i in range(len(selected))]; hi = [float(selected[i]["damage_ci_high"]) - values[i] for i in range(len(selected))]
    ax.errorbar(values, y, xerr=[lo, hi], fmt="o", color="#D55E00", capsize=3); ax.axvline(0, color="black", linewidth=1); ax.set_yticks(y, [r["cell"] for r in selected]); ax.set_xlabel("Arm-14 MAE damage vs chronological matched"); ax.set_title("Frozen temporal/pixel controls")
    save("fig02_temporal_effects", fig, [{"cell": r["cell"], "damage": r["damage"], "ci_low": r["damage_ci_low"], "ci_high": r["damage_ci_high"]} for r in selected], "Pixel controls are compared with the same recipient episodes and locked matched baseline.", "Temporal control damage forest plot."); figures.append({"figure": "fig02_temporal_effects", "status": "PASS"})

    # 3. Horizon profiles.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    data = []
    for ax, prefix, title in zip(axes, ("VF", "FF"), ("Video-only", "Fusion")):
        for cell in [f"{prefix}-P0", f"{prefix}-P1", f"{prefix}-P2", f"{prefix}-P3", f"{prefix}-P4"]:
            sub = [r for r in time_rows if r["cell"] == cell]
            x = [int(r["time_index"]) + 1 for r in sub]; yv = [float(r["arm_mae"]) for r in sub]; ax.plot(x, yv, label=cell); data.extend({"cell": cell, "time_index": xx, "arm_mae": yy} for xx, yy in zip(x, yv))
        ax.set_title(title); ax.set_xlabel("Horizon step"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Arm-14 MAE"); axes[1].legend(fontsize=7); fig.suptitle("Frozen error profiles over horizon")
    save("fig03_horizon_profiles", fig, data, "Step-resolved Arm-14 error for all chronological and pixel-control conditions.", "Two-panel horizon profiles for Video-only and Fusion Frozen models."); figures.append({"figure": "fig03_horizon_profiles", "status": "PASS"})

    # 4. Persistence relative skill.
    selected = [r for r in main_rows if r["cell"] in ["PERSISTENCE", "B-STATE", "VF-P0", "FF-P0", "FF-R1", "FF-R2", "FF-R3"]]
    persistence = float(next(r["arm_full_mae"] for r in selected if r["cell"] == "PERSISTENCE")); data = []
    labels=[]; vals=[]
    for r in selected:
        if r["cell"] == "PERSISTENCE": continue
        labels.append(r["cell"]); vals.append(persistence - float(r["arm_full_mae"])); data.append({"cell": r["cell"], "persistence_minus_cell_arm_mae": persistence - float(r["arm_full_mae"])})
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(labels, vals, color="#009E73"); ax.axhline(0, color="black", linewidth=1); ax.set_ylabel("Persistence MAE - cell MAE"); ax.set_title("Persistence-relative Frozen skill (positive is better)"); ax.tick_params(axis="x", rotation=35)
    save("fig04_persistence_skill", fig, data, "Descriptive persistence-relative skill; this is not a seed-robustness claim.", "Bar chart of improvement over persistence."); figures.append({"figure": "fig04_persistence_skill", "status": "PASS"})

    # 5. First difference and TV.
    cells = ["PERSISTENCE", "B-STATE", "VF-P0", "FF-P0", "FF-R1", "FF-R2", "FF-R3"]
    selected = [r for r in main_rows if r["cell"] in cells]; x = np.arange(len(selected)); width=.38
    fig, ax = plt.subplots(figsize=(9, 4.5)); ax.bar(x-width/2, [float(r["first_difference_mae"]) for r in selected], width, label="first difference"); ax.bar(x+width/2, [float(r["native_tv_ratio"]) for r in selected], width, label="native TV ratio"); ax.set_xticks(x, [r["cell"] for r in selected], rotation=35); ax.legend(); ax.set_title("Trajectory fidelity diagnostics"); ax.grid(axis="y", alpha=.2)
    save("fig05_first_difference_tv", fig, [{"cell": r["cell"], "first_difference_mae": r["first_difference_mae"], "native_tv_ratio": r["native_tv_ratio"]} for r in selected], "First-difference and native TV diagnostics are reported separately from primary MAE.", "Grouped bars for first-difference error and native TV ratio."); figures.append({"figure": "fig05_first_difference_tv", "status": "PASS"})

    # 6/7 representative command trajectories.
    rep_id = int(next(r["episode_id"] for r in manifest_rows if r["motion_stratum"] == "high"))
    target_bundle = load_cell("FF-P0"); pos = int(np.where(target_bundle["episode_index"] == rep_id)[0][0]); target = target_bundle["targets_native"][pos, :, 14:28].mean(axis=1); persistence = np.repeat(target_bundle["targets_native"][pos, 0, 14:28].mean(), 30)
    fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(target, color="black", label="GT"); ax.plot(persistence, "--", color="#555555", label="persistence")
    for cell, color in (("FF-P0", "#009E73"), ("FF-R1", "#D55E00")):
        b=load_cell(cell); ax.plot(b["predictions_native"][int(np.where(b["episode_index"]==rep_id)[0][0]),:,14:28].mean(axis=1), label=cell, color=color)
    ax.set_title(f"Representative command trajectory, episode {rep_id}"); ax.set_xlabel("Horizon step"); ax.set_ylabel("Native command value (uninterpreted units)"); ax.legend(fontsize=8)
    save("fig06_representative_trajectory", fig, [{"episode_id": rep_id, "time_index": t, "gt_arm_mean": float(target[t]), "persistence_arm_mean": float(persistence[t])} for t in range(30)], "Representative command-space trajectory; no physical units or Cartesian claim.", "Ground truth, persistence, matched Fusion and zero-video command trajectories."); figures.append({"figure": "fig06_representative_trajectory", "status": "PASS"})

    fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(target, color="black", label="GT")
    for cell, color in (("FF-P0", "#009E73"), ("FF-P1", "#CC79A7"), ("FF-P2", "#E69F00"), ("FF-P3", "#0072B2"), ("FF-P4", "#D55E00")):
        b=load_cell(cell); ax.plot(b["predictions_native"][int(np.where(b["episode_index"]==rep_id)[0][0]),:,14:28].mean(axis=1), label=cell, color=color)
    ax.set_title(f"Fusion temporal controls, episode {rep_id}"); ax.set_xlabel("Horizon step"); ax.set_ylabel("Native command value"); ax.legend(fontsize=7)
    save("fig07_temporal_trajectory", fig, [{"episode_id": rep_id, "time_index": t, "gt_arm_mean": float(target[t])} for t in range(30)], "Same episode and axis for chronological, repeated, reverse and shuffle controls.", "Fusion temporal-control trajectories."); figures.append({"figure": "fig07_temporal_trajectory", "status": "PASS"})

    # 8. Motion/horizon damage heatmap-like table.
    selected_cells = ["FF-P1", "FF-P2", "FF-P3", "FF-P4", "FF-R1", "FF-R2", "FF-R3"]
    effect_lookup = {(r["cell"], r["metric"]): r for r in effect_rows}
    horizons = ["arm_full", "arm_early", "arm_middle", "arm_late", "arm_endpoint"]
    matrix = np.asarray([[float(effect_lookup[(cell, metric)]["damage"]) for metric in horizons] for cell in selected_cells])
    fig, ax = plt.subplots(figsize=(8, 4.5)); image=ax.imshow(matrix, cmap="coolwarm", aspect="auto"); ax.set_xticks(range(len(horizons)), horizons, rotation=30); ax.set_yticks(range(len(selected_cells)), selected_cells); fig.colorbar(image, ax=ax, label="damage"); ax.set_title("Fusion damage by condition and horizon");
    save("fig08_motion_horizon_damage", fig, [{"cell": cell, **{metric: float(matrix[i,j]) for j, metric in enumerate(horizons)}} for i, cell in enumerate(selected_cells)], "Damage is shown for full and horizon slices; high-motion slices are retained in the CSV results.", "Condition by horizon damage heatmap."); figures.append({"figure": "fig08_motion_horizon_damage", "status": "PASS"})

    # 9. Joint x time damage heatmap.
    p0=load_cell("FF-P0"); r1=load_cell("FF-R1"); p0i={int(v):i for i,v in enumerate(p0["episode_index"].tolist())}; r1i={int(v):i for i,v in enumerate(r1["episode_index"].tolist())}; common=[int(v) for v in p0["episode_index"] if int(v) in r1i]; a=np.asarray([np.where(p0["episode_index"]==v)[0][0] for v in common]); b=np.asarray([r1i[v] for v in common]); heat=np.mean(np.abs(r1["predictions_standardized"][b]-r1["targets_standardized"][b]),axis=0)-np.mean(np.abs(p0["predictions_standardized"][a]-p0["targets_standardized"][a]),axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.5)); image=ax.imshow(heat[:,14:28].T, cmap="coolwarm", aspect="auto"); ax.set_xlabel("Horizon step"); ax.set_ylabel("Arm-14 joint index"); fig.colorbar(image, ax=ax, label="zero-video minus matched error"); ax.set_title("Fusion zero-video joint×time damage");
    save("fig09_joint_time_heatmap", fig, [{"time_index": t, "arm_joint_index": j, "damage": float(heat[t,14+j])} for t in range(30) for j in range(14)], "Joint×time error difference for zero-video versus matched Fusion.", "Joint by horizon heatmap of zero-video damage."); figures.append({"figure": "fig09_joint_time_heatmap", "status": "PASS"})

    # 10. Task support.
    task_selected=[r for r in task_rows if r["cell"] in {"FF-P0","FF-R2","FF-R3"}]; pivot={};
    for r in task_selected: pivot.setdefault(int(r["task_index"]),{})[r["cell"]]=float(r["arm_full_mae"])
    task_ids=sorted(pivot, key=lambda t: max(pivot[t].values())-min(pivot[t].values()), reverse=True)[:20]; fig, ax=plt.subplots(figsize=(10,4.5)); xx=np.arange(len(task_ids));
    for cell,color in (("FF-P0","#009E73"),("FF-R2","#0072B2"),("FF-R3","#D55E00")): ax.plot(xx,[pivot[t].get(cell,np.nan) for t in task_ids],"o-",label=cell,color=color)
    ax.set_xticks(xx, [str(t) for t in task_ids], rotation=60); ax.set_ylabel("Arm full MAE"); ax.set_title("Per-task Fusion support (largest spread tasks)"); ax.legend(fontsize=8)
    save("fig10_task_damage", fig, [{"task_index": t, **pivot[t]} for t in task_ids], "Per-task results are descriptive and retain task counts; no physical-unit claim.", "Per-task Arm-14 MAE for matched and mismatch Fusion controls."); figures.append({"figure": "fig10_task_damage", "status": "PASS"})

    # 11. Pairing QA.
    pair=rows(ROOT/"PAIRING_MANIFEST.csv"); cross_counts={}; same_eligible=sum(r["same_task_eligible"]=="True" for r in pair)
    for r in pair: cross_counts[int(r["cross_task_donor"])] = cross_counts.get(int(r["cross_task_donor"]), 0) + 1
    fig, ax=plt.subplots(figsize=(7,4)); ax.hist(list(cross_counts.values()), bins=np.arange(0.5,max(cross_counts.values())+1.5,1), color="#56B4E9", edgecolor="black"); ax.set_xlabel("Cross-task donor reuse count"); ax.set_ylabel("Number of donors"); ax.set_title(f"Pairing QA: same-task eligible {same_eligible}/614")
    save("fig11_pairing_qa", fig, [{"donor_reuse": k, "donor_count": v} for k,v in sorted({count:list(cross_counts.values()).count(count) for count in set(cross_counts.values())}.items())], "Pairing selection uses task and episode IDs only; same-task singleton recipients are NA.", "Histogram of cross-task donor reuse."); figures.append({"figure": "fig11_pairing_qa", "status": "PASS"})

    # 12. Decision summary.
    summary = [{"question": "matched_vs_zero", "evidence": "FF-R1 damage positive"}, {"question": "same_task_shortcut", "evidence": "FF-R2 close to FF-P0"}, {"question": "cross_task_dependence", "evidence": "FF-R3 damage positive"}, {"question": "temporal_order", "evidence": "reverse/shuffle damage smaller than repeat controls"}, {"question": "trajectory_noise", "evidence": "first-difference and TV reported separately"}]
    fig, ax=plt.subplots(figsize=(9,4)); ax.axis("off"); ax.table(cellText=[[r["question"],r["evidence"]] for r in summary], colLabels=["Question","Frozen evidence-consistent summary"], loc="center", cellLoc="left"); ax.set_title("Frozen-only attribution decision summary", pad=20)
    save("fig12_decision_summary", fig, summary, "This is an evidence-bounded exploratory summary; mechanisms use suggests/is-more-consistent-with language.", "Table of five attribution questions and evidence-consistent summaries."); figures.append({"figure": "fig12_decision_summary", "status": "PASS"})

    write_csv(ROOT / "figure_manifests/FIGURE_MANIFEST.csv", figures)
    write_json(ROOT / "figure_manifests/FIGURE_MANIFEST.json", {"status": "PASS", "count": len(figures), "figures": figures, "visual_qa": "pending_view_image"})
    print(json.dumps({"status": "PASS", "figures": len(figures)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
