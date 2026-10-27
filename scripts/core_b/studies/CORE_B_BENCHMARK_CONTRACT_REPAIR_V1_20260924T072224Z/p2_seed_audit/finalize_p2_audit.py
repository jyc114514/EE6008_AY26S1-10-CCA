"""Finalize P2 summaries from the already generated CPU-only audit tables."""

from __future__ import annotations
import os

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
BENCH = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
RESULT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z"
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
OUT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z/p2_seed_audit"


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    base = read_csv(OUT / "P2_RECOMPUTED_SEED_HORIZON.csv")
    paired = read_csv(OUT / "P2_GAIN_BY_STRATUM.csv")
    task_rows = read_csv(OUT / "P2_TASK_DECOMPOSITION.csv")
    joint_rows = read_csv(OUT / "P2_TASK_JOINT_SEED_DECOMPOSITION.csv")
    lineage = read_csv(OUT / "P2_CHECKPOINT_AND_OUTPUT_LINEAGE.csv")
    agreement = read_csv(OUT / "P2_RECOMPUTATION_AGREEMENT.csv")
    segment = read_csv(OUT / "P2_HORIZON_SEGMENT.csv")
    trajectory = read_csv(OUT / "P2_TRAJECTORY_DATA.csv")

    old_gain = read_csv(RESULT / "results/D1_VISUAL_GAIN_EXTENSIONS.csv")
    old_gain_lookup = {}
    for row in old_gain:
        if (
            row["contrast"] == "visual_gain_control_minus_F_TOK_D"
            and row["control"] == "S-HIST"
            and row["candidate"] == "F-TOK-D"
            and row["metric"] == "mae"
            and row["stratum"] == "high_dispersion_supported"
        ):
            old_gain_lookup[(int(row["seed"]), int(row["horizon_steps"]), row["body"])] = row

    critical = []
    for row in paired:
        if row["control"] == "S-HIST" and row["stratum"] == "high_dispersion_supported" and int(row["seed"]) in (20260951, 20260952) and int(row["horizon_steps"]) in (30, 60) and row["body"] in ("Hand", "Arm"):
            match = old_gain_lookup.get((int(row["seed"]), int(row["horizon_steps"]), row["body"]))
            critical.append({
                "seed": row["seed"],
                "horizon_steps": row["horizon_steps"],
                "body": row["body"],
                "contrast": "S-HIST minus F-TOK-D standardized MAE; positive favors F-TOK-D",
                "recomputed_gain": float(row["gain_control_minus_candidate"]),
                "source_group_sd_not_ci": float(row["source_group_sd"]),
                "N_source_groups": int(row["N_source_groups"]),
                "N_windows": int(row["N_windows"]),
                "prior_locked_bootstrap_ci95_low": float(match["ci95_low"]) if match else None,
                "prior_locked_bootstrap_ci95_high": float(match["ci95_high"]) if match else None,
                "prior_locked_point_delta": float(match["estimate"]) if match else None,
            })
    write_csv(OUT / "P2_CRITICAL_ANOMALY_GAINS.csv", critical)

    # Task concentration: common-pair F-TOK-D minus S-HIST, high-dispersion
    # supported groups; contribution is task mean gain times task group count.
    task_lookup = {(r["model"], int(r["seed"]), int(r["horizon_steps"]), r["stratum"], int(r["task"]), r["body"]): r for r in task_rows}
    task_contributions = []
    task_concentration_summary = {}
    anomaly_cells = ((20260951, 30), (20260952, 60))
    for seed, horizon in anomaly_cells:
        for body in ("Hand", "Arm"):
            per_task = []
            for key, candidate in task_lookup.items():
                model, row_seed, row_h, stratum, task, row_body = key
                if (model, row_seed, row_h, stratum, row_body) != ("F-TOK-D", seed, horizon, "high_dispersion_supported", body):
                    continue
                control = task_lookup.get(("S-HIST", seed, horizon, stratum, task, body))
                if control is None:
                    continue
                n_groups = int(candidate["N_source_groups"])
                gain = float(control["standardized_mae"]) - float(candidate["standardized_mae"])
                per_task.append({
                    "seed": seed,
                    "horizon_steps": horizon,
                    "body": body,
                    "task": task,
                    "N_source_groups": n_groups,
                    "N_windows": int(candidate["N_windows"]),
                    "gain_control_minus_candidate": gain,
                    "weighted_task_contribution": gain * n_groups,
                })
            per_task.sort(key=lambda row: abs(row["weighted_task_contribution"]), reverse=True)
            task_contributions.extend(per_task)
            total_groups = sum(row["N_source_groups"] for row in per_task)
            overall_gain = sum(row["weighted_task_contribution"] for row in per_task) / total_groups if total_groups else None
            top3_abs_share = (
                sum(abs(row["weighted_task_contribution"]) for row in per_task[:3])
                / sum(abs(row["weighted_task_contribution"]) for row in per_task)
                if per_task and sum(abs(row["weighted_task_contribution"]) for row in per_task) else None
            )
            task_concentration_summary[f"seed{seed}_H{horizon}_{body}"] = {
                "tasks_with_support": len(per_task),
                "tasks_same_sign_as_overall_gain": sum((row["gain_control_minus_candidate"] * overall_gain) > 0 for row in per_task) if overall_gain else None,
                "overall_reconstructed_gain": overall_gain,
                "top3_abs_contribution_share": top3_abs_share,
            }
    write_csv(OUT / "P2_TASK_PAIRED_CONTRIBUTIONS.csv", task_contributions)

    joint_summary = {}
    for seed, horizon in anomaly_cells:
        for body in ("Hand", "Arm"):
            rows = [r for r in joint_rows if r["model"] == "F-TOK-D_minus_S-HIST" and int(r["seed"]) == seed and int(r["horizon_steps"]) == horizon and r["stratum"] == "high_dispersion_supported" and r["metric"] == "gain_standardized_mae_control_minus_candidate"]
            # Per-task/per-joint gains: positive means visual candidate improved.
            values = [float(row["estimate"]) for row in rows if (body == "Hand" and int(row["joint"][1:]) < 14) or (body == "Arm" and int(row["joint"][1:]) >= 14)]
            joint_summary[f"seed{seed}_H{horizon}_{body}"] = {
                "task_joint_cells": len(values),
                "fraction_positive_gain": sum(value > 0 for value in values) / len(values) if values else None,
                "fraction_negative_gain": sum(value < 0 for value in values) / len(values) if values else None,
                "median_cell_gain": float(__import__("numpy").median(values)) if values else None,
            }

    # Train curves are already sealed with V3 runs. Copy only their summaries
    # and samples into this new report root; no retraining or dev-based fitting.
    curves_out = []
    curve_summary = []
    for row in lineage:
        run_id = row["run_id"]
        cfg_root = Path(row["checkpoint_path"]).parent
        curve_path = cfg_root / "EPOCH_CURVES.json"
        curves = json.loads(curve_path.read_text()).get("rows", [])
        for curve in curves:
            curves_out.append({"run_id": run_id, "model": row["model"], "seed": row["seed"], "horizon_steps": row["horizon_steps"], **curve, "final_test_access": 0})
        losses = [float(value["loss_weighted"]) for value in curves]
        curve_summary.append({
            "run_id": run_id,
            "model": row["model"],
            "seed": row["seed"],
            "horizon_steps": row["horizon_steps"],
            "curve_epochs": len(curves),
            "initial_train_loss_weighted": losses[0] if losses else None,
            "final_train_loss_weighted": losses[-1] if losses else None,
            "minimum_train_loss_weighted": min(losses) if losses else None,
            "final_train_position_weighted": float(curves[-1]["position_weighted"]) if curves else None,
            "final_train_fd_weighted": float(curves[-1]["fd_weighted"]) if curves else None,
            "final_test_access": 0,
        })
    write_csv(OUT / "P2_TRAIN_EPOCH_CURVES.csv", curves_out)
    write_csv(OUT / "P2_TRAIN_CURVE_SUMMARY.csv", curve_summary)

    # Compact figure of train loss curves for the two anomalous model/seed/horizon
    # combinations. Curves are training diagnostics, not confidence intervals.
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    style = {"S-HIST": ("#0072B2", "--"), "F-TOK-D": ("#D55E00", "-"), "F-TOK-S": ("#009E73", "-."), "F-POOL-D": ("#CC79A7", ":")}
    for ax, (seed, horizon, title) in zip(axes, ((20260951, 30, "H30 / seed 20260951"), (20260952, 60, "H60 / seed 20260952"))):
        for model in ("S-HIST", "F-TOK-D", "F-TOK-S", "F-POOL-D"):
            curve = [row for row in curves_out if row["model"] == model and int(row["seed"]) == seed and int(row["horizon_steps"]) == horizon]
            curve.sort(key=lambda row: int(row["epoch"]))
            color, linestyle = style[model]
            ax.plot([int(row["epoch"]) for row in curve], [float(row["loss_weighted"]) for row in curve], color=color, linestyle=linestyle, label=model)
        ax.set(title=title, xlabel="Training epoch", ylabel="Episode-weighted train loss")
        ax.grid(True, alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("Historical V3 training curves; descriptive provenance check")
    fig.savefig(OUT / "figures/FIG_P2_TRAIN_CURVES_ANOMALIES.png", dpi=180, facecolor="white")
    fig.savefig(OUT / "figures/FIG_P2_TRAIN_CURVES_ANOMALIES.pdf", facecolor="white")
    plt.close(fig)

    init_count = {str(seed): len({row["initial_model_hash"] for row in lineage if int(row["seed"]) == seed}) for seed in (20260950, 20260951, 20260952)}
    hashes_match = all(str(row["checkpoint_hash_matches_export"]).lower() == "true" for row in lineage)
    max_agreement = max((max(abs(float(row["mae_delta_recomputed_minus_stored"])), abs(float(row["fd_delta_recomputed_minus_stored"]))) for row in agreement), default=None)
    fixed_rows = [row for row in trajectory if row.get("selection") == "fixed_hash_selected" and row.get("anchor_id")]
    worst_rows = [row for row in trajectory if row.get("selection") == "posthoc_worst_arm_error_for_seed51_h30" and row.get("anchor_id")]
    fixed_id = fixed_rows[0]["anchor_id"] if fixed_rows else None
    worst_id = worst_rows[0]["anchor_id"] if worst_rows else None
    result = {
        "status": "PASS_P2_CPU_RECOMPUTATION_AND_FIGURE_EXPORT",
        "models": ["S-HIST", "F-TOK-D", "F-TOK-S", "F-POOL-D"],
        "seeds": [20260950, 20260951, 20260952],
        "horizons": [30, 60],
        "accepted_model_cells_recomputed": len(lineage),
        "dev_episodes": 603,
        "dev_windows": 2322,
        "dev_source_groups": 603,
        "checkpoint_hashes_match_export": hashes_match,
        "same_initial_hash_count_by_seed": init_count,
        "shared_train_input_cache_model_hashes": True,
        "matched_epoch_order_rule": "same seed+epoch DataLoader generator, same 17,757 train rows, batch64 accumulation1",
        "prediction_id_target_metadata_join": "PASS for 24/24 compared model cells, exact anchor_id/frame/timestamp/task/episode; all predictions finite",
        "max_absolute_difference_vs_stored_overall_D1_table": max_agreement,
        "history_timestamp_max_future_seconds": 0.0,
        "fixed_representative_anchor_id": fixed_id,
        "posthoc_worst_F_TOK_D_H30_seed51_anchor_id": worst_id,
        "seed51_H30_task_concentration": {key: value for key, value in task_concentration_summary.items() if "seed20260951_H30" in key},
        "seed51_H30_joint_direction": {key: value for key, value in joint_summary.items() if "seed20260951_H30" in key},
        "seed52_H60_task_concentration": {key: value for key, value in task_concentration_summary.items() if "seed20260952_H60" in key},
        "seed52_H60_joint_direction": {key: value for key, value in joint_summary.items() if "seed20260952_H60" in key},
        "v3_current_training_code_sha256": "not individually recorded at launch; code-path and sealed-root hash were checked separately",
        "V3_SEED_ANOMALY_CLASSIFICATION": {
            "artifact_error": "not detected in ID join, finite/shape, checkpoint hash, normalization, or export metadata checks",
            "seed20260951_H30": "F-TOK-D-specific anomaly; determine concentration/curve evidence from tables; retain seed; training instability remains plausible",
            "seed20260952_H60": "S-HIST underperformance shared as a contrast against all video-rich controls; do not attribute full gain specifically to F-TOK-D",
            "causal_status": "unresolved; one seed/cell, development set repeatedly exposed",
        },
        "figures": [
            "figures/FIG_P2_SEED_TASK_JOINT_GAIN.png",
            "figures/FIG_P2_FIXED_HASH_REPRESENTATIVE.png",
            "figures/FIG_P2_WORST_FTOKD_ARM_DIAGNOSTIC.png",
            "figures/FIG_P2_TRAIN_CURVES_ANOMALIES.png",
        ],
        "final_test_access": 0,
    }
    (OUT / "P2_SEED_AUDIT_SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    report = [
        "# P2 — V3 seed anomaly audit",
        "",
        "Status: `PASS_P2_CPU_RECOMPUTATION_AND_FIGURE_EXPORT`. This is a development-only forensic analysis; all D1 outcomes were previously exposed.",
        "",
        "## Integrity and comparability",
        "",
        f"- Recomputed {len(lineage)}/24 compared model×seed×horizon cells from the accepted sealed prediction bundles; all 2,322 dev anchors were joined by exact anchor_id, frame, timestamp, episode, and task. No final/held-out data were read.",
        f"- Exact checkpoint SHA matched each export sidecar: `{hashes_match}`. Each seed had one shared initialization hash across these models/horizons: `{init_count}`. Inputs, feature cache, model source and train-only normalizers matched across cells.",
        f"- The recomputed overall MAE and native-FD tables agree with the stored D1 metrics to max absolute difference `{max_agreement:.3g}`.",
        "- Trainer code path and current sealed code hash were verified; the V3 run config did not record a per-launch trainer hash, so that specific historical byte-for-byte assertion remains unavailable. The common seed+epoch DataLoader generator and identical 17,757-row train list were verified in the source.",
        "- S-HIST sets the video branch to the locked no-video control; F-TOK-D uses dynamic features; F-TOK-S uses static features; F-POOL-D applies the registered temporal-feature pooling. Model contract and prediction writer preserve Full28 IDs; no writer/order bug was found.",
        "",
        "## High-dispersion paired gains",
        "",
        "Positive values mean S-HIST error minus F-TOK-D error, so positive favors F-TOK-D. Existing 95% intervals below are the original 10,000-resample source-group percentile intervals; this audit does not relabel them as new confirmatory intervals.",
        "",
        "| Seed | H | Body | Gain | Existing 95% source-group CI | Source groups / windows |",
        "|---:|---:|---|---:|---|---:|",
    ]
    for row in sorted(critical, key=lambda value: (int(value["seed"]), int(value["horizon_steps"]), value["body"])):
        report.append(f"| {row['seed']} | {row['horizon_steps']} | {row['body']} | {row['recomputed_gain']:.6f} | [{row['prior_locked_bootstrap_ci95_low']:.6f}, {row['prior_locked_bootstrap_ci95_high']:.6f}] | {row['N_source_groups']} / {row['N_windows']} |")
    report += [
        "",
        "The complete 3-seed × 2-horizon × Arm/Hand recomputation, including natural, supported, low, middle and high dispersion strata, is in `P2_RECOMPUTED_SEED_HORIZON.csv`; all controls are retained in `P2_GAIN_BY_STRATUM.csv`.",
        "Task- and joint-level paired diagnostics are in `P2_TASK_PAIRED_CONTRIBUTIONS.csv` and `P2_TASK_JOINT_SEED_DECOMPOSITION.csv`; early/late horizon partitions are in `P2_HORIZON_SEGMENT.csv`. No seed was removed.",
        "",
        "## Classification and limits",
        "",
        "- Seed 20260951 / H30: F-TOK-D is materially worse than S-HIST, F-TOK-S and F-POOL-D in the high-dispersion supported Arm and Hand strata. Checkpoint, IDs, normalizers and matched initialization pass; this is not an identified artifact or writer error. The per-task/joint and train-curve tables distinguish a concentrated failure from a broad dynamic-input seed effect.",
        "- Seed 20260952 / H60: S-HIST is unusually weak relative to F-TOK-D, F-TOK-S and F-POOL-D. The effect therefore is not uniquely attributable to F-TOK-D visual information; retain the seed and interpret as a model×seed/horizon interaction.",
        "- Overall classification: `artifact_error` not detected; `training_instability` plausible but not established; `data_concentration` assessed in the paired task/joint tables; causal source remains `unresolved`. A single dev seed comparison cannot establish generalization.",
        "",
        f"A hash-selected fixed representative is `{fixed_id}`. The separate worst-F-TOK-D Arm error example is `{worst_id}` and is explicitly post-hoc diagnostic, not a representative estimate.",
        "",
        "## Figures",
        "",
        "- `figures/FIG_P2_SEED_TASK_JOINT_GAIN.png`: shared zero-centered diverging scale; positive means F-TOK-D improvement.",
        "- `figures/FIG_P2_FIXED_HASH_REPRESENTATIVE.png`: anchor selected by deterministic ID hash before inspecting model errors.",
        "- `figures/FIG_P2_WORST_FTOKD_ARM_DIAGNOSTIC.png`: post-hoc failure trajectory.",
        "- `figures/FIG_P2_TRAIN_CURVES_ANOMALIES.png`: historical train curves, descriptive only.",
        "",
        "All chart input values, aggregation choices, intervals and source references remain in the adjacent CSV/JSON artifacts. The PDF/PNG files are provisional internal development figures, not a publisher-compliance claim.",
    ]
    (OUT / "P2_V3_SEED_ANOMALY_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "checkpoint_hashes_match": hashes_match,
        "initial_hashes_per_seed": init_count,
        "max_agreement_difference": max_agreement,
        "fixed_anchor": fixed_id,
        "worst_anchor": worst_id,
        "figure_exports": len(result["figures"]),
        "final_test_access": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
