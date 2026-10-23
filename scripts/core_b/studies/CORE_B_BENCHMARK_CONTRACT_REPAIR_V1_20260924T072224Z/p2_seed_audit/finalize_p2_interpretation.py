"""Add evidence-based interpretation after task/joint and loss-curve checks."""

from __future__ import annotations
import os

import csv
import json
from pathlib import Path


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z/p2_seed_audit"


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    base = json.loads((OUT / "P2_SEED_AUDIT_SUMMARY.json").read_text())
    curve = rows(OUT / "P2_TRAIN_CURVE_SUMMARY.csv")
    ranges = rows(OUT / "P2_TRAJECTORY_DATA.csv")
    curve_map = {(r["model"], int(r["seed"]), int(r["horizon_steps"])): r for r in curve}
    range_map = {
        (r["model"], int(r["seed"]), int(r["horizon_steps"])): r
        for r in ranges if not r.get("selection") and not r.get("record_type")
    }
    key_51 = (20260951, 30)
    key_52 = (20260952, 60)
    train_51 = {model: float(curve_map[(model, key_51[0], key_51[1])]["final_train_loss_weighted"]) for model in ("S-HIST", "F-TOK-D", "F-TOK-S", "F-POOL-D")}
    train_52 = {model: float(curve_map[(model, key_52[0], key_52[1])]["final_train_loss_weighted"]) for model in ("S-HIST", "F-TOK-D", "F-TOK-S", "F-POOL-D")}
    range_51 = {model: float(range_map[(model, key_51[0], key_51[1])]["prediction_max"]) for model in train_51}
    range_52 = {model: float(range_map[(model, key_52[0], key_52[1])]["prediction_max"]) for model in train_52}
    task_summary = base["seed51_H30_task_concentration"] | base["seed52_H60_task_concentration"]
    joint_summary = base["seed51_H30_joint_direction"] | base["seed52_H60_joint_direction"]

    final = dict(base)
    final["classification_evidence"] = {
        "artifact_error": "not detected: all 24 checkpoint hashes/sidecars, IDs, frames, timestamps, target joins, finite outputs, shared input/cache/model/normalizer hashes pass.",
        "seed20260951_H30": {
            "high_supported_gain_s_hist_minus_f_tok_d": {"Hand": -0.027453432163679438, "Arm": -0.029802147269849207},
            "final_train_loss_by_model": train_51,
            "predicted_max_native_by_model": range_51,
            "task_consistency": {key: value for key, value in base["seed51_H30_task_concentration"].items()},
            "joint_direction": {key: value for key, value in base["seed51_H30_joint_direction"].items()},
            "interpretation": "F-TOK-D alone has worse high-stratum dev errors and materially higher training loss than the three matched controls, despite identical initialization and epoch order. This is strongest evidence for condition-specific training instability/optimization sensitivity; no index/checkpoint artifact or dominant small set of tasks was found.",
        },
        "seed20260952_H60": {
            "high_supported_gain_s_hist_minus_f_tok_d": {"Hand": 0.07288895002684102, "Arm": 0.0753810033217714},
            "final_train_loss_by_model": train_52,
            "predicted_max_native_by_model": range_52,
            "task_consistency": {key: value for key, value in base["seed52_H60_task_concentration"].items()},
            "joint_direction": {key: value for key, value in base["seed52_H60_joint_direction"].items()},
            "interpretation": "S-HIST has much higher final training loss and a compressed predicted maximum; F-TOK-D, F-TOK-S and F-POOL-D have similar scores. The apparent D gain is not unique to dynamic F-TOK-D and is consistent with an H60 state-only optimization failure/seed interaction.",
        },
        "data_concentration": "not the primary explanation: high-stratum task gains have 92 supported tasks; top 3 task absolute contribution share is about 12–20%; task×joint directions are mixed but majority follow each aggregate anomaly.",
        "training_instability": "strongly supported as the leading descriptive explanation, but not proven causal by one seed per cell and repeated development use.",
        "overall_causal_status": "unresolved; retain all seeds and avoid confirmation/generalization language.",
    }
    (OUT / "P2_SEED_AUDIT_FINAL.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")

    report = [
        "# P2 — V3 anomalous seed audit (final interpretation)",
        "",
        "Status: complete CPU reanalysis of the 24 relevant model×seed×horizon bundles. Development-only; all previous D1 outcomes were already exposed.",
        "",
        "## Provenance and replay checks",
        "",
        f"- All bundles joined exactly to the same 2,322 dev anchor IDs, episodes, tasks, frames and timestamps; normalized/native metrics reproduce the stored overall D1 table to max absolute difference {base['max_absolute_difference_vs_stored_overall_D1_table']:.3g}.",
        f"- All 24 checkpoint SHA values equal their prediction sidecar hashes; seed-matched initialization hash counts are {base['same_initial_hash_count_by_seed']}. Every compared run reports 40 epochs and 11,120 optimizer steps, with common train input/cache/model and normalizer hashes.",
        "- The training entrypoint uses the same seed+epoch DataLoader generator on the identical train rows. Export writer stores row IDs and timestamps and evaluation rejoins by ID. No wrong-writer, missing-modality, checkpoint, or ID-order artifact was found. Historical per-run launch-time trainer hash was not recorded.",
        "",
        "## The two outliers are different",
        "",
        "| Seed / horizon | Control | F-TOK-D | F-TOK-S | F-POOL-D | What the paired result means |",
        "|---|---:|---:|---:|---:|---|",
        "| 20260951 / H30, high-supported Hand sMAE | S-HIST 0.31042 | 0.33787 | 0.31155 | 0.30912 | Dynamic-token F-TOK-D alone is worse; gain S-HIST−F-TOK-D = −0.02745 [−0.03570, −0.01986]. |",
        "| 20260951 / H30, high-supported Arm sMAE | S-HIST 0.18037 | 0.21018 | 0.17744 | 0.17771 | Same dynamic-token-specific degradation; gain = −0.02980 [−0.03329, −0.02641]. |",
        "| 20260952 / H60, high-supported Hand sMAE | S-HIST 0.44294 | 0.37005 | 0.36846 | 0.37230 | S-HIST is the weak control; the three video-rich cells are similar; gain = +0.07289 [+0.06063, +0.08595]. |",
        "| 20260952 / H60, high-supported Arm sMAE | S-HIST 0.35570 | 0.28031 | 0.28757 | 0.28160 | Again not unique to F-TOK-D; gain = +0.07538 [+0.06837, +0.08264]. |",
        "",
        "Intervals are the existing 10,000-draw source-group percentile CIs; this forensic replay did not add a confirmatory test family.",
        "",
        "## Training and decomposition evidence",
        "",
        f"For H30/seed20260951, final weighted train loss is F-TOK-D {train_51['F-TOK-D']:.4f} versus S-HIST {train_51['S-HIST']:.4f}, F-TOK-S {train_51['F-TOK-S']:.4f}, and F-POOL-D {train_51['F-POOL-D']:.4f}. For H60/seed20260952, S-HIST is {train_52['S-HIST']:.4f} versus F-TOK-D {train_52['F-TOK-D']:.4f}, F-TOK-S {train_52['F-TOK-S']:.4f}, and F-POOL-D {train_52['F-POOL-D']:.4f}. The output maximum also contracts for seed52/H60 S-HIST (1.747) relative to the raw target max (1.960).",
        f"Seed51/H30 high-stratum effects span 92 tasks: {base['seed51_H30_task_concentration']['seed20260951_H30_Hand']['tasks_same_sign_as_overall_gain']}/92 Hand and {base['seed51_H30_task_concentration']['seed20260951_H30_Arm']['tasks_same_sign_as_overall_gain']}/92 Arm task estimates share the aggregate degradation; top-3 task contribution shares are 19.9% and 14.0%. Joint-level cell gains are negative in 58.9% Hand and 67.3% Arm cells.",
        f"Seed52/H60 improvements span 92 tasks: {base['seed52_H60_task_concentration']['seed20260952_H60_Hand']['tasks_same_sign_as_overall_gain']}/92 Hand and {base['seed52_H60_task_concentration']['seed20260952_H60_Arm']['tasks_same_sign_as_overall_gain']}/92 Arm task estimates favor video-rich cells; top-3 shares are 14.2% and 12.5%.",
        "",
        "The leading descriptive explanation is seed×horizon×input-condition training instability, not an ID/checkpoint artifact and not a small number of tasks. This is still not a causal or generalization claim: there is one training seed per cell in this study and the dev set was repeatedly examined. Preserve all seeds.",
        "",
        f"Fixed hash-selected representative: `{base['fixed_representative_anchor_id']}`. Separate post-hoc worst F-TOK-D Arm diagnostic: `{base['posthoc_worst_seed51_h30_anchor_id']}`.",
        "",
        "Final-test/held-out access: `0`.",
    ]
    (OUT / "P2_V3_SEED_ANOMALY_AUDIT_FINAL.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_FINAL_P2_INTERPRETATION", "classification": final["classification_evidence"]["overall_causal_status"], "final_test_access": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
