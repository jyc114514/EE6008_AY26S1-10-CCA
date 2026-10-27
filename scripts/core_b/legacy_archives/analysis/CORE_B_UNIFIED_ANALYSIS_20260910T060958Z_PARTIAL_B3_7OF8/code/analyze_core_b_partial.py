#!/usr/bin/env python3
"""Seal a provisional Core B analysis while B3-FUS-S-L is still missing.

Only already complete B3 pairs are included.  The script intentionally emits
no factorial interaction, candidate gate, or final Holm conclusion.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


OUT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260910T060958Z_PARTIAL_B3_7OF8'))
OLD_ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/analysis/CORE_B_UNIFIED_ANALYSIS_20260909T163929Z'))


def load_module():
    spec = importlib.util.spec_from_file_location("core_b_full_harvest", OLD_ROOT / "code/analyze_core_b_harvest.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.OUT = OUT
    return module


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def partial_b3_records(m, inputs, b2_records):
    by = {r["cell"]: r for r in b2_records if r["stage"] == "B2"}
    specs = [
        ("B3-V-D-F", by["B2-V-0125"], "video_only", "dynamic", "frozen", False, "B2-V-0125"),
        ("B3-V-D-L", m.EXP / "runs/corrected/B3-V-D-L", m.EXP / "runs/corrected/B3-V-D-L/predictions_best.npz", "video_only", "dynamic", "lora", True, ""),
        ("B3-V-S-F", m.EXP / "runs/B3-V-S-F", m.EXP / "runs/B3-V-S-F/predictions_best.npz", "video_only", "static", "frozen", False, ""),
        ("B3-V-S-L", m.EXP / "runs/corrected/B3-V-S-L", m.EXP / "runs/corrected/B3-V-S-L/predictions_best.npz", "video_only", "static", "lora", True, ""),
        ("B3-FUS-D-F", by["B2-FUS-0125"], "fusion", "dynamic", "frozen", False, "B2-FUS-0125"),
        ("B3-FUS-D-L", m.EXP / "runs/corrected/B3-FUS-D-L", m.EXP / "runs/corrected/B3-FUS-D-L/predictions_best.npz", "fusion", "dynamic", "lora", True, ""),
        ("B3-FUS-S-F", m.EXP / "runs/B3-FUS-S-F", m.EXP / "runs/B3-FUS-S-F/predictions_best.npz", "fusion", "static", "frozen", False, ""),
    ]
    records = []
    for item in specs:
        cell = item[0]
        if item[-1]:
            source = item[1]
            record = m.record_from_prediction(cell=cell, stage="B3", branch=source["branch"], video_input=source["video_input"], adaptation=source["adaptation"], prediction_path=source["prediction_path"], run_dir=source["run_dir"], inputs=inputs, source_kind="exact_reuse_of_selected_B2_dense_control", canonical_status="accepted_reused", required_keys={"episode_index", "original_task_index", "targets_standardized", "predictions_standardized", "targets_native", "predictions_native", "last_observed_action"}, reused_from=item[-1])
        else:
            run_dir, prediction, branch, video, adaptation, raw, _ = item[1:]
            record = m.record_from_prediction(cell=cell, stage="B3", branch=branch, video_input=video, adaptation=adaptation, prediction_path=prediction, run_dir=run_dir, inputs=inputs, source_kind="corrected_raw_lora" if raw else "pooled_static_runner", canonical_status="accepted", required_keys={"episode_index", "targets_standardized", "predictions_standardized", "targets_native", "predictions_native"}, raw_lora=raw)
        record["strategy"] = "dense_0.125s_32_pixel_frames"; record["encoder_slots"] = 32
        records.append(record)
    return records


def provisional_row(candidate, reference, indices):
    cand = candidate["metrics"]["standardized"]["episode_arm_mae"]
    ref = reference["metrics"]["standardized"]["episode_arm_mae"]
    samples = (ref - cand)[indices].mean(axis=1)
    relative = 1.0 - cand[indices].mean(axis=1) / np.maximum(ref[indices].mean(axis=1), 1e-12)
    lo, hi = np.quantile(samples, [0.025, 0.975]); rlo, rhi = np.quantile(relative, [0.025, 0.975])
    return {"comparison": f"{candidate['cell']}_vs_{reference['cell']}", "candidate": candidate["cell"], "reference": reference["cell"], "metric": "standardized_arm14_trajectory_mae", "delta_definition": "reference MAE minus candidate MAE; positive means candidate lower error", "improvement_point": float((ref-cand).mean()), "ci95_lower": float(lo), "ci95_upper": float(hi), "relative_improvement_point": float(1-cand.mean()/max(ref.mean(),1e-12)), "relative_ci95_lower": float(rlo), "relative_ci95_upper": float(rhi), "candidate_hand_mae": float(candidate["metrics"]["standardized"]["hand_mae"]), "reference_hand_mae": float(reference["metrics"]["standardized"]["hand_mae"]), "candidate_full28_mae": float(candidate["metrics"]["standardized"]["full28_mae"]), "reference_full28_mae": float(reference["metrics"]["standardized"]["full28_mae"]), "bootstrap_n": int(indices.shape[0]), "bootstrap_seed": 20260910, "final_test_access": 0, "p_value": "not_calculated_in_provisional_snapshot", "holm": "not_calculated_in_provisional_snapshot"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    m = load_module()
    inputs = m.load_canonical_inputs()
    b1, b1_acceptance = m.load_b1_records(inputs)
    b2, b2_acceptance, proxy_inventory = m.load_b2_records(inputs)
    b3 = partial_b3_records(m, inputs, b2)
    controls = m.add_controls(inputs)
    proxy = [r for r in b2 if r["stage"] == "B2_PROXY"]
    b2_formal = [r for r in b2 if r["stage"] == "B2"]
    formal = b1 + b2_formal + b3
    m.attach_metrics(formal + controls + proxy, inputs)
    by = {r["cell"]: r for r in formal + controls}
    rng = np.random.default_rng(20260910)
    indices = rng.integers(0, 614, size=(10_000, 614), dtype=np.uint16)
    np.save(OUT / "data/paired_episode_bootstrap_indices.npy", indices, allow_pickle=False)
    write_json(OUT / "BOOTSTRAP_RULE.json", {"status": "provisional", "replicates": 10_000, "seed": 20260910, "unit": "episode", "same_indices_for_all_provisional_effects": True, "ci": "percentile 2.5%/97.5%", "final_test_access": 0})
    comparisons = []
    for candidate, reference in [
        ("B3-V-D-L", "B3-V-D-F"),
        ("B3-V-S-L", "B3-V-S-F"),
        ("B3-FUS-D-L", "B3-FUS-D-F"),
        ("B3-V-D-F", "B3-V-S-F"),
        ("B3-FUS-D-F", "B3-FUS-S-F"),
        ("B3-V-D-L", "PERSISTENCE"),
        ("B3-V-S-L", "PERSISTENCE"),
        ("B3-FUS-D-L", "B-STATE"),
    ]:
        comparisons.append(provisional_row(by[candidate], by[reference], indices))
    write_csv(OUT / "CORE_B_PROVISIONAL_PAIRWISE.csv", comparisons)
    write_json(OUT / "CORE_B_PROVISIONAL_PAIRWISE.json", {"status": "provisional", "rows": comparisons, "final_holm": "not_calculated", "fusion_interaction": "not_calculated", "candidate_gate": "not_calculated", "missing_cell": "B3-FUS-S-L"})
    results = [m.metric_row(r) for r in formal + controls]
    write_csv(OUT / "CORE_B_PROVISIONAL_RESULTS.csv", results)
    write_json(OUT / "CORE_B_PROVISIONAL_RESULTS.json", {"status": "provisional", "rows": results, "b3_logical_coverage": "7/8", "final_test_access": 0})
    write_csv(OUT / "CORE_B_PROVISIONAL_PER_EPISODE.csv", m.per_episode_rows(formal + controls, inputs))
    write_csv(OUT / "CORE_B_PROVISIONAL_PER_TIMESTEP.csv", m.per_timestep_rows(formal + controls))
    write_csv(OUT / "CORE_B_PROVISIONAL_PER_JOINT_TIME.csv", m.per_joint_time_rows(formal + controls))
    write_csv(OUT / "CORE_B_PROVISIONAL_PER_TASK.csv", m.per_task_rows(formal + controls, inputs))
    write_json(OUT / "B1_REACCEPTANCE.json", {"status": "pass", "logical_coverage": "7/7", "accepted_count": 7, "core_b_b1_retrained": False, "final_test_access": 0, "rows": b1_acceptance})
    write_csv(OUT / "B1_REACCEPTANCE.csv", b1_acceptance)
    write_json(OUT / "B2_REACCEPTANCE.json", {"status": "pass", "logical_coverage": "6/6", "pixel_observation_protocol_verified": True, "selected_strategy": "dense_0.125s_32_pixel_frames", "final_test_access": 0, "rows": b2_acceptance})
    write_csv(OUT / "B2_REACCEPTANCE.csv", b2_acceptance)
    write_csv(OUT / "B2_PROXY_INVENTORY.csv", proxy_inventory)
    write_csv(OUT / "B2_PROXY_RESULTS.csv", [m.metric_row(r) for r in proxy])
    write_json(OUT / "B2_PROXY_RECLASSIFICATION.json", {"status": "auxiliary_token_aggregation_proxy", "formal_selection_included": False, "missing_from_formal": [r["cell"] for r in proxy], "final_test_access": 0})
    controller = m.read_json(m.B3_CONTROLLER)
    inventory = m.write_inventory(b1, b2_formal, b3, proxy, controller)
    inventory.append({"stage": "B3", "cell": "B3-FUS-S-L", "research_question": "corrected raw-LoRA static fusion", "branch": "fusion", "video_input": "static", "adaptation": "lora", "protocol_id": "core_b_b3_raw_lora_20260908_batch8_accum4_corrected_feature_norm", "seed": 20260903, "population": "train=2741;dev=614", "target_shape": "614x30x28", "source_kind": "currently_running_or_missing", "status": "provisional_missing_from_snapshot", "reused_from": "", "prediction_path": m.EXP / "runs/corrected/B3-FUS-S-L", "final_test_access": 0})
    write_csv(OUT / "CORE_B_LOGICAL_CELL_INVENTORY.csv", inventory)
    write_json(OUT / "CORE_B_LOGICAL_CELL_INVENTORY.json", {"status": "provisional", "logical_coverage": "B1 7/7; B2 6/6; B3 7/8", "missing": "B3-FUS-S-L", "rows": inventory, "final_test_access": 0})
    lineage = m.write_lineage(formal + controls, inputs)
    write_csv(OUT / "SOURCE_MANIFEST.csv", lineage); write_json(OUT / "SOURCE_MANIFEST.json", lineage)
    write_json(OUT / "LIVE_STATE_RECONCILIATION.json", {"observed_utc": m.utc_now(), "controller_path": m.B3_CONTROLLER, "controller_status": controller, "b1": "7/7", "b2_true_pixel": "6/6", "b3_available": "7/8", "missing_cell": "B3-FUS-S-L", "final_holm": "not_calculated", "fusion_interaction": "not_calculated", "candidate_gate": "not_calculated", "new_gpu_jobs_started": False, "final_test_access": 0})
    (OUT / "LIVE_STATE_RECONCILIATION.md").write_text("# Provisional Core B live state\n\nB1=7/7 and true-pixel B2=6/6 are complete. B3 has seven usable logical cells; `B3-FUS-S-L` is missing from this snapshot. Only provisional paired effects and bootstrap CIs are reported. Full factorial, fusion interaction, candidate gate and final Holm conclusions are deliberately withheld.\n", encoding="utf-8")
    (OUT / "CORE_B_PROVISIONAL_SUMMARY.zh-CN.md").write_text("# Core B provisional analysis\n\nB1 canonical 7/7 and true-pixel B2 6/6 were independently revalidated. B3 currently contributes seven complete logical cells; `B3-FUS-S-L` is not included. This snapshot reports only complete dynamic-LoRA, video-static-LoRA and complete frozen-pair effects with 10,000 episode bootstrap CIs. It does not calculate the complete B3 factorial, fusion interaction, candidate gate or final Holm conclusion.\n\nThe old contextual-token sparse B2 runs remain auxiliary proxy evidence and are excluded from formal pixel-sampling selection. Final-test access is 0.\n", encoding="utf-8")
    write_json(OUT / "ANALYSIS_PROTOCOL_LOCKED.json", {"analysis_id": "CORE_B_UNIFIED_ANALYSIS_20260910T060958Z_PARTIAL_B3_7OF8", "status": "provisional", "b1": "7/7", "b2_true_pixel": "6/6", "b3": "7/8", "missing_cell": "B3-FUS-S-L", "reported_statistics": ["paired episode bootstrap effect and 95% CI"], "withheld_statistics": ["complete B3 factorial interaction", "fusion interaction", "candidate gate", "final Holm conclusion"], "unit": "episode", "bootstrap_replicates": 10000, "bootstrap_seed": 20260910, "final_test_access": 0, "new_gpu_jobs_started": False})
    write_json(OUT / "ANALYSIS_STATUS.json", {"status": "provisional", "analysis_id": "CORE_B_UNIFIED_ANALYSIS_20260910T060958Z_PARTIAL_B3_7OF8", "b1": "7/7", "b2_true_pixel": "6/6", "b3": "7/8", "missing_cell": "B3-FUS-S-L", "final_test_access": 0, "full_factorial": "withheld", "fusion_interaction": "withheld", "candidate_gate": "withheld", "final_holm": "withheld"})
    print(json.dumps({"status": "provisional", "b1": "7/7", "b2_true_pixel": "6/6", "b3": "7/8", "missing": "B3-FUS-S-L", "out": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
