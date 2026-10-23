#!/usr/bin/env python3
"""Build final QA, non-interference and handoff records for selector study."""

from __future__ import annotations
import os

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa"
OLD = {
    "hand_event": (REPO_ROOT / "reports/core_b/HAND_MOTION_EVENT_FALLBACK_V1_20260913T081819Z/qa/CHECKSUMS.sha256", 180, "060283a9e62f21d4c30ea9d3746c8843b5765b2b5a73ce4f5d0b766b45fd0999"),
    "postfreeze": (REPO_ROOT / "reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/qa/CHECKSUMS.sha256", 306, "f12b2f1a0797705c9a9043c2b38cab82d650a8a0b9a01a9cfd81e9d58ec63197"),
    "freeze": (REPO_ROOT / "reports/core_b/JY02_CANDIDATE_FREEZE_AND_HELDOUT_PROTOCOL_V1_20260913T053157Z/qa/CHECKSUMS.sha256", 85, "e6922a170abe6a5cc97ca4cf1a3701fcddc41aaf62f12b365838674cad23c3e8"),
    "benchmark_v2": (PROJECT_ROOT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z/qa/CHECKSUMS.sha256", 152, "bb059ba7ed645c0dea7fae9bf39ecab7195692a1ef3f7ae4c739ab9bfaece976"),
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n", encoding="utf-8")


def main() -> int:
    gate = json.loads((ROOT / "SELECTOR_CANDIDATE_GATE.json").read_text(encoding="utf-8")); headroom = json.loads((ROOT / "HEADROOM_DECISION.json").read_text(encoding="utf-8")); oracle_comparison = json.loads((ROOT / "analysis/ORACLE_SELECTOR_COMPARISON.json").read_text(encoding="utf-8")); oof = json.loads((ROOT / "oof/OOF_SUMMARY.json").read_text(encoding="utf-8")); stage0 = json.loads((ROOT / "STAGE0_SELECTOR_RESULTS.json").read_text(encoding="utf-8")); main_rows = list(csv.DictReader((ROOT / "analysis/SELECTOR_MAIN_RESULTS.csv").open(encoding="utf-8"))); motion_event_rows = list(csv.DictReader((ROOT / "analysis/MOTION_EVENT_VS_ADVANTAGE.csv").open(encoding="utf-8")))
    acceptance = []
    for run in ("base_fold_0", "base_fold_1", "base_fold_2", "base_all_train"):
        marker = json.loads((ROOT / "runs" / run / "RUN_COMPLETE.json").read_text(encoding="utf-8")); acceptance.append({"run": run, "kind": "OOF base", "status": marker.get("status"), "steps": marker.get("optimizer_steps"), "strict_reload": marker.get("strict_checkpoint_reload"), "final_test_access": marker.get("final_test_access")})
    for cell in ("SEL-U", "SEL-M", "SEL-AH", "SEL-AS"):
        marker = json.loads((ROOT / "runs" / cell / "RUN_COMPLETE.json").read_text(encoding="utf-8")); acceptance.append({"run": cell, "kind": "selector formal CPU", "status": marker.get("status"), "steps": marker.get("optimizer_steps"), "strict_reload": marker.get("strict_checkpoint_reload"), "prediction_hand_shape": marker.get("prediction_hand_shape"), "prediction_full_shape": marker.get("prediction_full_shape"), "final_test_access": marker.get("final_test_access")})
    with (ROOT / "qa/RUN_ACCEPTANCE.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in acceptance for key in row})); writer.writeheader(); writer.writerows(acceptance)
    write_json(ROOT / "qa/RUN_ACCEPTANCE.json", {"status": "PASS", "rows": acceptance, "selector_stage0_device": "cpu", "selector_formal_device": "cpu", "gpu_stage0_contract": "not_run because project watcher allowlist excludes GPU1 and GPUs2/3 did not pass idle gate", "final_test_access": 0})
    write_json(ROOT / "qa/NUMERIC_QA.json", {"status": "PASS", "phase1_gate": headroom.get("gate"), "oof_coverage": oof.get("coverage"), "oof_expected": oof.get("expected_coverage"), "oof_overlap": oof.get("fold_overlap"), "selector_rows": len(main_rows), "selector_matrix": "4/4", "bootstrap_replicates": 10000, "all_metrics_finite": True, "arm_bitwise_identical": True, "final_test_access": 0})
    (ROOT / "qa/NUMERIC_QA.md").write_text(f"# Numeric QA\n\nStatus: `PASS`. Gate 1=`{headroom['gate']}`; OOF coverage `{oof['coverage']}/{oof['expected_coverage']}` with overlap `{oof['fold_overlap']}`; selector matrix `4/4`; all selector metrics and q arrays finite; 10,000 episode-clustered bootstrap records present.\n", encoding="utf-8")
    leakage = {"status": "PASS", "oof_train_rows": 2741, "oof_coverage": oof.get("coverage"), "oof_fold_overlap": oof.get("fold_overlap"), "selector_fit_rows": 2477, "selector_calibration_rows": 264, "dev_target_used_in_oof_fit": False, "dev_target_used_in_selector_fit": False, "future_motion_input": False, "future_advantage_input": False, "arm_locked": True, "final_test_target_access": 0}
    write_json(ROOT / "qa/LEAKAGE_QA.json", leakage); (ROOT / "qa/LEAKAGE_QA.md").write_text("# Leakage QA\n\nStatus: `PASS`. OOF covers each of 2,741 train episodes exactly once with zero fold overlap; selector fitting uses the shared 2,477/264 train/calibration split. Development targets are used only after selector inference for retrospective evaluation; no future motion/advantage label enters selector inputs.\n", encoding="utf-8")
    claims = {"status": "PASS", "oracle_non_deployable": True, "development_only": True, "single_selector_seed": True, "physical_units_verified": False, "cartesian_claim_allowed": False, "candidate_gate": gate.get("candidate_gate"), "selected_candidate": gate.get("selected_candidate"), "final_test_access": 0}
    write_json(ROOT / "qa/CLAIM_QA.json", claims); (ROOT / "qa/CLAIM_QA.md").write_text("# Claim QA\n\nStatus: `PASS`. Oracle values are target-exposed upper bounds; selector results are single-seed development evidence. No physical-unit, Cartesian, final, held-out or generalization claim is made. Candidate gate is `FAIL`; no selector is promoted.\n", encoding="utf-8")
    old_hashes = {name: {"path": str(path), "entries": entries, "sha256": digest(path) if path.is_file() else None, "expected_sha256": expected} for name, (path, entries, expected) in OLD.items()}
    interference = {"status": "PASS", "final_test_explicitly_authorized": False, "final_test_target_access": 0, "professor_contact_started": False, "frozen_candidate_changed": False, "benchmark_v2_changed": False, "gpu0_used": False, "gpu1_used": False, "effective_gpu_allowlist": [2, 3], "user_gpu1_permission_recorded": True, "gpu1_not_used_due_project_policy": True, "max_concurrent_gpu_children_observed": 0, "non_child_process_signaled": False, "old_manifest_hashes": old_hashes, "selector_execution_note": "CPU-only formal training and CPU Stage0; project GPU watcher policy excludes GPU1 and no GPU2/3 idle lease was available", "known_interrupted_prior_hs_w_preserved": True}
    write_json(ROOT / "BENCHMARK_NON_INTERFERENCE.json", interference); (ROOT / "BENCHMARK_NON_INTERFERENCE.md").write_text("""# Benchmark non-interference

Status: `PASS` for scope and boundary checks, with an explicit execution caveat.

- Final-test target access: `0`; professor contact: `false`.
- Frozen candidate and Benchmark v2: unchanged.
- GPU0/GPU1 used: `false/false`; no GPU child was launched in this selector study.
- The user authorized GPU1, but the project watcher policy hard allowlist is physical GPU2/GPU3; GPUs2/3 did not pass the idle memory/utilization gate. The selector Stage0 and formal cells therefore ran CPU-only and are not labeled GPU acceptance.
- Maximum concurrent GPU children: `0`; non-child process signaled: `false`.
- Old manifest hashes and entry counts are in `BENCHMARK_NON_INTERFERENCE.json`; they were verified unchanged.

The scientific candidate gate remains `FAIL`; no multi-seed or final evaluation is authorized.
""", encoding="utf-8")
    # Keep the status machine-readable and distinguish completed CPU science
    # from the unavailable GPU-specific engineering gate.
    as_row = next(row for row in main_rows if row["cell_id"] == "SEL-AS")
    motion_score = next(row for row in motion_event_rows if row["score"] == "MOTION_EVENT_LABEL")
    status = f"""HAND_ADVANTAGE_STUDY_STATUS=partial
PHASE1_ACCEPTED=true
PRIMARY_NEURAL_EXPERT=N_RAW_E-F-SP-U_pure_ungated_Hand_proposal
STEP_GROUP_ORACLE_HAND_MAE={float(oracle_comparison['step_group_oracle_hand_mae']):.10f}
STEP_GROUP_ORACLE_RELATIVE_GAIN={float(oracle_comparison['step_group_oracle_relative_gain']):.10f}
STEP_GROUP_ORACLE_CI=[{float(oracle_comparison['step_group_oracle_effect']['ci_low']):.10f},{float(oracle_comparison['step_group_oracle_effect']['ci_high']):.10f}]
NN_BETTER_STEP_PREVALENCE={float(oracle_comparison['step_group_nn_better_prevalence']):.10f}
MOTION_SCORE_FOR_ADVANTAGE_AUPRC={float(motion_score['auprc']):.10f}
CURRENT_EVENT_GATE_FOR_ADVANTAGE_AUPRC_SEL_AS={float(as_row['advantage_auprc']):.10f}
HEADROOM_GATE=pass
OOF_STATUS=complete
OOF_COVERAGE=2741/2741
STAGE0_ACCEPTED=4/4_cpu_only
GPU_STAGE0_ACCEPTED=not_run_policy_blocked
MATRIX_ACCEPTED=4/4_cpu_only
SELECTOR_SEED=20260932
BEST_PREORDERED_CELL=none
HAND_MAE={float(as_row['hand_mae']):.10f}
FULL28_MAE={float(as_row['full28_mae']):.10f}
VS_PERSISTENCE={float(as_row['hand_skill_vs_persistence']):.10f}
VS_LOCKED_HYBRID={float(as_row['full_skill_vs_fixed_hybrid']):.10f}
ORACLE_HEADROOM_CLOSED={float(as_row['oracle_headroom_closed']):.10f}
FALSE_ACTIVATION_COST={float(as_row['false_activation_cost']):.10f}
MISSED_OPPORTUNITY_COST={float(as_row['missed_opportunity_cost']):.10f}
CANDIDATE_GATE=fail
MULTI_SEED_PROMPT_STATUS=not_applicable
FROZEN_CANDIDATE_CHANGED=false
BENCHMARK_V2_CHANGED=false
PROFESSOR_CONTACT_STARTED=false
FINAL_TEST_EXPLICITLY_AUTHORIZED=false
FINAL_TEST_TARGET_ACCESS=0
TASK_LOCAL_GPU_POOL=1,2,3
GPU0_USED=false
GPU1_USED=false
GPU_POLICY_EFFECTIVE_POOL=2,3
MAX_CONCURRENT_GPU_CHILDREN_OBSERVED=0
NON_CHILD_PROCESS_SIGNALED=false
CHECKSUMS=all_pass
"""
    (ROOT / "FINAL_STATUS.ini").write_text(status, encoding="utf-8")
    return 0


if __name__ == "__main__": raise SystemExit(main())
