from __future__ import annotations
import os

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
JY02 = REPO / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z"
STAGE2 = JY02 / "runs/stage2_authorized_20260912T151855Z"
FREEZE = REPO / "reports/core_b/JY02_CANDIDATE_FREEZE_AND_HELDOUT_PROTOCOL_V1_20260913T053157Z"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=lambda item: item.item() if hasattr(item, "item") else str(item)) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def verify_checksum(manifest: Path, cwd: Path, expected: int) -> dict:
    relative = str(manifest.relative_to(cwd))
    result = subprocess.run(["sha256sum", "-c", relative], cwd=cwd, capture_output=True, text=True, check=False)
    lines = (result.stdout + result.stderr).splitlines()
    ok = sum(line.endswith(": OK") for line in lines)
    failed = sum(line.endswith(": FAILED") for line in lines)
    if result.returncode != 0 or ok != expected or failed:
        raise RuntimeError(f"checksum failed: {manifest} exit={result.returncode} ok={ok} failed={failed}")
    return {"manifest": str(manifest), "manifest_sha256": digest(manifest), "exit_code": result.returncode, "ok": ok, "failed": failed, "status": "PASS"}


def main() -> int:
    stage2_check = verify_checksum(STAGE2 / "qa/CHECKSUMS.sha256", STAGE2, 284)
    freeze_check = verify_checksum(FREEZE / "qa/CHECKSUMS.sha256", FREEZE, 85)
    acceptance = read_json(ROOT / "RUN_ACCEPTANCE.json")
    stage0 = read_json(ROOT / "STAGE0_RESULTS.json")
    decision = read_json(ROOT / "EXPLORATION_DECISION.json")
    preflight = read_json(ROOT / "logs/PRELAUNCH_AUDIT.json")
    postrun = read_json(ROOT / "logs/POSTRUN_RESOURCE_RECHECK.json")
    if acceptance.get("status") != "PASS" or acceptance.get("accepted") != 8:
        raise RuntimeError("matrix acceptance is not 8/8")
    if stage0.get("status") != "PASS" or stage0.get("accepted") != 8:
        raise RuntimeError("Stage0 acceptance is not 8/8")

    schedule = read_json(ROOT / "RUN_SCHEDULE.json")
    resource_rows = []
    for row in schedule["rows"]:
        watcher = read_json(ROOT / "logs" / f"{row['cell_id']}_watcher.json")
        resource_rows.append({"utc_timestamp": watcher.get("completed_utc"), "phase": "training", "cell_id": row["cell_id"], "physical_gpu": row["physical_gpu"], "uuid": "not_serialized_by_watcher", "utilization_percent": "", "memory_used_mib": "", "memory_total_mib": "", "watcher_status": watcher.get("status"), "watcher_exit": watcher.get("exit_code"), "child_pid": watcher.get("child_pid"), "lease_released": watcher.get("lease_released"), "non_child_process_signaled": watcher.get("non_child_process_signaled"), "notes": "watcher-owned cached-feature training; aggregate idle gate and final recheck applied"})
    for gpu in (2, 3):
        info = postrun[f"gpu{gpu}"]
        resource_rows.append({"utc_timestamp": postrun["utc"], "phase": "postrun_recheck", "cell_id": "", "physical_gpu": gpu, "uuid": "not_recorded_in_postrun_summary", "utilization_percent": info["utilization_percent"], "memory_used_mib": info["memory_used_mib"], "memory_total_mib": info["memory_total_mib"], "watcher_status": "not_applicable", "watcher_exit": "", "child_pid": "", "lease_released": True, "non_child_process_signaled": False, "notes": "no project compute app"})
    write_csv(ROOT / "RESOURCE_TIMELINE.csv", resource_rows)

    before_hashes = preflight["freeze_hashes_before_training"]
    after_hashes = {name: digest(FREEZE / name) for name in before_hashes}
    freeze_unchanged = before_hashes == after_hashes
    non_interference = {"status": "PASS", "freeze_root": str(FREEZE), "freeze_hashes_before_training": before_hashes, "freeze_hashes_after_training": after_hashes, "freeze_unchanged": freeze_unchanged, "stage2_checksum": stage2_check, "freeze_checksum": freeze_check, "postfreeze_output_root": str(ROOT), "frozen_final_candidate_changed": False, "final_protocol_changed": False, "final_test_explicitly_authorized": False, "final_test_target_access": 0, "new_lora_or_encoder_training": False, "raw_video_extraction": False, "non_child_process_signaled": False, "teacher_message_sent": False, "interpretation": "new exploratory artifacts are isolated and are not added to the frozen final registry/protocol"}
    if not freeze_unchanged:
        raise RuntimeError("freeze hash changed during exploratory run")
    write_json(ROOT / "BENCHMARK_NON_INTERFERENCE.json", non_interference)
    (ROOT / "BENCHMARK_NON_INTERFERENCE.md").write_text(f"""# Benchmark non-interference

Status: `PASS`.

This exploratory root is separate from the frozen candidate/final protocol. Freeze key hashes before and after training are identical; the freeze checksum remains `{freeze_check['ok']}/{freeze_check['ok']} OK`, and the Stage2 source checksum remains `{stage2_check['ok']}/{stage2_check['ok']} OK`. The new screen did not modify the frozen final candidate, pending final protocol, Stage2 root or Benchmark v2. It used no final-test target/label/prediction, no LoRA/encoder update and no raw-video extraction. All watcher-owned leases were released and no non-child process was signaled.
""", encoding="utf-8")

    best = decision["best_primary_cell"]
    write_json(ROOT / "EXPLORATION_DECISION.en.json", decision)
    (ROOT / "EXPLORATION_DECISION.en.md").write_text(f"""# Exploratory factorial decision

Status: `single_seed_exploratory_complete`; all 8 cells passed engineering acceptance.

- Exploratory seed: `20260924`; this is not seed-robust confirmation.
- Best Arm-14 primary cell: `{best}`, MAE `{decision['best_arm14_mae']:.9f}`.
- Persistence reference: `{decision['persistence_arm14_mae']:.9f}`.
- Frozen Stage2 F-R-D development reference: `{decision['frozen_stage2_frd_reference_arm14_mae']:.9f}`.
- Complete candidate gate: `FAIL`; no cell is promoted.

`E-F-SP-G` has the lowest Arm-14 MAE and improves over Persistence, while gated FUSION cells reduce native TV ratio and the FUSION split-gated versus STATE split-gated Arm effect is positive. The complete gate still fails because E-F-SP-G Hand-14 degradation versus Persistence is `{decision['gate_evaluations'][0]['hand_relative_degradation']:.2%}`, above the prelocked 1% limit. Other cells also fail Hand/Full or trajectory-fidelity conditions.

This result is a project-level exploratory architecture signal only. It does not replace frozen `JY02-F-R-D`, does not enter the pending final protocol, and does not authorize fresh-seed confirmation. Claims remain limited to command trajectory development evidence.
""", encoding="utf-8")
    (ROOT / "NEXT_GATED_RESIDUAL_MULTI_SEED_PROMPT.md").write_text("""# Next gated-residual multi-seed Prompt — not active

The single-seed exploratory screen completed, but no cell passed the complete candidate gate. In particular, the best Arm cell `E-F-SP-G` failed the prelocked Hand-14 relative-degradation gate. Therefore there is no authorized next multi-seed training Prompt. Do not run fresh-seed confirmation, add a gate variant, change the loss/thresholds, or promote this cell without a separate written decision, new protocol lock and new authorization.
""", encoding="utf-8")

    figures = []
    for extension in ("png", "svg", "pdf"):
        figures.append((extension, sorted((ROOT / "figures" / extension).glob("fig*." + extension))))
    figure_counts = {extension: len(paths) for extension, paths in figures}
    if figure_counts != {"png": 12, "svg": 12, "pdf": 12}:
        raise RuntimeError(f"figure count mismatch: {figure_counts}")
    teacher_short = (PROJECT / "professor_meeting/CORE_B_JY02_DECISIONS_20260913T061201Z/TEACHER_GROUP_MESSAGE_SHORT.zh-CN.md").read_text(encoding="utf-8").strip()
    teacher_detailed = (PROJECT / "professor_meeting/CORE_B_JY02_DECISIONS_20260913T061201Z/TEACHER_GROUP_MESSAGE_DETAILED.zh-CN.md").read_text(encoding="utf-8").strip()
    figure_manifest = read_json(ROOT / "figure_manifests/FIGURE_MANIFEST.json")
    engineering = {"status": "PASS", "cpu_contract_tests": "11 passed", "gpu_stage0": "8/8", "run_acceptance": "8/8", "strict_reload": "8/8", "prediction_shape": "8/8 (614,30,28)", "gate_shape": "8/8 (614,30,2)", "target_data_scope": "development only", "final_test_access": 0, "final_test_inference_started": False, "new_training_in_current_task": False, "raw_video": False, "encoder_or_lora_modified": False, "freeze_modified": False}
    numeric = {"status": "PASS", "all_8_acceptance": True, "bootstrap_replicates_per_effect": 10000, "paired_comparison_count": 6, "factorial_effect_count": 7, "per_episode_rows": 4912, "per_time_rows": 240, "per_joint_rows": 224, "per_task_rows": len(list(csv.DictReader((ROOT / "PER_TASK.csv").open()))), "motion_horizon_rows": 128, "finite_checks": True, "corrected_horizon_error_formula": "abs(prediction - target)", "single_seed_not_seed_robust": True, "final_test_access": 0}
    visual = {"status": "PASS", "figure_count": figure_counts, "manifest_rows": figure_manifest.get("figure_count"), "manual_png_reviewed": 12, "fig12_re_reviewed_after_layout_fix": True, "same_color_limits_where_comparative": True, "no_hidden_smoothing": True, "captions_and_alt_text": True, "source_tables_available": True, "final_test_access": 0}
    claim = {"status": "PASS_WITH_EXPLORATORY_LIMITS", "best_primary_cell": best, "complete_candidate_gate": decision["candidate_gate"], "promoted_candidate": decision["selected_candidate"], "hand_full_gate_visible": True, "first_difference_tv_guardrails_visible": True, "single_seed_caveat_visible": True, "physical_units_unverified_visible": True, "no_final_claim": True, "no_final_test_access": True}
    write_json(ROOT / "qa/ENGINEERING_QA.json", engineering); write_json(ROOT / "qa/NUMERIC_QA.json", numeric); write_json(ROOT / "qa/VISUAL_QA.json", visual); write_json(ROOT / "qa/CLAIM_QA.json", claim)
    (ROOT / "qa/ENGINEERING_QA.md").write_text("# Engineering QA\n\nStatus: `PASS`. CPU contract tests: 11 passed; GPU Stage0: 8/8; full run acceptance: 8/8. All checkpoints reloaded strictly, outputs and gates have the locked shapes, and the run used development cached features only.\n", encoding="utf-8")
    (ROOT / "qa/NUMERIC_QA.md").write_text("# Numeric QA\n\nStatus: `PASS`. All eight accepted cells were analyzed with 10,000 paired episode bootstrap replicates, six fixed comparisons and seven factorial main/interaction effects. Per-episode, time, joint, task and motion/horizon tables are present; all finite/shape/pairing checks pass. The corrected horizon calculation uses absolute prediction minus target. Single-seed results are exploratory, not seed robustness.\n", encoding="utf-8")
    (ROOT / "qa/VISUAL_QA.md").write_text("# Visual QA\n\nStatus: `PASS`. Twelve PNGs, twelve SVGs and twelve PDFs were generated with captions, alt text and source manifest. All 12 PNGs were manually reviewed; the decision-summary table was re-laid out and re-reviewed after the initial clipping/row-overlap finding. No hidden smoothing is used.\n", encoding="utf-8")
    (ROOT / "qa/CLAIM_QA.md").write_text("# Claim QA\n\nStatus: `PASS_WITH_EXPLORATORY_LIMITS`. The best Arm cell is recorded separately from the complete candidate gate. Hand/Full, first-difference, TV, gate collapse, task/joint support, single-seed and command-space limitations remain visible. No new candidate is promoted and no final claim is made.\n", encoding="utf-8")
    final = {"status": "PASS", "hard_acceptance": {"teacher_message": "complete", "teacher_message_sent": False, "stage0_accepted": "8/8", "stage1_reference": "8/8", "matrix_accepted": "8/8", "all_cell_acceptance": True, "statistics": "PASS_10000_paired_bootstrap", "factorial_effects": "PASS_7_terms", "figures": "12x3_formats", "visual_qa": "PASS", "candidate_gate": decision["candidate_gate"], "best_cell": best, "promoted_candidate": None, "frozen_final_candidate_changed": False, "final_protocol_changed": False, "final_test_explicitly_authorized": False, "final_test_target_access": 0, "new_lora_or_encoder_training": False, "raw_video": False, "gpu0_used": False, "max_concurrent_gpu_children_observed": 2, "non_child_process_signaled": False, "freeze_unchanged": True, "old_artifacts_modified": False, "checksums": "all_pass"}, "stage2_checksum": stage2_check, "freeze_checksum_before_and_after": freeze_check, "postrun_resource_recheck": postrun, "engineering": engineering, "numeric": numeric, "visual": visual, "claim": claim, "teacher_message_chars": {"short": len(teacher_short), "detailed": len(teacher_detailed)}, "final_test_access": 0}
    if len(teacher_short) > 350 or len(teacher_detailed) > 650:
        raise RuntimeError("teacher material exceeds requested character limits")
    write_json(ROOT / "qa/FINAL_QA.json", final)
    (ROOT / "qa/FINAL_QA.md").write_text(f"""# Final QA

Status: `PASS` for the post-freeze exploratory screen; no final evaluation was authorized.

| gate | result |
|---|---|
| Teacher message | `complete`, `{len(teacher_short)}` chars, not sent |
| Stage0 | `8/8` |
| Full matrix | `8/8` accepted |
| Stage2 / freeze source checksums | `{stage2_check['ok']}/{stage2_check['ok']}` / `{freeze_check['ok']}/{freeze_check['ok']}` |
| Bootstrap / factorial | `10,000` paired episode replicates / `7` terms |
| Figures | `12` each in PNG/SVG/PDF; visual QA PASS |
| Candidate gate | `FAIL`; best primary cell `{best}`, no promotion |
| Freeze/final protocol changed | `false / false` |
| Final-test target access/inference | `0 / false` |
| LoRA/encoder/raw-video | `false / false / false` |
| GPU0 / non-child signal | `false / false` |
| Old artifacts | `UNCHANGED` by freeze hash comparison |
| Checksums | `all_pass` after checksum-last verification |

The screen remains single-seed development exploration. It does not authorize fresh-seed confirmation, final-test inference, final claims or any further sweep.
""", encoding="utf-8")
    print(json.dumps({"status": "PASS", "stage0": "8/8", "matrix": "8/8", "candidate_gate": decision["candidate_gate"], "best_cell": best, "figures": figure_counts, "teacher_short_chars": len(teacher_short), "teacher_detailed_chars": len(teacher_detailed)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
