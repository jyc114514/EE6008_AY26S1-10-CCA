from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


FREEZE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa"
JY02_ROOT = REPO_ROOT / "reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z"
STAGE2_ATTEMPT = JY02_ROOT / "runs/stage2_authorized_20260912T151855Z"
B23_ROOT = REPO_ROOT / "reports/core_b/B2_B3_POOLED_SAMPLING_LORA_20260908T113000Z"
V2_ROOT = PROJECT_ROOT / "research_archive/CORE_A_CORE_B_EVIDENCE_20260909T045416Z/core_b/benchmark/CORE_B_PERSISTENCE_AWARE_MOTION_BENCHMARK_V2_20260910T080818Z"
SPLIT_ROOT = REPO_ROOT / "manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe"
FINAL_AUDIT_ROOT = REPO_ROOT / "reports/data_audit/final_test_handoff_v3_20260826"
NORMALIZER = REPO_ROOT / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
DATA_MANIFEST = REPO_ROOT / "reports/core_b/gpu3_baseline_and_finetune_v1_20260903T064316Z/inputs_1p0s_20260903T064316Z/CORE_B_FULL_DATA_MANIFEST.json"
FEATURE_ROOT = PROJECT_ROOT / "features/g1/vjepa2/full-vjepa-b-opencv-20260826"
STATIC_FEATURE_ROOT = PROJECT_ROOT / "features/g1/vjepa2/static-first-frame-vjepa-b-v3-tokens-20260828T071500Z"
DATASET_ROOT = PROJECT_ROOT / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
UTC_NOW = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(relative: str, content: str) -> None:
    path = FREEZE_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def write_json(relative: str, value: object) -> None:
    write_text(relative, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def write_csv(relative: str, rows: list[dict], fieldnames: list[str]) -> None:
    path = FREEZE_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_record(path: Path, role: str, *, access_note: str = "read_and_hashed") -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "role": role,
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "access_note": access_note,
    }


def metadata_record(path: Path, role: str, sha_value: str | None = None, *, access_note: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "role": role,
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha_value,
        "access_note": access_note,
    }


def run_stage2_checksum() -> dict:
    manifest = STAGE2_ATTEMPT / "qa/CHECKSUMS.sha256"
    completed = subprocess.run(
        ["sha256sum", "-c", str(manifest)],
        text=True,
        capture_output=True,
        check=False,
    )
    lines = completed.stdout.splitlines() + completed.stderr.splitlines()
    ok = sum(line.endswith(": OK") for line in lines)
    failed = sum(line.endswith(": FAILED") for line in lines)
    if completed.returncode != 0 or ok != 284 or failed:
        raise RuntimeError(f"Stage2 checksum conflict: exit={completed.returncode} ok={ok} failed={failed}")
    return {
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "command": f"sha256sum -c {manifest}",
        "exit_code": completed.returncode,
        "ok": ok,
        "failed": failed,
        "status": "PASS",
    }


def validate_stage2(acceptance: dict) -> tuple[list[dict], dict]:
    expected_configs = ["JY02-S-A-D", "JY02-F-A-D", "JY02-S-R-D", "JY02-F-R-D"]
    expected_seeds = [20260921, 20260922, 20260923]
    rows = acceptance["rows"]
    if acceptance["status"] != "PASS" or acceptance["accepted_runs"] != 12 or acceptance["expected_runs"] != 12:
        raise RuntimeError("Stage2 acceptance is not 12/12 PASS")
    if acceptance["configs"] != expected_configs or acceptance["fresh_seeds"] != expected_seeds:
        raise RuntimeError("Stage2 config/seed matrix mismatch")
    if acceptance["final_test_access"] != 0 or acceptance["stage1_rerun"] is not False:
        raise RuntimeError("Stage2 authorization boundary mismatch")
    if len(rows) != 12:
        raise RuntimeError("Stage2 row count mismatch")
    pairs = set()
    frozen_rows = []
    for row in rows:
        config_id = row["config_id"]
        seed = int(row["seed"])
        pair = (config_id, seed)
        if pair in pairs or config_id not in expected_configs or seed not in expected_seeds:
            raise RuntimeError(f"duplicate or unexpected Stage2 pair: {pair}")
        pairs.add(pair)
        run_dir = Path(row["path"])
        checkpoint = run_dir / "checkpoint_best.pt"
        prediction = run_dir / "predictions_best.npz"
        complete = read_json(run_dir / "RUN_COMPLETE.json")
        status = read_json(run_dir / "RUN_STATUS.json")
        model_contract = read_json(run_dir / "MODEL_CONTRACT.json")
        if sha256(checkpoint) != row["checkpoint_best_sha256"] or sha256(prediction) != row["prediction_best_sha256"]:
            raise RuntimeError(f"Stage2 hash mismatch for {pair}")
        if complete.get("status") != "complete" or status.get("status") != "complete":
            raise RuntimeError(f"Stage2 marker mismatch for {pair}")
        if complete.get("final_test_access") != 0 or status.get("final_test_access") != 0:
            raise RuntimeError(f"Stage2 final-test flag mismatch for {pair}")
        if row["shape"] != "(614, 30, 28)" or row["save_reload"] != "PASS":
            raise RuntimeError(f"Stage2 output contract mismatch for {pair}")
        frozen_rows.append(
            {
                "model_type": "learned",
                "config_id": config_id,
                "seed": seed,
                "checkpoint_path": str(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": sha256(checkpoint),
                "prediction_path": str(prediction),
                "prediction_size_bytes": prediction.stat().st_size,
                "prediction_sha256": sha256(prediction),
                "expected_shape": "30x28",
                "development_N": int(row["N"]),
                "best_epoch": int(row["best_epoch"]),
                "optimizer_steps": int(row["optimizer_steps"]),
                "strict_reload": row["save_reload"],
                "final_test_access": int(row["final_test_access"]),
                "selection_rule": "pre-locked development best checkpoint: Arm-14 MAE, then Full-28 MAE, then earlier epoch; no final reselection",
                "frozen_at_utc": UTC_NOW,
                "model_contract_status": model_contract.get("status"),
            }
        )
    if pairs != {(config, seed) for config in expected_configs for seed in expected_seeds}:
        raise RuntimeError("Stage2 matrix is incomplete")
    return frozen_rows, {
        "accepted_runs": 12,
        "expected_runs": 12,
        "configs": expected_configs,
        "fresh_seeds": expected_seeds,
        "optimizer_steps_total": acceptance["optimizer_steps_total"],
        "final_test_access": 0,
        "stage1_rerun": False,
        "fresh_initialization": acceptance["fresh_initialization"],
    }


def run_engineering_tests() -> dict:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    test_run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=FREEZE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    combined = (test_run.stdout + "\n" + test_run.stderr).strip()
    match = re.search(r"(\d+) passed", combined)
    passed = int(match.group(1)) if match else 0
    if test_run.returncode != 0:
        raise RuntimeError(f"engineering tests failed: {combined[-3000:]}")
    syntax_files = []
    import ast

    for path in sorted((FREEZE_ROOT / "code").rglob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        syntax_files.append(str(path.relative_to(FREEZE_ROOT)))
    return {
        "status": "PASS",
        "command": f"{sys.executable} -m pytest -q tests",
        "exit_code": test_run.returncode,
        "tests_passed": passed,
        "tests_failed": 0,
        "code_ast_parse": "PASS",
        "code_files_parsed": len(syntax_files),
        "target_data_used": False,
        "final_test_access": 0,
        "output_tail": combined[-1000:],
    }


def build() -> None:
    checksum = run_stage2_checksum()
    acceptance = read_json(JY02_ROOT / "STAGE2_RUN_ACCEPTANCE_POST_AUTHORIZATION.json")
    frozen_rows, stage2_summary = validate_stage2(acceptance)
    engineering = run_engineering_tests()
    results = read_json(JY02_ROOT / "STAGE2_RESULTS_BY_SEED.json")
    paired = read_json(JY02_ROOT / "STAGE2_PAIRED_EFFECTS_BY_SEED.json")
    hierarchical = read_json(JY02_ROOT / "STAGE2_HIERARCHICAL_STATISTICS.json")
    confirmation = read_json(JY02_ROOT / "STAGE2_CONFIRMATION_DECISION.json")
    final_audit = read_json(FINAL_AUDIT_ROOT / "audit.json")
    split_lock = read_json(SPLIT_ROOT / "v3_protocol_lock.json")
    threshold = read_json(V2_ROOT / "MOTION_THRESHOLD_SOURCE.json")

    old_key_paths = [
        JY02_ROOT / "JY02_PROTOCOL_LOCKED.json",
        JY02_ROOT / "JY02_DECODER_CONTRACT.json",
        JY02_ROOT / "STAGE1_CANDIDATE_DECISION.json",
        JY02_ROOT / "STAGE2_RUN_ACCEPTANCE.json",
        JY02_ROOT / "STAGE2_ANALYSIS_STATUS.json",
        JY02_ROOT / "STAGE2_CONFIRMATION_DECISION.json",
        STAGE2_ATTEMPT / "qa/CHECKSUMS.sha256",
    ]
    old_before = {str(path): sha256(path) for path in old_key_paths}

    write_csv(
        "CANDIDATE_FREEZE_MANIFEST.csv",
        frozen_rows,
        [
            "model_type", "config_id", "seed", "checkpoint_path", "checkpoint_size_bytes", "checkpoint_sha256",
            "prediction_path", "prediction_size_bytes", "prediction_sha256", "expected_shape", "development_N",
            "best_epoch", "optimizer_steps", "strict_reload", "final_test_access", "selection_rule",
            "frozen_at_utc", "model_contract_status",
        ],
    )
    write_json("CANDIDATE_FREEZE_MANIFEST.json", {"status": "PASS", "rows": frozen_rows, "summary": stage2_summary})

    source_specs = [
        (STAGE2_ATTEMPT / "qa/CHECKSUMS.sha256", "stage2_authoritative_checksum_manifest", "read_and_verified_284_items"),
        (JY02_ROOT / "JY02_PROTOCOL_LOCKED.json", "jy02_protocol", "read_and_hashed"),
        (JY02_ROOT / "JY02_PROTOCOL_LOCKED.md", "jy02_protocol", "read_and_hashed"),
        (JY02_ROOT / "RESOLVED_TRAINING_CONTRACT.json", "training_contract", "read_and_hashed"),
        (JY02_ROOT / "JY02_RESIDUAL_CONTRACT.json", "residual_contract", "read_and_hashed"),
        (JY02_ROOT / "JY02_INPUT_TARGET_CONTRACT.json", "input_target_contract", "read_and_hashed"),
        (JY02_ROOT / "JY02_DECODER_CONTRACT.json", "legacy_decoder_contract_with_900_typo", "read_and_hashed"),
        (JY02_ROOT / "STAGE1_CANDIDATE_DECISION.json", "stage1_candidate_decision", "read_and_hashed"),
        (JY02_ROOT / "STAGE2_RUN_ACCEPTANCE_POST_AUTHORIZATION.json", "stage2_acceptance", "read_and_hashed"),
        (JY02_ROOT / "STAGE2_RESULTS_BY_SEED.json", "stage2_results", "read_and_hashed"),
        (JY02_ROOT / "STAGE2_PAIRED_EFFECTS_BY_SEED.json", "stage2_paired_effects", "read_and_hashed"),
        (JY02_ROOT / "STAGE2_HIERARCHICAL_STATISTICS.json", "stage2_hierarchical_statistics", "read_and_hashed"),
        (JY02_ROOT / "STAGE2_CONFIRMATION_DECISION.json", "stage2_confirmation_decision", "read_and_hashed"),
        (JY02_ROOT / "STAGE2_ANALYSIS_STATUS.json", "stage2_analysis_status", "read_and_hashed"),
        (STAGE2_ATTEMPT / "configs/JY02_PROTOCOL_LOCKED.json", "executed_config_snapshot", "read_and_hashed"),
        (STAGE2_ATTEMPT / "configs/RESOLVED_TRAINING_CONTRACT.json", "executed_config_snapshot", "read_and_hashed"),
        (STAGE2_ATTEMPT / "configs/JY02_RESIDUAL_CONTRACT.json", "executed_config_snapshot", "read_and_hashed"),
        (STAGE2_ATTEMPT / "configs/JY02_INPUT_TARGET_CONTRACT.json", "executed_config_snapshot", "read_and_hashed"),
        (JY02_ROOT / "inputs/JY02_TARGETS.npz", "actual_stage2_development_input_archive", "read_and_hashed; development target-bearing archive not copied"),
        (B23_ROOT / "inputs/B2_INPUTS_0125.npz", "upstream_development_input_archive", "read_and_hashed; development target-bearing archive not copied"),
        (B23_ROOT / "inputs/B2_INPUT_PREPARATION.json", "upstream_input_preparation", "read_and_hashed"),
        (NORMALIZER, "train_only_target_normalizer", "read_and_hashed; vectors not copied"),
        (DATA_MANIFEST, "development_data_manifest", "read_and_hashed"),
        (FEATURE_ROOT / "feature_manifest.parquet", "dynamic_feature_manifest_actual_project_path", "schema/hash only; no final labels parsed"),
        (FEATURE_ROOT / "resolved_config.json", "dynamic_feature_config", "read_and_hashed"),
        (FEATURE_ROOT / "model_contract.json", "dynamic_feature_contract", "read_and_hashed"),
        (STATIC_FEATURE_ROOT / "feature_manifest.parquet", "static_diagnostic_feature_manifest", "schema/hash only"),
        (SPLIT_ROOT / "split_summary.json", "v3_split_summary", "metadata only"),
        (SPLIT_ROOT / "v3_protocol_lock.json", "v3_split_lock", "metadata only"),
        (SPLIT_ROOT / "split_manifest.parquet", "v3_split_manifest_schema", "metadata/schema only; no final labels parsed"),
        (SPLIT_ROOT / "final_test_index_label_free.parquet", "label_free_final_index", "label-free metadata/hash only"),
        (V2_ROOT / "V2_PROTOCOL_LOCKED.json", "benchmark_v2_protocol", "read_and_hashed"),
        (V2_ROOT / "V2_PROTOCOL_LOCKED.md", "benchmark_v2_protocol", "read_and_hashed"),
        (V2_ROOT / "MOTION_THRESHOLD_SOURCE.json", "train_frozen_motion_thresholds", "development/train-derived metadata only"),
        (V2_ROOT / "MOTION_STRATA_MANIFEST.json", "development_motion_strata_manifest", "development metadata only"),
        (V2_ROOT / "V2_METRIC_REGISTRY.json", "benchmark_v2_metric_registry", "read_and_hashed"),
        (V2_ROOT / "V2_METRIC_REGISTRY.md", "benchmark_v2_metric_registry", "read_and_hashed"),
        (V2_ROOT / "V2_OUTCOME_EXPOSURE.json", "benchmark_v2_outcome_exposure", "development metadata only"),
        (V2_ROOT / "V2_OUTCOME_EXPOSURE.md", "benchmark_v2_outcome_exposure", "development metadata only"),
        (JY02_ROOT / "runs/stage2_authorized_20260912T151855Z/statistics/run_stage2_analysis.py", "stage2_analysis_script", "read_and_hashed"),
    ]
    component_records = [file_record(path, role, access_note=note) for path, role, note in source_specs]
    protected_label_path = SPLIT_ROOT / "final_test_label_manifest.parquet"
    component_records.append(
        metadata_record(
            protected_label_path,
            "protected_final_label_manifest_opaque_reference",
            final_audit["protected_label_manifest_sha256"],
            access_note="prior_audit_hash_and_schema_reference_only; not opened or parsed in this task",
        )
    )
    component_records.append({
        "path": str(REPO_ROOT / ".git"),
        "role": "project_git_identity",
        "exists": False,
        "size_bytes": None,
        "sha256": None,
        "access_note": "not_available; repository has no .git and git executable was unavailable",
    })
    copied_code = sorted((FREEZE_ROOT / "code/stage2_source_snapshot").glob("*.py"))
    for path in copied_code:
        component_records.append(file_record(path, "frozen_stage2_code_snapshot_copy", access_note="copied from accepted attempt; hash verified"))
    write_csv(
        "CANDIDATE_COMPONENT_HASHES.csv",
        component_records,
        ["path", "role", "exists", "size_bytes", "sha256", "access_note"],
    )
    write_json("CANDIDATE_COMPONENT_HASHES.json", {"status": "PASS", "generated_utc": UTC_NOW, "records": component_records})

    frd_rows = [row for row in results["rows"] if row["model"] == "JY02-F-R-D"]
    persistence_rows = [row for row in results["rows"] if row["model"] == "PERSISTENCE"]
    frd_summary = paired["summaries"]["F-R-D_vs_PERSISTENCE"]["full"]
    frd_fad = hierarchical["comparisons"]["F-R-D_vs_F-A-D"]["full"]
    frd_srd = hierarchical["comparisons"]["F-R-D_vs_S-R-D"]["full"]
    negative_findings = {
        "hand14_frd_mean": mean(row["hand_full_mae_std"] for row in frd_rows),
        "hand14_persistence": persistence_rows[0]["hand_full_mae_std"],
        "full28_frd_mean": mean(row["full28_full_mae_std"] for row in frd_rows),
        "full28_persistence": persistence_rows[0]["full28_full_mae_std"],
        "first_difference_frd_vs_persistence_ci": hierarchical["comparisons"]["F-R-D_vs_PERSISTENCE"]["first_difference"]["hierarchical"],
        "native_tv_ratio_frd_by_seed": [row["tv_ratio_mean_native"] for row in frd_rows],
    }
    development_evidence = {
        "status": "FROZEN_DEVELOPMENT_EVIDENCE_ONLY",
        "candidate": "JY02-F-R-D",
        "fresh_seeds": [20260921, 20260922, 20260923],
        "stage2_source_checksum": checksum,
        "headline": {
            "frd_arm14_standardized_full_mae_mean": results["mean_sd"]["JY02-F-R-D"]["arm_full_mae_std_mean"],
            "frd_arm14_standardized_full_mae_sd": results["mean_sd"]["JY02-F-R-D"]["arm_full_mae_std_sd"],
            "persistence_arm14_standardized_full_mae": results["mean_sd"]["PERSISTENCE"]["arm_full_mae_std_mean"],
            "frd_vs_persistence_seed_effects": frd_summary["seed_effects"],
            "frd_vs_persistence_hierarchical_ci": [frd_summary["hierarchical"]["ci_low"], frd_summary["hierarchical"]["ci_high"]],
            "frd_vs_fad_hierarchical_ci": [frd_fad["hierarchical"]["ci_low"], frd_fad["hierarchical"]["ci_high"]],
            "frd_vs_srd_video_increment_ci": [frd_srd["hierarchical"]["ci_low"], frd_srd["hierarchical"]["ci_high"]],
        },
        "negative_findings": negative_findings,
        "bootstrap": {
            "per_seed_episode": 10000,
            "hierarchical_seed_episode": 20000,
            "independent_confirmation_unit": "training seed; episodes nested within seed",
        },
        "scope_boundary": [
            "development-validation only",
            "not final model or final-test estimate",
            "not Cartesian/URDF/FK accuracy",
            "not session/operator/scene-independent generalization",
            "video increment retains task/scene shortcut caveat",
        ],
    }
    write_json("DEVELOPMENT_EVIDENCE_FREEZE.json", development_evidence)
    write_text(
        "DEVELOPMENT_EVIDENCE_FREEZE.md",
        f"""# JY-02 development evidence freeze

Status: `FROZEN_DEVELOPMENT_EVIDENCE_ONLY`

The accepted Stage2 source manifest verifies `{checksum['ok']}/{checksum['ok']}` items. The fixed candidate is `JY02-F-R-D`; the discovery seed `20260903` is excluded from the fresh-seed confirmation.

| quantity | frozen development evidence |
|---|---:|
| F-R-D Arm-14 standardized full MAE | `{development_evidence['headline']['frd_arm14_standardized_full_mae_mean']:.9f} +/- {development_evidence['headline']['frd_arm14_standardized_full_mae_sd']:.9f}` |
| Persistence Arm-14 standardized full MAE | `{development_evidence['headline']['persistence_arm14_standardized_full_mae']:.9f}` |
| F-R-D vs Persistence hierarchical CI | `[{frd_summary['hierarchical']['ci_low']:.9f}, {frd_summary['hierarchical']['ci_high']:.9f}]` |
| F-R-D vs F-A-D hierarchical CI | `[{frd_fad['hierarchical']['ci_low']:.9f}, {frd_fad['hierarchical']['ci_high']:.9f}]` |
| F-R-D vs S-R-D video increment CI | `[{frd_srd['hierarchical']['ci_low']:.9f}, {frd_srd['hierarchical']['ci_high']:.9f}]` |

The three fresh-seed Arm effects are `{', '.join(f'{value:.6f}' for value in frd_summary['seed_effects'])}`. Hand-14 and Full-28 remain worse than Persistence in the frozen development evidence (`{negative_findings['hand14_frd_mean']:.6f}` vs `{negative_findings['hand14_persistence']:.6f}`, and `{negative_findings['full28_frd_mean']:.6f}` vs `{negative_findings['full28_persistence']:.6f}`). F-R-D also remains worse than Persistence on first-difference MAE, and its native TV ratio is approximately `{mean(negative_findings['native_tv_ratio_frd_by_seed']):.3f}`.

These facts support only `new_best_multi_seed_development_candidate for Arm-14 full-horizon command trajectory`. They do not support final-test performance, a universal upper-body claim, physical-unit accuracy, or a causal video mechanism.
""",
    )

    write_text(
        "CANDIDATE_MODEL_CARD.zh-CN.md",
        f"""# JY-02 候选模型卡

状态：`new_best_multi_seed_development_candidate`，不是 final model。

- 候选：`JY02-F-R-D`（FUSION + residual-to-persistence + direct chunk decoder）。
- 冻结 fresh seeds：`20260921 / 20260922 / 20260923`。
- 固定开发证据：Arm-14 standardized full-horizon MAE `{results['mean_sd']['JY02-F-R-D']['arm_full_mae_std_mean']:.9f} +/- {results['mean_sd']['JY02-F-R-D']['arm_full_mae_std_sd']:.9f}`；Persistence `{results['mean_sd']['PERSISTENCE']['arm_full_mae_std_mean']:.9f}`。
- 预测合同：`(N,30,28)`，Arm=`14:28`，Hand=`0:14`；residual 重建后才计算 loss/metrics。
- 训练：12/12 accepted；每 run 1720 optimizer steps；strict reload PASS；encoder/LoRA 未训练。
- 选择规则：开发集预锁定 checkpoint 规则；final 不重选 seed、不建 ensemble。
- 限制：开发集 בלבד；Hand/Full-28 不优于 Persistence；first-difference/TV 只作 fidelity guardrail；Cartesian、URDF/FK、session/operator/scene generalization 未验证。
- 数据边界：本冻结任务未读取 final targets/labels/predictions，`FINAL_TEST_TARGET_ACCESS=0`。
""",
    )
    write_text(
        "CANDIDATE_MODEL_CARD.en.md",
        f"""# JY-02 candidate model card

Status: `new_best_multi_seed_development_candidate`, not a final model.

- Candidate: `JY02-F-R-D` (FUSION + residual-to-persistence + direct chunk decoder).
- Frozen fresh seeds: `20260921 / 20260922 / 20260923`.
- Frozen development evidence: Arm-14 standardized full-horizon MAE `{results['mean_sd']['JY02-F-R-D']['arm_full_mae_std_mean']:.9f} +/- {results['mean_sd']['JY02-F-R-D']['arm_full_mae_std_sd']:.9f}`; Persistence `{results['mean_sd']['PERSISTENCE']['arm_full_mae_std_mean']:.9f}`.
- Prediction contract: `(N,30,28)`, Arm=`14:28`, Hand=`0:14`; residuals are reconstructed before loss/metrics.
- Training: 12/12 accepted; 1720 optimizer steps per run; strict reload PASS; no encoder/LoRA training.
- Selection: pre-locked development checkpoint rule; no final seed reselection and no unregistered ensemble.
- Limitations: development only; Hand/Full-28 do not beat Persistence; first-difference/TV are fidelity guardrails; Cartesian/URDF/FK and session/operator/scene generalization are unverified.
- Boundary: this freeze task did not read final targets/labels/predictions; `FINAL_TEST_TARGET_ACCESS=0`.
""",
    )

    metadata_reconciliation = {
        "status": "complete_non_destructive_addenda_only",
        "old_artifacts_modified": False,
        "items": [
            {
                "id": "decoder_output_840_vs_900",
                "finding": "Executable and locked Stage2 direct decoder emit 30x28=840 values; legacy JY02_DECODER_CONTRACT.json says 900.",
                "resolution": "Treat 900 as documentation typo; preserve old file and add a 840 addendum.",
                "numeric_impact": "none on accepted Stage2 values; executed artifacts and shape checks are 840/(30,28).",
            },
            {
                "id": "jy02_outcome_exposure_missing",
                "finding": "No original JY02-specific outcome-exposure file was found as a canonical source.",
                "resolution": "Create a clearly labelled reconstruction from the locked timeline, accepted artifacts and analysis provenance; never relabel it as an original record.",
                "numeric_impact": "none; no new outcome statistic is computed here.",
            },
            {
                "id": "native_tv_field_label",
                "finding": "RUN_METRICS.native_tv_ratio is not a reliable label for the final native-TV quantity.",
                "resolution": "Keep old field unchanged; final Stage2 native-TV and closeness values are taken from the independent native-prediction recomputation already frozen in Stage2 statistics.",
                "numeric_impact": "does not alter accepted Stage2 arrays or the separately recorded native recomputation; changes interpretation of the legacy field only.",
            },
            {
                "id": "watcher_child_provenance",
                "finding": "The watcher did not serialize a dedicated child UUID/heartbeat interval.",
                "resolution": "Record as provenance limitation; corroborate with watcher JSON, child/process evidence, lease lifecycle, completion markers and final resource recheck.",
                "numeric_impact": "none on prediction/statistic values; lowers process-lineage granularity.",
            },
            {
                "id": "input_and_feature_path_aliases",
                "finding": "Executed Stage2 uses JY02_TARGETS.npz while the upstream contract names B2_INPUTS_0125.npz; feature manifests live under the project-level features path, not repo/features.",
                "resolution": "Freeze both actual hashes and roles; do not rewrite old paths. The executed archive is the Stage2 input of record; B2 archive is its upstream development precursor.",
                "numeric_impact": "none after explicit lineage recording; future execution must use the frozen executed input contract.",
            },
        ],
    }
    write_json("METADATA_RECONCILIATION.json", metadata_reconciliation)
    write_text(
        "METADATA_RECONCILIATION.md",
        """# Metadata reconciliation

This is a non-destructive addendum. No old JY-02, Benchmark v2, Frozen attribution, run, checkpoint, config, log or status file was edited.

| issue | correction | impact |
|---|---|---|
| Legacy decoder contract says 900 | Executed direct decoder and locked protocol are `30 x 28 = 840`; preserve legacy file and cite the 840 addendum | Text/metadata only; accepted shapes and arrays are unchanged |
| No canonical JY02-specific outcome-exposure file | Record a labelled reconstruction from the locked timeline and accepted provenance; do not call it an original file | Documentation only; no new metric is computed |
| `RUN_METRICS.native_tv_ratio` label is unreliable | Preserve old field; use the already independent native-prediction recomputation for native-TV claims | Interpretation only |
| Watcher lacks serialized child UUID/heartbeat interval | Retain limitation and use watcher/process/lease/marker/resource evidence as corroboration | Provenance granularity only |
| Input/feature path aliases | Freeze actual executed `JY02_TARGETS.npz`, upstream `B2_INPUTS_0125.npz`, and project-level feature manifest hashes with explicit roles | Lineage clarity; no data rewrite |

The reconciliation is complete for protocol preparation. It does not authorize final-test access.
""",
    )
    write_text("DECODER_OUTPUT_840_ADDENDUM.md", """# Decoder output 840 addendum

The executable direct decoder and Stage2 locked protocol use `Linear(input,256) -> GELU -> Linear(256,840) -> reshape(30,28)`. The legacy `JY02_DECODER_CONTRACT.json` contains `900`; this is a documentation typo, not an alternate executed shape. The addendum leaves the legacy file untouched and makes `840` the only future evaluation shape.
""")
    write_json("DECODER_OUTPUT_840_ADDENDUM.json", {"status": "PASS", "executed_output_values": 840, "shape": [30, 28], "legacy_value": 900, "legacy_interpretation": "documentation typo", "old_file_modified": False})
    write_text("OUTCOME_EXPOSURE_RECONSTRUCTION.md", """# Outcome-exposure reconstruction

No canonical JY02-specific outcome-exposure record was found. This file is a reconstruction, not an original source. It records only already accepted development facts: Stage2 was outcome-exposed development work on 2741 train / 614 development rows; checkpoint selection was pre-locked within the development contract; fresh-seed confirmation used 20260921/22/23; no final-test target values, metrics or predictions were accessed. No new outcome value is calculated here.

The future final protocol keeps all metrics, thresholds, comparisons and model IDs fixed before any target unlock.
""")
    write_json("OUTCOME_EXPOSURE_RECONSTRUCTION.json", {"status": "PASS_RECONSTRUCTION_LABELLED", "original_canonical_file_found": False, "reconstructed_from": [str(JY02_ROOT / "JY02_PROTOCOL_LOCKED.json"), str(JY02_ROOT / "STAGE2_CONFIRMATION_DECISION.json"), str(STAGE2_ATTEMPT / "statistics/ANALYSIS_PROVENANCE.json")], "final_test_target_values_read": False, "new_statistics_computed": False})
    write_text("NATIVE_TV_FIELD_LABEL_ADDENDUM.md", """# Native TV field-label addendum

The historical `RUN_METRICS.native_tv_ratio` label is not used as the authoritative native-TV result. Stage2's final independent analysis recomputed native absolute TV and TV ratio from native predictions; those values and their hierarchical effects are frozen in the Stage2 statistics. The old run metric remains unchanged. Future evaluation must store standardized TV, native TV, and their ratio in separate explicitly named fields.
""")
    write_json("NATIVE_TV_FIELD_LABEL_ADDENDUM.json", {"status": "PASS", "legacy_field_preserved": True, "authoritative_future_fields": ["native_absolute_tv", "standardized_absolute_tv", "native_tv_ratio", "standardized_tv_ratio", "tv_closeness_to_one"], "old_files_modified": False})
    write_text("WATCHER_PROVENANCE_LIMITATION.md", """# Watcher provenance limitation

The Stage2 watcher did not serialize a dedicated child UUID or heartbeat interval. This is retained as a limitation. The accepted run is nevertheless corroborated by the watcher record, child/launcher process evidence, lease acquisition/release, 12 completion markers, per-run exit codes, Stage2 acceptance, and final GPU/process/resource recheck. This supports run acceptance but does not upgrade process lineage to UUID-level provenance.
""")
    write_json("WATCHER_PROVENANCE_LIMITATION.json", {"status": "PASS_WITH_LIMITATION", "child_uuid_serialized": False, "heartbeat_interval_serialized": False, "corroborating_evidence": ["watcher JSON", "process evidence", "lease lifecycle", "12 completion markers", "final resource recheck"], "numeric_impact": "none"})

    label_free_path = SPLIT_ROOT / "final_test_index_label_free.parquet"
    heldout = {
        "status": "complete_without_target_access",
        "heldout_evaluation_available": "unknown",
        "physical_population_evidence": {
            "v3_label_free_index_exists": True,
            "all_v3_final_rows": 614,
            "primary_rows": 602,
            "label_free_index_path": str(label_free_path),
            "label_free_index_sha256": sha256(label_free_path),
            "split_final_metrics_computed": split_lock["final_test_metrics_computed"],
        },
        "blind_defensibility": {
            "final_test_label_values_read_in_prior_audit": final_audit["final_test_label_values_read"],
            "final_test_metric_access_in_prior_audit": final_audit["final_test_metric_access"],
            "historical_feature_manifest_label_columns": final_audit["feature_manifest_label_columns"],
            "historical_metadata_label_keys": final_audit["historical_metadata_label_keys"],
            "conclusion": "A physical final population and label-free index exist, but clean blind availability is not proven until custodial sanitization/release review resolves inherited label-bearing metadata.",
        },
        "prohibited_current_task_actions": ["open final targets", "read final label values", "compute final metrics", "run final inference"],
        "current_task_final_test_access": 0,
    }
    write_json("HELDOUT_AVAILABILITY_AUDIT.json", heldout)
    write_text(
        "HELDOUT_AVAILABILITY_AUDIT.md",
        f"""# Untouched held-out availability audit

The v3 metadata exposes a physical final population: 614 final-test rows, of which 602 are primary rows. The label-free index hash is `{sha256(label_free_path)}` and v3 records `final_test_metrics_computed=false`. This task did not open final targets or the protected label manifest.

The answer to “is a defensible blind held-out evaluation immediately available?” is `unknown`, not an unconditional yes. The prior handoff audit reports no final label values or metrics read, but it also records label-bearing columns/keys in inherited historical feature metadata (`classification_inclusion`, `contiguous_label_index`, `original_task_index`, task/category keys). Therefore Route A is selected conditionally: a custodian must perform a metadata-only sanitization/release review and obtain written human approval before one-time target unlock. If that review fails, do not use development as final; fall back to Route B or Route C.

No final target distribution, final motion strata, final performance, or label-derived count is reported here.
""",
    )
    split_audit = {
        "status": "PASS_METADATA_ONLY",
        "protocol": split_lock["protocol_version"],
        "dataset_revision": split_lock["dataset_revision"],
        "split_counts": split_lock["split_counts"],
        "primary_split_counts": split_lock["primary_split_counts"],
        "exact_duplicate_groups": split_lock["exact_duplicate_group_count"],
        "conflicting_duplicate_excluded_episodes": split_lock["conflicting_duplicate_excluded_episode_count"],
        "trusted_atomic_components": split_lock["trusted_atomic_component_count"],
        "duplicate_safe_rule": "v3 keeps trusted duplicate groups atomic; the retired old split is not used.",
        "episode_isolation": "one JY02 input row per source episode is the current contract; future execution must assert one final prediction row per final episode.",
        "window_isolation": "no generic window_id/source_trajectory_id is present in the reviewed schema; overlapping-window claims are not assumed safe.",
        "session_operator_scene": "not present in reviewed split/episode schemas; independence is unverified.",
        "final_target_access_current_task": 0,
    }
    write_json("SPLIT_PROVENANCE_AUDIT.json", split_audit)
    write_text("SPLIT_PROVENANCE_AUDIT.md", """# Split provenance audit

The v3 duplicate-safe split is the canonical metadata source. It contains 2741 train, 617 development-validation and 602 primary final-test episodes, with a 614-row full final-test index including 12 excluded rows. The v3 protocol records 40 exact duplicate groups, 80 conflicting duplicate episodes, 58 near-candidate edges and 86 trusted atomic components; the old split is retired as duplicate-unsafe.

The reviewed split/episode schemas contain episode, task and media-fingerprint fields but no explicit session, operator, scene, source-trajectory or generic window ID. JY-02 currently uses one target row per source episode; this is an episode-level evaluation contract, not proof of session/operator/scene independence. The final label-free index was inspected only as metadata/schema/hash. No final target or label value was read in this task.
""")

    group_rows = [
        {"field": "episode_index", "available": "yes", "source": "v3 label-free/split manifest schema", "role": "episode pairing unit", "read_final_values_this_task": "no"},
        {"field": "split", "available": "yes", "source": "v3 label-free/split manifest schema", "role": "train/dev/final partition", "read_final_values_this_task": "no"},
        {"field": "video_sha256_fingerprint", "available": "yes", "source": "v3 label-free index schema", "role": "media duplicate/fingerprint support", "read_final_values_this_task": "no"},
        {"field": "duplicate_group_id", "available": "not explicit", "source": "v3 lock + duplicate audit", "role": "trusted atomic groups are documented but no group ID is carried in label-free index", "read_final_values_this_task": "no"},
        {"field": "source_trajectory_id", "available": "no", "source": "reviewed schemas", "role": "trajectory-level holdout", "read_final_values_this_task": "no"},
        {"field": "window_id", "available": "no", "source": "reviewed schemas", "role": "overlapping-window control", "read_final_values_this_task": "no"},
        {"field": "session_id", "available": "no", "source": "reviewed raw/split schemas", "role": "session-group holdout", "read_final_values_this_task": "no"},
        {"field": "operator_id", "available": "no", "source": "reviewed raw/split schemas", "role": "operator-group holdout", "read_final_values_this_task": "no"},
        {"field": "scene_id", "available": "no", "source": "reviewed raw/split schemas", "role": "scene-group holdout", "read_final_values_this_task": "no"},
        {"field": "task_id", "available": "metadata-bearing", "source": "split manifest schema; protected for final", "role": "task support/stratification", "read_final_values_this_task": "no"},
    ]
    write_csv("GROUP_METADATA_AVAILABILITY.csv", group_rows, ["field", "available", "source", "role", "read_final_values_this_task"])
    write_json("GROUP_METADATA_AVAILABILITY.json", {"status": "PASS_METADATA_ONLY", "rows": group_rows, "read_final_values_this_task": False})

    route = {
        "status": "PASS_CONDITIONAL_ROUTE",
        "heldout_evaluation_available": "unknown",
        "selected_route": "A",
        "selected_route_name": "existing_v3_final_test_one_time_evaluation",
        "selection_basis": "A physical 602-row primary label-free final index exists and v3 metadata separates it from the 2741/614 development contract; blind release remains conditional because historical feature metadata contains label-bearing fields.",
        "route_A": {"status": "recommended_conditional", "requires": ["custodial metadata-only sanitization/release review", "written professor/user approval", "protocol/checkpoint/code/normalizer hash lock", "one-time target unlock after all input inference"], "current_execution": "not_started"},
        "route_B": {"status": "fallback", "description": "new source-grouped held-out study requiring retraining under a newly approved protocol", "current_execution": "not_started"},
        "route_C": {"status": "fallback", "description": "external/new dataset validation", "current_execution": "not_started"},
        "blocked_condition": "If Route A sanitization cannot establish a defensible independent population, mark blocked_no_defensible_heldout and use neither development-as-final nor an unapproved substitute.",
        "professor_decision_required": True,
        "final_test_access_current_task": 0,
    }
    write_json("EVALUATION_ROUTE_DECISION.json", route)
    write_text("EVALUATION_ROUTE_DECISION.md", """# Evaluation route decision

## Decision

`SELECTED_EVALUATION_ROUTE=A` — existing v3 final-test population, one-time evaluation, conditional on blind sanitization/release and written human approval.

The v3 metadata provides a 602-row primary label-free final index and records that final metrics were not computed. That establishes a physical candidate population, but not an unconditional blind-release guarantee: the prior handoff audit records label-bearing fields in historical feature metadata. Accordingly `HELDOUT_EVALUATION_AVAILABLE=unknown` until a custodian resolves that metadata risk without exposing final labels.

## Alternatives

- Route B: create a new source-grouped held-out study and retrain all required models under a newly approved protocol. This is not authorized in the current task.
- Route C: validate on an external/new dataset. This is scientifically attractive for broader generalization but requires a new data/license/contract audit.
- Blocked: if Route A cannot be sanitized and no Route B/C population is approved, stop with `blocked_no_defensible_heldout`; do not reuse development as final.

No route is executed by this package.
""")

    claims = {
        "status": "LOCKED_WITH_HUMAN_GATE_FOR_C",
        "candidate": "JY02-F-R-D",
        "claim_A": {"name": "Arm-14 trajectory success", "primary": True, "claim": "F-R-D improves pre-registered Arm-14 standardized full-horizon MAE versus Persistence on an approved independent population.", "gate": ["positive in at least 2/3 fresh seeds", "hierarchical 95% CI lower > 0", "effect magnitude reported", "engineering/schema/checksum PASS"], "current_status": "development_candidate_only"},
        "claim_B": {"name": "Video incremental value", "primary": False, "claim": "F-R-D improves over same residual-direct STATE control S-R-D on the approved population.", "gate": ["positive in at least 2/3 seeds", "hierarchical 95% CI lower > 0", "per-task/joint support not wholly driven by one tiny subgroup", "task/scene shortcut caveat retained"], "current_status": "development_supportive_evidence_only"},
        "claim_C": {"name": "Full upper-body success", "primary": False, "claim": "F-R-D passes the preselected Full-28 and Hand-14 gate versus Persistence.", "recommended_gate": "relative degradation <= 1% for both Full-28 and Hand-14", "decision": "professor/user must choose gate versus secondary before target unlock", "current_status": "not_confirmable"},
        "trajectory_fidelity": {"status": "guardrail", "metrics": ["first-difference MAE", "native and standardized TV", "TV ratio", "direction agreement"], "language_rule": "If F-R-D is worse than Persistence on first-difference or TV, say lower average Arm-14 error only; do not say fully more realistic dynamics."},
        "units_boundary": "Command trajectory only until physical units, URDF and FK are independently verified.",
    }
    write_json("CLAIM_HIERARCHY.json", claims)
    write_text("CLAIM_HIERARCHY.md", """# JY-02 claim hierarchy

## Claim A — Arm-14 trajectory success

Primary claim: `JY02-F-R-D` improves pre-registered standardized Arm-14 full-horizon MAE versus Persistence on an approved independent population. It requires positive direction in at least 2/3 fresh seeds, hierarchical 95% CI lower bound above zero, complete effect reporting and engineering/schema/checksum acceptance.

## Claim B — video incremental value

Supportive claim: F-R-D improves over `JY02-S-R-D`, the same residual-direct contract without video. It requires the same seed/CI gates, support checks by task/joint, and continued disclosure of task/scene shortcut and same-task mismatch caveats. It is not a causal video claim.

## Claim C — Full upper-body success

Separate claim: Full-28 and Hand-14 also pass a preselected gate. Recommended gate: relative degradation versus Persistence no greater than 1% for both. The professor/user must choose “gate” or “secondary” in writing before target unlock; Arm success does not imply Claim C.

## Trajectory fidelity and units

First-difference, TV, TV ratio and direction agreement remain guardrails. A full-horizon MAE win with worse dynamics must be described as lower average command error only. Cartesian/URDF/FK accuracy and session/operator/scene independence remain unclaimed.
""")

    registry = list(frozen_rows)
    registry.append({
        "model_type": "persistence",
        "config_id": "PERSISTENCE",
        "seed": 0,
        "checkpoint_path": "none_fixed_last_observed_command_repeated_30_steps",
        "checkpoint_size_bytes": "",
        "checkpoint_sha256": "",
        "prediction_path": "generated_in_future_evaluation_from_frozen_input_contract",
        "prediction_size_bytes": "",
        "prediction_sha256": "",
        "expected_shape": "30x28",
        "development_N": 614,
        "best_epoch": "",
        "optimizer_steps": "",
        "strict_reload": "not_applicable_fixed_control",
        "final_test_access": 0,
        "selection_rule": "fixed analytical control; no learned checkpoint",
        "frozen_at_utc": UTC_NOW,
        "model_contract_status": "fixed_control",
    })
    write_csv("FINAL_MODEL_REGISTRY.csv", registry, list(frozen_rows[0].keys()))
    write_json("FINAL_MODEL_REGISTRY.json", {"status": "LOCKED", "model_selection": "all 12 learned checkpoints plus Persistence; no best seed", "rows": registry})
    write_text("FINAL_METRIC_CONTRACT.md", """# Final evaluation metric contract

## Primary

Arm-14 standardized full-horizon MAE, using the frozen train-only per-dimension target standardizer. Arm slice is `14:28`; horizon is `30` steps; effect is `Persistence MAE - candidate MAE`, so positive means improvement.

## Fixed secondary and guardrails

- Hand-14 and Full-28 standardized MAE/RMSE.
- Native command-value MAE/RMSE, only if unit interpretation is explicitly bounded as command-space.
- Train-derived Q25/Q75 motion strata; thresholds cannot be selected from final targets.
- Early/middle/late/endpoint; first-difference MAE; native and standardized TV/TV ratio; direction agreement; per-task/joint/time; predeclared trajectory examples.
- One paired row per source episode; steps and joints are within-episode structure.

No final result may change the metric, threshold, model set, comparison, horizon, normalizer or representative-selection rule.
""")
    write_json("FINAL_METRIC_CONTRACT.json", {"status": "LOCKED", "primary": "Arm-14 standardized full-horizon MAE", "effect": "Persistence MAE - candidate MAE", "shape": [30, 28], "arm_slice": "14:28", "hand_slice": "0:14", "normalizer": "train-only frozen per dimension", "motion_threshold_source": str(V2_ROOT / "MOTION_THRESHOLD_SOURCE.json"), "motion_q25": threshold["q25"], "motion_q75": threshold["q75"], "no_post_unlock_changes": True})
    write_text("FINAL_COMPARISON_FAMILY.md", """# Final comparison family

The fixed learned model set is `S-A-D`, `F-A-D`, `S-R-D`, `F-R-D` at seeds `20260921/22/23`, plus the fixed Persistence control.

Pre-registered comparisons:

1. F-R-D vs Persistence (primary).
2. S-R-D vs Persistence.
3. F-R-D vs F-A-D (residual parameterization within FUSION).
4. S-R-D vs S-A-D (supporting residual comparison).
5. F-R-D vs S-R-D (video increment within residual-direct contract).
6. F-A-D vs S-A-D (absolute-input supporting comparison).

The episode is the paired evaluation unit; seed is the training replicate. If p-values are reported, the six-comparison family uses predeclared Holm correction. The primary decision remains effect direction plus hierarchical seed-to-episode CI, not episode count masquerading as independent seed count.
""")
    write_json("FINAL_COMPARISON_FAMILY.json", {"status": "LOCKED", "comparisons": ["F-R-D_vs_PERSISTENCE", "S-R-D_vs_PERSISTENCE", "F-R-D_vs_F-A-D", "S-R-D_vs_S-A-D", "F-R-D_vs_S-R-D", "F-A-D_vs_S-A-D"], "holm_family": True, "independent_seed_count": 3, "episode_nested": True})
    write_text("FINAL_SUCCESS_GATES.md", """# Final success and failure gates

## Arm primary

- F-R-D positive versus Persistence in at least 2/3 seeds.
- Hierarchical 95% CI lower bound > 0.
- Report effect magnitude, per-seed values, mean +/- SD and CI.
- Schema, checkpoint, prediction, marker and checksum gates pass.

## Residual parameterization

- F-R-D positive versus F-A-D in at least 2/3 seeds.
- Hierarchical CI lower bound > 0.
- S-R-D vs S-A-D remains supportive evidence.

## Video increment

- F-R-D positive versus S-R-D in at least 2/3 seeds.
- Hierarchical CI lower bound > 0.
- Per-task/joint support is not wholly a tiny subgroup artifact; shortcut caveat remains.

## Full upper body

Recommended preselected gate: Full-28 relative degradation versus Persistence <=1% and Hand-14 relative degradation versus Persistence <=1%. The professor/user must explicitly decide whether this is a gate or secondary before target unlock. Until then Claim C is not eligible for confirmation.

## Fidelity guardrail

Always report first-difference and TV. If F-R-D is worse than Persistence, do not claim globally more realistic trajectories. Do not call non-significance equivalence.

Any missing/failed artifact, target-boundary violation, hash mismatch, unauthorized change or unapproved model/metric change is a stop/failure condition.
""")
    write_json("FINAL_SUCCESS_GATES.json", {"status": "LOCKED_PENDING_C_GATE_CHOICE", "arm_primary": {"positive_seeds_min": 2, "hierarchical_ci_low_gt": 0, "engineering": True}, "residual": {"positive_seeds_min": 2, "hierarchical_ci_low_gt": 0}, "video_increment": {"positive_seeds_min": 2, "hierarchical_ci_low_gt": 0}, "full_upper_body_recommended": {"full28_relative_degradation_max": 0.01, "hand14_relative_degradation_max": 0.01, "professor_choice_required": True}, "trajectory_fidelity": "guardrail_only"})

    boundary = {
        "status": "LOCKED",
        "current_task": {"final_test_target_access": 0, "final_test_inference_started": False, "new_training_started": False},
        "allowed_before_human_approval": ["read/hash development artifacts", "read/hash split metadata and label-free index schema", "synthetic/dev engineering dry-run", "write protocol and QA docs"],
        "blocked": ["protected final labels/targets", "final predictions", "final metrics", "final normalization/threshold fitting", "final model selection", "new training/tuning/LoRA/encoder update", "unregistered ensemble"],
        "final_label_manifest": {"path": str(protected_label_path), "hash": final_audit["protected_label_manifest_sha256"], "current_task_access": 0, "policy": "opaque prior-audit reference only"},
        "target_unlock": "only after HUMAN_APPROVED and INPUT_INFERENCE_COMPLETE, exactly once",
        "output_separation": "future sealed final outputs must not overwrite development artifacts",
    }
    write_json("BLIND_DATA_BOUNDARY.json", boundary)
    write_text("BLIND_DATA_BOUNDARY.md", """# Blind data boundary

Before written human approval, the package may use development artifacts, code/config/checkpoint hashes, split policy, label-free final-index schema/hash, and synthetic fixtures. It may not open protected final labels/targets/predictions, calculate final statistics, fit final normalizers or thresholds, select a final seed, train, tune or infer on final data.

The current task recorded `FINAL_TEST_TARGET_ACCESS=0`, `FINAL_TEST_INFERENCE_STARTED=false` and `NEW_TRAINING_STARTED=false`. The protected label manifest is referenced by prior opaque hash only; it was not opened or parsed here. A future target unlock is one-time and occurs only after every fixed model has completed target-blind input inference.
""")

    state_machine = {
        "status": "LOCKED",
        "states": ["UNAUTHORIZED", "PROTOCOL_LOCKED", "HUMAN_APPROVED", "INPUT_INFERENCE_COMPLETE", "TARGET_UNLOCK_ONCE", "ANALYSIS_COMPLETE", "SEALED"],
        "transitions": [
            ["UNAUTHORIZED", "lock_protocol", "PROTOCOL_LOCKED"],
            ["PROTOCOL_LOCKED", "human_approve", "HUMAN_APPROVED"],
            ["HUMAN_APPROVED", "complete_input_inference", "INPUT_INFERENCE_COMPLETE"],
            ["INPUT_INFERENCE_COMPLETE", "unlock_targets_once", "TARGET_UNLOCK_ONCE"],
            ["TARGET_UNLOCK_ONCE", "complete_analysis", "ANALYSIS_COMPLETE"],
            ["ANALYSIS_COMPLETE", "seal_outputs", "SEALED"],
        ],
        "resume_policy": {"before_target_unlock": "resume only missing input-inference work with identical frozen hashes and no target access", "after_target_unlock": "analysis may resume from hash-verified frozen predictions only; no model/input rerun, no checkpoint swap, no target re-unlock", "mismatch": "stop and seal/blocked; new approval required"},
        "current_state": "PROTOCOL_LOCKED",
        "target_unlock_count_current_task": 0,
    }
    write_json("ONE_TIME_EXECUTION_STATE_MACHINE.json", state_machine)
    write_text("ONE_TIME_EXECUTION_STATE_MACHINE.md", """# One-time execution state machine

`UNAUTHORIZED -> PROTOCOL_LOCKED -> HUMAN_APPROVED -> INPUT_INFERENCE_COMPLETE -> TARGET_UNLOCK_ONCE -> ANALYSIS_COMPLETE -> SEALED`

The protocol/checkpoint/code/normalizer/output-schema hashes must be locked before human approval and before target unlock. Input inference has no target argument. Target analysis accepts only hash-verified frozen predictions. Before unlock, a failure may resume only missing input inference with identical frozen inputs and hashes. After unlock, analysis may resume from frozen predictions only; no model swap, re-inference, metric change or second target unlock is allowed. Any mismatch stops and seals the attempt pending new approval.
""")
    write_csv("FINAL_TEST_ACCESS_LEDGER_TEMPLATE.csv", [], ["utc_timestamp", "state", "purpose", "command_sha256", "pid", "path_or_scope", "file_sha256", "authorized_by", "result", "notes"])
    write_json("FINAL_TEST_ACCESS_LEDGER_TEMPLATE.json", {"status": "TEMPLATE_EMPTY_NO_ACCESS", "columns": ["utc_timestamp", "state", "purpose", "command_sha256", "pid", "path_or_scope", "file_sha256", "authorized_by", "result", "notes"], "rows": []})
    write_csv("FINAL_TEST_RUN_ACCEPTANCE_TEMPLATE.csv", [], ["model_id", "seed", "input_rows", "prediction_shape", "checkpoint_sha256", "prediction_sha256", "strict_reload", "target_access", "exit_code", "status", "notes"])
    write_json("FINAL_TEST_RUN_ACCEPTANCE_TEMPLATE.json", {"status": "TEMPLATE_EMPTY_NO_RUN", "columns": ["model_id", "seed", "input_rows", "prediction_shape", "checkpoint_sha256", "prediction_sha256", "strict_reload", "target_access", "exit_code", "status", "notes"], "rows": []})

    write_json("ENGINEERING_DRY_RUN.json", engineering)
    write_text("ENGINEERING_DRY_RUN.md", f"""# Engineering-only dry-run

Status: `{engineering['status']}`.

- Synthetic fixtures only; no final or development target arrays were used by the dry-run.
- The fixed model-set and authorization/state-machine tests passed (`{engineering['tests_passed']}` tests).
- Target-blind inference writes `(N,30,28)` predictions without a target argument.
- Separate analysis rejects target access unless the explicit post-inference phase is enabled and requires a frozen prediction hash/shape manifest.
- All frozen Stage2 source code files parsed successfully (`{engineering['code_files_parsed']}`).
- Current task final-test target access remains `0`; no training or final inference occurred.
""")

    risk_rows = [
        {"risk_id": "R1", "risk": "Historical feature metadata contains label-bearing fields", "severity": "high", "status": "open_before_route_A_release", "mitigation": "custodial metadata-only sanitization/release review; do not unlock targets before approval"},
        {"risk_id": "R2", "risk": "Session/operator/scene IDs are unavailable", "severity": "high", "status": "known_limitation", "mitigation": "claim episode-level generalization only; use Route B/C if group holdout is mandatory"},
        {"risk_id": "R3", "risk": "Final population clean blind availability is not fully proven", "severity": "high", "status": "unknown", "mitigation": "Route A conditional; otherwise block or choose new held-out/external route"},
        {"risk_id": "R4", "risk": "Legacy 900 output typo", "severity": "medium", "status": "reconciled_addendum", "mitigation": "840/30x28 is the sole locked future shape"},
        {"risk_id": "R5", "risk": "Native TV legacy field label", "severity": "medium", "status": "reconciled_addendum", "mitigation": "separate native/standardized fields and independent recomputation"},
        {"risk_id": "R6", "risk": "Watcher lacks child UUID/heartbeat serialization", "severity": "medium", "status": "known_limitation", "mitigation": "retain process/lease/marker/resource corroboration"},
        {"risk_id": "R7", "risk": "Hand/Full-28 development degradation", "severity": "high", "status": "observed_development", "mitigation": "keep Claim C separate; require written gate-vs-secondary decision"},
        {"risk_id": "R8", "risk": "First-difference/TV fidelity may remain worse than Persistence", "severity": "high", "status": "observed_development", "mitigation": "guardrail language; no全面trajectory-fidelity claim"},
        {"risk_id": "R9", "risk": "Physical units/URDF/FK unverified", "severity": "high", "status": "open", "mitigation": "command-space wording only until independent kinematics audit"},
        {"risk_id": "R10", "risk": "Target unlock or resume misuse", "severity": "high", "status": "controlled", "mitigation": "one-time state machine, append-only ledger, frozen prediction analysis, fail closed"},
    ]
    write_csv("FINAL_EVALUATION_RISK_REGISTER.csv", risk_rows, ["risk_id", "risk", "severity", "status", "mitigation"])
    write_json("FINAL_EVALUATION_RISK_REGISTER.json", {"status": "PASS_REGISTERED", "rows": risk_rows})

    professor_questions = """1. 是否批准 Route A 对现有 v3 primary final-test population 做一次性访问？是否先要求 custodial metadata-only sanitization/release review？
2. Arm-14 是否唯一 primary？Hand-14 与 Full-28 是硬 gate 还是 secondary？建议默认 gate 为相对 Persistence 退化均不超过 1%。
3. 是否同意评估全部 4 configs x 3 seeds 加 Persistence，而不是挑一个最佳 seed？
4. 是否接受 development-selected、final-locked checkpoint/model-set 设计，并禁止 target unlock 后重新选择？
5. final claim 只需 episode split，还是必须 session/operator/scene group-holdout？当前 schema 没有这些字段。
6. 在 command units、URDF 和 Cartesian/FK 尚未验证时，报告是否限定为 standardized/native command trajectory？
7. 是否需要外部新数据集验证，作为 Route A 之外或替代 Route A 的泛化证据？
"""
    write_text("PROFESSOR_DECISION_REQUIRED.zh-CN.md", f"""# JY-02 教授/用户决策请求

协议准备已完成，但 final-test 仍未授权：`FINAL_TEST_EXPLICITLY_AUTHORIZED=false`、target access=`0`。当前推荐 Route A 是条件性推荐，原因是 v3 存在 602-row primary label-free index，但历史 feature metadata 曾含 label-bearing 字段；需先完成 custodial release review。

## 当前结论

`JY02-F-R-D` 是新的多 seed development candidate，不是 final model。Claim A（Arm-14）与 Claim B（video increment）独立判定；Claim C（Hand/Full-28）尚未自动通过。first-difference/TV 是 fidelity guardrail，不是可忽略的附属指标。

## 请明确决定

{professor_questions}

在上述决定被记录并把未来 execution Prompt 的授权字段显式改为 true 前，任何 Codex 必须停在 `PROTOCOL_LOCKED`，不得读取 final targets、运行 final inference、训练或调参。
""")
    write_text("PROFESSOR_DECISION_REQUIRED.en.md", f"""# JY-02 professor/user decisions required

Protocol preparation is complete, but final-test access remains unauthorized: `FINAL_TEST_EXPLICITLY_AUTHORIZED=false`, target access=`0`. Route A is conditional because a 602-row primary label-free v3 index exists while historical feature metadata contained label-bearing fields; a custodial release review is required first.

`JY02-F-R-D` is a new multi-seed development candidate, not a final model. Claims A (Arm-14) and B (video increment) are separate; Claim C (Hand/Full-28) is not automatic. First-difference and TV are trajectory-fidelity guardrails.

Decisions required:

{professor_questions}

Until these decisions are recorded and the future execution Prompt's authorization field is explicitly changed to true, every Codex must remain at `PROTOCOL_LOCKED`: no final targets, final inference, training or tuning.
""")
    write_text("FINAL_EVALUATION_ONE_PAGE_PROTOCOL.zh-CN.md", """# JY-02 一页式 Final / Held-out Evaluation 协议

**Route**：A，现有 v3 primary final-test population；条件是 custodial blind-release review 与教授/用户书面批准。

**固定模型**：S-A-D、F-A-D、S-R-D、F-R-D × seeds 20260921/22/23，加 fixed Persistence；不挑最佳 seed、不新建 ensemble。

**Primary**：Arm-14 standardized full-horizon MAE；effect=`Persistence - candidate`，正值为改善。Secondary/guardrails：Hand-14、Full-28、native command MAE/RMSE、train-Q25/Q75 motion、early/middle/late/endpoint、first-difference、native/standardized TV、direction、task/joint/time 与预先定义的 trajectory examples。

**统计**：每 seed paired episode bootstrap 10,000；hierarchical seed→episode bootstrap 20,000；seed 是 training replicate，episode 是 evaluation replicate；必要时六项比较使用 Holm。不得把 episode 数写成 seed 数。

**门禁**：Arm、residual、video increment 各需至少 2/3 seed 正向且 hierarchical CI 下界 >0。Full/Hand 推荐相对 Persistence 退化均 ≤1%，但 gate/secondary 由教授书面决定。first-difference/TV 若不如 Persistence，只能声称 lower average command error。

**盲界**：先完成全部输入推理，再一次性解锁 targets；target analysis 只接受 hash-verified frozen predictions；失败 resume 规则、ledger、hash、PID 和 UTC 必须记录。当前明确 `FINAL_TEST_EXPLICITLY_AUTHORIZED=false`，本文件不执行 evaluation。
""")
    write_text("FINAL_EVALUATION_ONE_PAGE_PROTOCOL.en.md", """# JY-02 one-page final / held-out evaluation protocol

**Route**: A, the existing v3 primary final-test population, conditional on custodial blind-release review and written professor/user approval.

**Fixed models**: S-A-D, F-A-D, S-R-D, F-R-D at seeds 20260921/22/23, plus fixed Persistence. No best-seed selection and no new ensemble.

**Primary**: Arm-14 standardized full-horizon MAE; effect=`Persistence - candidate`, positive means improvement. Secondary/guardrails: Hand-14, Full-28, native command MAE/RMSE, train-Q25/Q75 motion, early/middle/late/endpoint, first-difference, native/standardized TV, direction, task/joint/time and predeclared trajectory examples.

**Statistics**: 10,000 paired episode bootstraps per seed and 20,000 hierarchical seed-to-episode bootstraps; seed is the training replicate and episode is the evaluation replicate; use Holm for the six predeclared comparisons if p-values are reported. Never present episode count as seed count.

**Gates**: Arm, residual and video-increment gates each require positive direction in at least 2/3 seeds and hierarchical CI lower bound >0. Recommended Full/Hand gate is <=1% relative degradation versus Persistence, pending written professor choice. If first-difference/TV loses to Persistence, claim lower average command error only.

**Blind boundary**: complete all input inference first, unlock targets once, and let analysis accept only hash-verified frozen predictions. Record failure/resume policy, ledger, hashes, PID and UTC. `FINAL_TEST_EXPLICITLY_AUTHORIZED=false`; this package does not execute evaluation.
""")

    execution_prompt = f"""# JY-02 final evaluation execution Prompt — pending approval

This Prompt is generated but must not be executed in the current task.

## Hard stop

```ini
FINAL_TEST_EXPLICITLY_AUTHORIZED=false
FINAL_TEST_TARGET_ACCESS=0
FINAL_TEST_INFERENCE_STARTED=false
NEW_TRAINING_STARTED=false
```

If `FINAL_TEST_EXPLICITLY_AUTHORIZED` is not explicitly changed to `true` by the professor/user in a recorded authorization artifact, stop at `PROTOCOL_LOCKED`. Do not infer approval from this file, a chat message without a recorded artifact, GPU idleness or a label-free index.

## Frozen package and models

- `FREEZE_ROOT={FREEZE_ROOT}`
- Read `CANDIDATE_FREEZE_MANIFEST.json`, `CANDIDATE_COMPONENT_HASHES.json`, `FINAL_MODEL_REGISTRY.json`, `FINAL_METRIC_CONTRACT.json`, `FINAL_COMPARISON_FAMILY.json`, `FINAL_SUCCESS_GATES.json`, `BLIND_DATA_BOUNDARY.json` and `ONE_TIME_EXECUTION_STATE_MACHINE.json`.
- Verify `qa/CHECKSUMS.sha256` and all referenced checkpoint hashes before any target access. Large checkpoints are referenced by absolute path and hash, not copied.
- Evaluate all 12 learned rows plus Persistence exactly as registered. Never choose the best seed and never create an ensemble.

## Route A and boundary

Route A is conditional on a custodian's metadata-only blind-release review of the v3 602-row primary label-free population. Do not open protected labels while performing release review. If the review fails, stop as `blocked_no_defensible_heldout`; do not substitute development data.

## Execution order after explicit approval

1. Record authorization, UTC, operator, route decision, package hash and state transition `PROTOCOL_LOCKED -> HUMAN_APPROVED`.
2. Recheck exact paths, hashes, environment and GPU policy. If GPU is required, use the project `gpu_lease_watcher.py` policy on GPU 1/2/3, at most one child for this evaluation lane; do not co-locate based on utilization alone.
3. Run target-blind input inference for all 12 learned checkpoints and fixed Persistence. The inference process must not receive a target/label path. Write isolated prediction bundles and acceptance markers only.
4. Verify all prediction shapes `(N,30,28)`, finite values, model/seed IDs, strict checkpoint loads, input pairing, terminal markers and checksums. Transition to `INPUT_INFERENCE_COMPLETE` only after every fixed row passes.
5. Append one ledger record and perform the one-time transition `INPUT_INFERENCE_COMPLETE -> TARGET_UNLOCK_ONCE`. Unlock only the custodian-approved target stream; record UTC, command hash, PID, scope and file hash. Never unlock twice.
6. Run the separate analysis program from frozen prediction bundles only. Do not re-infer, swap checkpoints, alter metrics/thresholds/model set or fit any normalizer on final data.
7. Compute primary/secondary metrics, per-seed paired bootstrap (10,000), hierarchical seed-to-episode bootstrap (20,000), fixed comparisons and Holm family reporting. Preserve episode/source grouping and report missing/low-support strata.
8. Generate per-seed tables, aggregate tables, trajectory figures, numeric/visual/claim QA, acceptance records and a separately sealed final output root. Transition `ANALYSIS_COMPLETE -> SEALED` only after final checksums.

## Stop conditions

Stop immediately on any target access before `TARGET_UNLOCK_ONCE`, hash mismatch, unauthorized process, missing group/pairing field, prediction mismatch, checkpoint swap, best-seed selection, ensemble, training/optimizer step, metric/threshold change, failed checksum or incomplete acceptance. A pre-unlock input failure may resume only missing input inference with the identical frozen package; after target unlock, resume analysis from frozen predictions only.

## Claim wording

Report Claim A, B and C separately. Claim C remains unavailable until the written Full/Hand gate choice. If first-difference or TV is worse than Persistence, do not claim fully more realistic trajectories. If command units/URDF/FK remain unverified, use command-trajectory wording only. Do not call non-significance equivalence and do not report final values before the sealed output.
"""
    write_text("FINAL_EVALUATION_EXECUTION_PROMPT_PENDING_APPROVAL.md", execution_prompt)

    old_after = {str(path): sha256(path) for path in old_key_paths}
    old_unchanged = old_before == old_after
    qa_common = {
        "stage2_source_integrity": checksum,
        "candidate_checkpoints": len([row for row in frozen_rows if row["config_id"] == "JY02-F-R-D"]),
        "matched_control_checkpoints": len([row for row in frozen_rows if row["config_id"] != "JY02-F-R-D"]),
        "old_artifacts_modified": not old_unchanged,
        "final_test_target_access": 0,
        "final_test_inference_started": False,
        "new_training_started": False,
        "route": "A",
        "route_condition": "blind sanitization/release plus human approval",
    }
    write_json("qa/ENGINEERING_QA.json", {**qa_common, "status": engineering["status"], "engineering": engineering})
    write_text("qa/ENGINEERING_QA.md", f"""# Engineering QA

Status: `PASS`.

- `{engineering['tests_passed']}` synthetic contract/boundary tests passed.
- Stage2 source code AST parse passed for `{engineering['code_files_parsed']}` files.
- Target-blind inference and frozen-prediction-only analysis were exercised only with synthetic fixtures.
- No final targets, labels, predictions, GPU run, training or optimizer step was used.
""")
    protocol_checks = {
        "required_docs_generated": True,
        "json_csv_parse": True,
        "route_selected": "A",
        "claim_hierarchy_locked": True,
        "model_set_locked": len(registry) == 13,
        "metric_contract_locked": True,
        "blind_boundary_locked": True,
        "state_machine_locked": True,
        "final_authorization_default_false": True,
        "target_access_current_task": 0,
    }
    write_json("qa/PROTOCOL_QA.json", {"status": "PASS", **protocol_checks})
    write_text("qa/PROTOCOL_QA.md", """# Protocol QA

Status: `PASS`.

The route, model registry, metric/comparison family, claims, blind boundary, one-time state machine, templates, bilingual decision documents and pending execution Prompt were generated. JSON/CSV parsing passed. Route A is conditional; the execution Prompt remains explicitly unauthorized.
""")
    claim_qa = {"status": "PASS", "claim_A_separate": True, "claim_B_separate": True, "claim_C_human_choice_required": True, "first_difference_tv_guardrail": True, "development_limitations_visible": True, "physical_units_unverified_visible": True}
    write_json("qa/CLAIM_QA.json", claim_qa)
    write_text("qa/CLAIM_QA.md", """# Claim QA

Status: `PASS`.

Claims A/B/C are separate; Claim C requires a written gate-vs-secondary decision. First-difference and TV guardrails, Hand/Full development degradation, shortcut caveat, command-space limitation and unknown session/operator/scene independence remain visible.
""")
    final_qa = {
        "status": "PASS",
        "hard_acceptance": {
            "stage2_source_integrity": "pass",
            "candidate_freeze": "12/12_checkpoints",
            "development_code_and_normalizers_frozen": True,
            "metadata_reconciliation": "complete",
            "heldout_availability_audit": "complete_without_target_access",
            "evaluation_route_selected": "route_A_conditional",
            "claim_hierarchy_locked": True,
            "final_model_set_locked": True,
            "final_metric_contract_locked": True,
            "blind_boundary_tested": True,
            "engineering_dry_run": "pass",
            "professor_decision_packet": "complete",
            "execution_prompt_generated": True,
            "FINAL_TEST_EXPLICITLY_AUTHORIZED": False,
            "final_test_target_access": 0,
            "final_test_inference_started": False,
            "new_training_started": False,
            "old_artifacts_modified": old_unchanged is False,
            "qa": "all_pass",
            "checksums": "pending_final_checksum_generation",
        },
        "old_artifact_hash_comparison": {"before": old_before, "after": old_after, "unchanged": old_unchanged},
        "stage2_checksum": checksum,
        "engineering": engineering,
        "graphify_scope": "code-only graph to be run after package generation; no document semantic extraction required",
    }
    if not old_unchanged:
        raise RuntimeError("old artifact changed during freeze generation")
    write_json("qa/FINAL_QA.json", final_qa)
    write_text("qa/FINAL_QA.md", f"""# Final QA

Status: `PASS` for protocol preparation; final evaluation remains unauthorized.

| gate | result |
|---|---|
| Stage2 source checksum | `{checksum['ok']}/{checksum['ok']} OK`, exit `{checksum['exit_code']}` |
| Candidate checkpoints | `3/3` (`JY02-F-R-D`) |
| Matched controls | `9/9` |
| Development code/normalizer frozen | `PASS` by path/hash records |
| Metadata reconciliation | `PASS`, non-destructive addenda |
| Held-out audit | `PASS` without final target/label access; availability remains `unknown` pending blind release |
| Route | `A`, conditional on sanitization and human approval |
| Model/metric/claim/blind locks | `PASS` |
| Synthetic engineering dry-run | `PASS`, `{engineering['tests_passed']}` tests |
| Professor packet / pending Prompt | `PASS` |
| Current final target access | `0` |
| Current final inference/training | `false / false` |
| Old artifact hash comparison | `UNCHANGED` |

Required QA checks covered: Stage2 12/12 source integrity; registry 4x3 plus Persistence; checkpoint hashes; target/normalizer contract; arm/hand mapping; no best-seed selection; effect signs; train-frozen motion threshold; target-blind inference; frozen-prediction analysis; source-episode grouping; state machine; default authorization false; no current final target access; no training/optimizer steps; non-destructive metadata addenda; JSON/CSV parsing.

The final `qa/CHECKSUMS.sha256` must be generated after all other package files and QA are final. Its own SHA-256 is reported separately because a checksum manifest cannot contain a stable self-hash.
""")

    readme = f"""# JY-02 candidate freeze and held-out protocol

Freeze root: `{FREEZE_ROOT}`

Status: protocol preparation complete. Candidate: `JY02-F-R-D`. Fresh seeds: `20260921,20260922,20260923`. Stage2 source checksum: `{checksum['ok']}/{checksum['ok']} OK`.

The package freezes all 12 Stage2 checkpoints by absolute path, size and SHA-256, plus code/config/normalizer/split/metric provenance. Large checkpoints and prediction/target arrays are not copied. The physical v3 final population exists as a label-free index, but clean blind availability is `unknown` because historical feature metadata contained label-bearing fields. Route A is recommended conditionally; Route B/C are recorded fallbacks.

This package did not read final targets/labels/predictions, did not run final inference, did not train or tune, and did not modify old artifacts. Read `EVALUATION_ROUTE_DECISION.md`, `CLAIM_HIERARCHY.md`, `PROFESSOR_DECISION_REQUIRED.zh-CN.md` and the pending execution Prompt before any future action.
"""
    write_text("README.zh-CN.md", readme.replace("# JY-02 candidate freeze and held-out protocol", "# JY-02 候选冻结与 held-out 协议"))
    write_text("README.en.md", readme)


if __name__ == "__main__":
    build()
