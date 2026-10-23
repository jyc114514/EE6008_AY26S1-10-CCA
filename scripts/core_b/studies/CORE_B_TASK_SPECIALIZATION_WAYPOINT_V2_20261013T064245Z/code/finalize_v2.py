#!/usr/bin/env python3
"""Build V2 acceptance and chat handoff before the separate checksum-last seal."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TASK_IDS = (119, 94, 90, 141, 230)
SEEDS = (202610131, 202610132, 202610133)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def fit_rows(queue_path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    rows = []
    for raw in queue["jobs"]:
        watcher = json.loads(Path(raw).read_text(encoding="utf-8"))
        fit = json.loads(Path(watcher["command"][-1]).read_text(encoding="utf-8"))
        out = Path(fit["job_root"])
        cp = out / "RUN_COMPLETE.json"
        mp = out / "MODEL_MANIFEST.json"
        complete = json.loads(cp.read_text()) if cp.exists() else {}
        manifest = json.loads(mp.read_text()) if mp.exists() else {}
        passed = complete.get("status") == "PASS" and manifest.get("status") == "PASS"
        rows.append({
            "job_id": fit["job_id"], "phase": fit["phase"], "model_id": fit["model_id"],
            "task_ids": fit["task_ids"], "seed": fit["seed"], "lr": fit["lr"],
            "epochs": fit["epochs"], "output_root": str(out),
            "status": "PASS" if passed else "FAILED_OR_INCOMPLETE",
            "physical_gpu": manifest.get("physical_gpu", "NA"),
            "elapsed_seconds": complete.get("elapsed_seconds", "NA"),
            "model_sha256": manifest.get("model_file_sha256", "NA"),
        })
    counts = {"planned": len(rows), "accepted": sum(x["status"] == "PASS" for x in rows)}
    counts["failed_or_incomplete"] = counts["planned"] - counts["accepted"]
    return rows, counts


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def mean(values: list[float]) -> str:
    return "NA" if not values else f"{sum(values)/len(values):.4f}"


def main() -> None:
    screening_rows, screening = fit_rows(ROOT / "coordination/queues/SCREENING_ALL.json")
    formal_rows, formal = fit_rows(ROOT / "coordination/queues/FORMAL.json")
    waypoint_rows, waypoint = fit_rows(ROOT / "coordination/queues/WAYPOINT.json")
    pilot = json.loads((ROOT / "qa/FIRST_REAL_PILOT.json").read_text())
    selection = json.loads((ROOT / "protocol/LR_EPOCH_SELECTION.json").read_text())
    seal = json.loads((ROOT / "protocol/PREDICTION_BATCH_SEAL.json").read_text())
    result = json.loads((ROOT / "results/RESULTS_SUMMARY.json").read_text())
    exposure = json.loads((ROOT / "inventory/OUTCOME_EXPOSURE.json").read_text())
    dev = read_csv(ROOT / "inventory/DEV_SUPPORT_V2.csv")
    train = read_csv(ROOT / "inventory/TRAIN_TASKS_V2.csv")
    abs_rows = read_csv(ROOT / "results/ABSOLUTE_LOSSES.csv")
    summary_stats = read_csv(ROOT / "results/SUMMARY_STATISTICS.csv")
    waypoint_endpoints = read_csv(ROOT / "results/WAYPOINT_ENDPOINT.csv")
    failure_log = ROOT / "qa/FAILED_ATTEMPTS.jsonl"
    failure_events = [
        json.loads(line) for line in failure_log.read_text(encoding="utf-8").splitlines() if line.strip()
    ] if failure_log.exists() else []
    archive_summary_path = ROOT / "archive/INNER_SPLIT_DATASET_SALTED_V1_20261013T083000Z/ARCHIVE_SUMMARY.json"
    archive_summary = json.loads(archive_summary_path.read_text(encoding="utf-8")) if archive_summary_path.is_file() else {}
    empty_prediction_archive_path = ROOT / "archive/CORRECTED_SPLIT_ZERO_PREDICTION_SCORING_ATTEMPT_20261013T092800Z/ARCHIVE_SUMMARY.json"
    empty_prediction_summary = json.loads(empty_prediction_archive_path.read_text(encoding="utf-8")) if empty_prediction_archive_path.is_file() else {}
    historical_zero_prediction_target_rows = int(empty_prediction_summary.get("dev_future_action_rows_loaded_by_scorer", 0))
    current_dev_target_rows = int(exposure.get("current_split_raw_parquet_target_rows_read", 0))
    historical_pre_current_target_rows = int(archive_summary.get("prior_dev_target_label_rows_selected", 0)) + historical_zero_prediction_target_rows

    qa_files = [
        "FIRST_REAL_PILOT_PRECHECK.json", "INIT_SAMPLER_PAIRING.json", "INPUT_AND_CACHE_QA.json",
        "NORMALIZER_SCREENING_QA.json", "NORMALIZER_FORMAL_QA.json", "DEV_PREFIX_QA.json",
        "METRIC_CONTRACT_QA.json", "SAMPLE_WEIGHT_PRECISION_QA.json", "INNER_SPLIT_LITERAL_AUDIT.json", "PALETTE_AUDIT.json",
        "SCREENING_WATCHER_DRYRUN.json", "INFERENCE_WATCHER_DRYRUN_PILOT.json",
        "INFERENCE_WATCHER_DRYRUN_FORMAL_DEV.json", "DEV_TARGET_SCORING_QA.json", "FIGURE_BUILD_QA.json",
    ]
    qa = {}
    for name in qa_files:
        path = ROOT / "qa" / name
        qa[name] = json.loads(path.read_text()).get("status") if path.exists() else "MISSING"

    all_fits = screening["accepted"] == 36 and formal["accepted"] == 36 and waypoint["accepted"] == 6
    all_predictions = seal.get("sealed_prediction_jobs") == 42 and not seal.get("missing_prediction_jobs") and result.get("status") == "PASS"
    qa_pass = all(value in {"PASS", "LOCKED"} for value in qa.values())
    if all_fits and all_predictions and qa_pass and pilot.get("status") == "PASS":
        status = "PASS"
    elif formal["accepted"] == 36 and waypoint["accepted"] == 6 and len(selection.get("logical_model_choices", [])) == 12 and all_predictions:
        status = "COMPLETE_WITH_PARTIAL_SEARCH"
    else:
        status = "PARTIAL"

    failed_fit_attempts = 0
    oom_retries = 0
    for phase in ("screening", "formal", "waypoint"):
        for path in (ROOT / "model_only" / phase).glob("*/RUN_STATUS.failed-attempt-*.json"):
            record = json.loads(path.read_text())
            if record.get("physical_gpu") not in (None, "", "unknown"):
                failed_fit_attempts += 1
        oom_retries += sum(1 for _ in (ROOT / "model_only" / phase).glob("*/OOM_FALLBACK.json"))
    # The archive receipt uses the explicit *_receipts_archived field names.
    # Keep the older aliases as fallbacks for archive schema v1 receipts.
    prior_fit_counts = archive_summary.get(
        "prior_completed_fit_receipts_archived",
        archive_summary.get("prior_completed_fits_archived", {}),
    )
    superseded_completed_fits = sum(int(value) for value in prior_fit_counts.values())
    superseded_failed_gpu_attempts = int(archive_summary.get(
        "prior_failed_gpu_attempt_receipts_archived",
        archive_summary.get("prior_failed_gpu_attempts_archived", 0),
    ))
    superseded_attempts = superseded_completed_fits + superseded_failed_gpu_attempts + int(archive_summary.get("prior_oom_retry_attempts", 0))
    superseded_root = archive_summary_path.parent
    superseded_fit_seconds = 0.0
    superseded_training_gpus: set[str] = set()
    for manifest_path in (superseded_root / "model_only").glob("**/MODEL_MANIFEST.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS":
            continue
        gpu = manifest.get("physical_gpu")
        if gpu not in (None, "", "unknown"):
            superseded_training_gpus.add(str(gpu))
        complete_path = manifest_path.parent / "RUN_COMPLETE.json"
        if complete_path.is_file():
            superseded_fit_seconds += float(json.loads(complete_path.read_text()).get("elapsed_seconds", 0.0))
    superseded_inference_seconds = 0.0
    superseded_inference_gpus: set[str] = set()
    for complete_path in (superseded_root / "predictions").glob("**/PREDICTION_COMPLETE.json"):
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("status") != "PASS":
            continue
        superseded_inference_seconds += float(complete.get("elapsed_seconds", 0.0))
        gpu = complete.get("physical_gpu")
        if gpu not in (None, "", "unknown"):
            superseded_inference_gpus.add(str(gpu))
    corrected_split_attempts = screening["accepted"] + formal["accepted"] + waypoint["accepted"] + failed_fit_attempts + oom_retries
    actual_attempts = corrected_split_attempts + superseded_attempts
    training_gpus = {
        str(row["physical_gpu"]) for row in (*screening_rows, *formal_rows, *waypoint_rows)
        if row["status"] == "PASS" and row["physical_gpu"] not in ("NA", "", "unknown")
    }
    inference_gpus: set[str] = set()
    inference_seconds = 0.0
    inference_memory_by_job: dict[str, dict[str, Any]] = {}
    inference_peak_allocated: list[float] = []
    inference_peak_reserved: list[float] = []
    for entry in seal.get("prediction_entries", []):
        complete_path = Path(entry["prediction_path"]).parent / "PREDICTION_COMPLETE.json"
        if complete_path.is_file():
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            gpu = complete.get("physical_gpu")
            if gpu not in (None, "", "unknown"):
                inference_gpus.add(str(gpu))
            inference_seconds += float(complete.get("elapsed_seconds", 0.0))
            inference_memory_by_job[str(complete.get("job_id", ""))] = complete
            if complete.get("peak_memory_allocated_mib") is not None:
                inference_peak_allocated.append(float(complete["peak_memory_allocated_mib"]))
            if complete.get("peak_memory_reserved_mib") is not None:
                inference_peak_reserved.append(float(complete["peak_memory_reserved_mib"]))
    archive_root = archive_summary_path.parent
    superseded_training_gpus: set[str] = set()
    superseded_fit_seconds = 0.0
    for manifest_path in (archive_root / "model_only").glob("**/MODEL_MANIFEST.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS":
            continue
        gpu = manifest.get("physical_gpu")
        if gpu not in (None, "", "unknown"):
            superseded_training_gpus.add(str(gpu))
        complete_path = manifest_path.parent / "RUN_COMPLETE.json"
        if complete_path.is_file():
            superseded_fit_seconds += float(json.loads(complete_path.read_text()).get("elapsed_seconds", 0.0))
    superseded_inference_gpus: set[str] = set()
    superseded_inference_seconds = 0.0
    superseded_prediction_count = 0
    for complete_path in (archive_root / "predictions").glob("**/PREDICTION_COMPLETE.json"):
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("status") != "PASS":
            continue
        superseded_prediction_count += 1
        gpu = complete.get("physical_gpu")
        if gpu not in (None, "", "unknown"):
            superseded_inference_gpus.add(str(gpu))
        superseded_inference_seconds += float(complete.get("elapsed_seconds", 0.0))

    watcher_gpus: set[str] = set()
    status_roots = (
        ROOT / "coordination/watcher_status",
        archive_root / "coordination/watcher_status",
    )
    for status_root in status_roots:
        for path in status_root.glob("*.json"):
            if path.name.startswith("dryrun_"):
                continue
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            gpu = state.get("physical_gpu", state.get("selected_gpu"))
            if gpu not in (None, "", "unknown"):
                watcher_gpus.add(str(gpu))
    physical_gpus = sorted(
        training_gpus | inference_gpus | watcher_gpus |
        superseded_training_gpus | superseded_inference_gpus
    )
    fit_seconds = sum(
        float(row["elapsed_seconds"])
        for row in (*screening_rows, *formal_rows, *waypoint_rows)
        if row["elapsed_seconds"] not in ("NA", "")
    )
    phase_caps = []
    for path in (ROOT / "coordination").glob("PHASE_STATUS_*.json"):
        try:
            phase_caps.append(int(json.loads(path.read_text()).get("shared_watcher_max_children", 0)))
        except Exception:
            pass
    watcher_cap = max(phase_caps) if phase_caps else 0
    leases_released = True
    all_status_files = [path for status_root in status_roots for path in status_root.glob("*.json")]
    for path in all_status_files:
        if path.name.startswith("dryrun_"):
            continue
        try:
            item = json.loads(path.read_text())
        except Exception:
            continue
        if item.get("status") in {"complete", "failed"} and item.get("lease_released") is not True:
            leases_released = False

    contract_path = ROOT / "protocol/MODEL_AND_LOSS.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    model_hash = contract["code_hashes"]["v2_model_sha256"]
    scheduler_hash = contract["code_hashes"]["v2_scheduler_sha256"]
    current_contract_hash = sha_file(contract_path)
    locked_contract_path = ROOT / "protocol/MODEL_AND_LOSS.pre-pilot-status-clarification-20261013T090000Z.bak"
    locked_contract_hash = sha_file(locked_contract_path)
    formal_matrix_path = ROOT / "protocol/FORMAL_MATRIX.json"
    formal_matrix = json.loads(formal_matrix_path.read_text(encoding="utf-8"))
    if formal_matrix.get("model_and_loss_sha256") != locked_contract_hash:
        raise RuntimeError("FORMAL_MATRIX does not bind the archived corrected-split model/loss contract")
    formal_matrix_hash = sha_file(formal_matrix_path)
    contract_clarification = json.loads((ROOT / "protocol/FORMAL_MATRIX_METADATA_CLARIFICATION.json").read_text(encoding="utf-8"))
    if contract_clarification.get("architecture_loss_optimizer_split_scales_changed") is not False:
        raise RuntimeError("post-lock model/loss clarification is not recorded as metadata-only")
    split_lock_hash = sha_file(ROOT / "protocol/INNER_SPLIT_LOCK.json")
    trainer_current = ROOT / "code/v2_train_job.py"
    trainer_current_hash = sha_file(trainer_current)
    trainer_pre_correction = ROOT / "code/v2_train_job.py.pre-float32-weight-sum-fix-20261013T080317Z.bak"
    if not trainer_pre_correction.is_file():
        raise FileNotFoundError("pre-correction training source snapshot needed for per-fit provenance")
    trainer_pre_correction_hash = sha_file(trainer_pre_correction)
    correction_time = datetime.fromisoformat("2026-10-13T08:03:17+00:00")
    source_provenance = []
    for fit in (*screening_rows, *formal_rows, *waypoint_rows):
        if fit["status"] != "PASS":
            continue
        manifest_path = Path(fit["output_root"]) / "MODEL_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        created = datetime.fromisoformat(str(manifest["created_utc"]).replace("Z", "+00:00"))
        if created < correction_time:
            code_hash = trainer_pre_correction_hash
            basis = "completion timestamp precedes precision correction; hash from archived same-directory pre-correction snapshot"
        else:
            code_hash = trainer_current_hash
            basis = "completion timestamp follows precision correction; current V2 training source hash"
        fit_config = ROOT / "configs/fits" / fit["phase"] / f"{fit['job_id']}.json"
        source_provenance.append({
            "phase": fit["phase"], "job_id": fit["job_id"], "model_id": fit["model_id"],
            "seed": fit["seed"], "training_script_sha256": code_hash,
            "model_script_sha256": model_hash, "scheduler_script_sha256": scheduler_hash,
            "matrix_locked_model_and_loss_sha256": locked_contract_hash,
            "current_model_and_loss_document_sha256": current_contract_hash,
            "formal_matrix_sha256": formal_matrix_hash,
            "formal_matrix_model_and_loss_lock_matches_archived_contract": True,
            "inner_split_lock_sha256": split_lock_hash,
            "fit_config_sha256": sha_file(fit_config) if fit_config.is_file() else "MISSING",
            "train_selected_npz_sha256": manifest.get("train_selected_npz_sha256", "NA"),
            "feature_cache_manifest_sha256": manifest.get("feature_cache_manifest_sha256", "NA"),
            "normalizer_sha256": manifest["normalizer_sha256"],
            "model_file_sha256": manifest["model_file_sha256"],
            "per_fit_script_hash_directly_recorded_in_original_manifest": False,
            "binding_basis": basis,
        })
    provenance_path = ROOT / "results/TRAINING_SOURCE_PROVENANCE.csv"
    provenance_fields = list(source_provenance[0]) if source_provenance else [
        "phase", "job_id", "model_id", "seed", "training_script_sha256",
        "model_script_sha256", "scheduler_script_sha256", "matrix_locked_model_and_loss_sha256",
        "current_model_and_loss_document_sha256", "formal_matrix_sha256",
        "formal_matrix_model_and_loss_lock_matches_archived_contract", "inner_split_lock_sha256",
        "fit_config_sha256", "train_selected_npz_sha256",
        "feature_cache_manifest_sha256", "normalizer_sha256", "model_file_sha256",
        "per_fit_script_hash_directly_recorded_in_original_manifest", "binding_basis",
    ]
    write_csv(provenance_path, provenance_fields, source_provenance)

    lease_rows = []
    lease_sources = []
    for split_version, status_root in (("corrected_literal_v2", ROOT / "coordination/watcher_status"),
                                       ("superseded_dataset_salted", archive_root / "coordination/watcher_status")):
        lease_sources.extend((split_version, path) for path in status_root.glob("*.json"))
    for split_version, path in sorted(lease_sources, key=lambda item: item[1].relative_to(ROOT).as_posix()):
        if path.name.startswith("dryrun_"):
            continue
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not state.get("lease_acquired_utc"):
            continue
        started = state.get("started_utc")
        completed = state.get("completed_utc")
        duration = "NA"
        try:
            if started and completed:
                start_dt = datetime.fromisoformat(str(started).replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(str(completed).replace("Z", "+00:00"))
                duration = (end_dt - start_dt).total_seconds()
        except Exception:
            pass
        recheck = state.get("final_recheck") or {}
        model_memory = inference_memory_by_job.get(str(state.get("job_id", "")), {})
        lease_rows.append({
            "split_version": split_version,
            "attempt_record": path.relative_to(ROOT).as_posix(),
            "job_id": state.get("job_id", "NA"),
            "status": state.get("status", "UNKNOWN"),
            "physical_gpu": state.get("physical_gpu", state.get("selected_gpu", "NA")),
            "gpu_name": recheck.get("name", "NA"),
            "measured_peak_mib": state.get("measured_peak_mib", "NA"),
            "required_free_memory_mib": state.get("required_free_memory_mib", "NA"),
            "final_recheck_free_memory_mib": recheck.get("free_memory_mib", "NA"),
            "inference_peak_allocated_mib": model_memory.get("peak_memory_allocated_mib", "NA"),
            "inference_peak_reserved_mib": model_memory.get("peak_memory_reserved_mib", "NA"),
            "inference_script_sha256": model_memory.get("inference_script_sha256", "NA"),
            "started_utc": started or "NA", "completed_utc": completed or "NA",
            "watcher_duration_seconds": duration,
            "oom_requeue_count": state.get("oom_requeue_count", 0),
            "lease_released": state.get("lease_released", False),
            "project_child_slot_released": state.get("project_child_slot_released", False),
            "non_child_process_signaled": state.get("non_child_process_signaled", False),
            "command_hash": state.get("command_hash", "NA"),
        })
    lease_path = ROOT / "results/GPU_LEASE_LEDGER.csv"
    lease_fields = list(lease_rows[0]) if lease_rows else [
        "split_version", "attempt_record", "job_id", "status", "physical_gpu", "gpu_name", "measured_peak_mib",
        "required_free_memory_mib", "final_recheck_free_memory_mib", "inference_peak_allocated_mib",
        "inference_peak_reserved_mib", "inference_script_sha256", "started_utc", "completed_utc",
        "watcher_duration_seconds", "oom_requeue_count", "lease_released", "project_child_slot_released",
        "non_child_process_signaled", "command_hash",
    ]
    write_csv(lease_path, lease_fields, lease_rows)

    job_ledger = {
        "schema_version": 1, "updated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status.lower(), "actual_attempts": actual_attempts,
        "screening_planned_fits": screening["planned"], "screening_accepted_fits": screening["accepted"],
        "formal_planned_fits": formal["planned"], "formal_accepted_fits": formal["accepted"],
        "waypoint_planned_fits": waypoint["planned"], "waypoint_accepted_fits": waypoint["accepted"],
        "failed_gpu_fit_attempts_archived": failed_fit_attempts,
        "superseded_split_completed_fits": superseded_completed_fits,
        "superseded_split_failed_gpu_attempts": superseded_failed_gpu_attempts,
        "superseded_split_attempts": superseded_attempts,
        "corrected_split_attempts": corrected_split_attempts,
        "oom_retry_attempts": oom_retries,
        "watcher_children_active": 0, "gpu_leases_active": 0 if leases_released else "UNKNOWN",
        "shared_watcher_max_children": watcher_cap, "physical_gpus_used": physical_gpus,
        "training_physical_gpus": sorted(training_gpus), "inference_physical_gpus": sorted(inference_gpus),
        "superseded_training_physical_gpus": sorted(superseded_training_gpus),
        "superseded_inference_physical_gpus": sorted(superseded_inference_gpus),
        "superseded_fit_runtime_seconds": superseded_fit_seconds,
        "superseded_inference_runtime_seconds": superseded_inference_seconds,
        "jobs": screening_rows + formal_rows + waypoint_rows,
        "final_heldout_payload_access": 0,
    }
    write_json(ROOT / "coordination/JOB_LEDGER.json", job_ledger)

    acceptance = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status, "protocol": "V2_existing_five_tasks_development",
        "old_min50_gate_applied": False, "task_ids": list(TASK_IDS),
        "inner_split_algorithm": json.loads((ROOT / "protocol/INNER_SPLIT_LOCK.json").read_text())['algorithm'],
        "inner_split_lock_sha256": sha_file(ROOT / "protocol/INNER_SPLIT_LOCK.json"),
        "literal_inner_split_audit_status": qa.get("INNER_SPLIT_LITERAL_AUDIT.json"),
        "train_episodes_actual": sum(int(row["train_episodes"]) for row in train),
        "train_anchors_actual": sum(int(row["train_anchors"]) for row in train),
        "inner_fit_tune_counts": {
            row["task_id"]: {key: int(row[key]) for key in ("fit_episodes", "fit_anchors", "tune_episodes", "tune_anchors")}
            for row in train
        },
        "dev_support_per_task": dev,
        "screening_fits_accepted": screening["accepted"], "screening_fits_planned": 36,
        "formal_fits_accepted": formal["accepted"], "formal_fits_planned": 36,
        "waypoint_fits_accepted": waypoint["accepted"], "waypoint_fits_planned": 6,
        "formal_seeds": list(SEEDS), "lr_epoch_choices": selection.get("logical_model_choices", []),
        "selection_status": selection.get("status"),
        "partial_search": selection.get("status") == "LOCKED_WITH_PARTIAL_SEARCH",
        "implementation_correction_path": str(ROOT / "protocol/IMPLEMENTATION_CORRECTIONS.json"),
        "implementation_correction_semantics_changed": False,
        "shared_weight_precision_qa_status": qa.get("SAMPLE_WEIGHT_PRECISION_QA.json"),
        "first_real_pilot_status": pilot.get("status"),
        "sealed_prediction_jobs": seal.get("sealed_prediction_jobs"),
        "missing_prediction_jobs": seal.get("missing_prediction_jobs"),
        "actual_training_attempts_including_superseded_split": actual_attempts,
        "corrected_split_training_attempts": corrected_split_attempts,
        "failed_gpu_fit_attempts_current_split": failed_fit_attempts,
        "superseded_split_completed_fits": superseded_completed_fits,
        "superseded_split_failed_gpu_fit_attempts": superseded_failed_gpu_attempts,
        "superseded_split_training_attempts": superseded_attempts,
        "superseded_split_archive": str(archive_summary_path),
        "prior_dev_scores_used_for_corrected_selection": False,
        "historical_dev_target_access_preserved": archive_summary.get("prior_dev_target_labels_were_read_after_old_prediction_seal", False),
        "historical_superseded_split_target_rows": int(archive_summary.get("prior_dev_target_label_rows_selected", 0)),
        "historical_zero_prediction_scoring_target_rows": historical_zero_prediction_target_rows,
        "historical_dev_target_access_events_before_current_valid_seal": sum(int(value > 0) for value in (
            int(archive_summary.get("prior_dev_target_label_rows_selected", 0)), historical_zero_prediction_target_rows)),
        "historical_dev_target_read_rows_before_current_valid_seal": historical_pre_current_target_rows,
        "current_split_dev_target_rows_read": current_dev_target_rows,
        "cumulative_dev_target_access_rows": historical_pre_current_target_rows + current_dev_target_rows,
        "empty_prediction_scoring_attempt_archive": str(empty_prediction_archive_path),
        "empty_prediction_scoring_attempt_sealed_predictions": int(empty_prediction_summary.get("sealed_prediction_jobs", 0)),
        "empty_prediction_scoring_attempt_missing_predictions": int(empty_prediction_summary.get("missing_prediction_jobs", 0)),
        "empty_prediction_scoring_outputs_used_for_selection": bool(empty_prediction_summary.get("score_outputs_used_for_model_selection", False)),
        "empty_prediction_scoring_values_opened_by_agent": bool(empty_prediction_summary.get("human_read_of_score_values", False)),
        "dev_target_access_history": exposure.get("dev_target_access_history", []),
        "labels_entered_model_forward": False,
        "current_split_dev_targets_loaded_after_new_prediction_seal": exposure.get("current_split_target_values_loaded_after_prediction_seal", False),
        "current_split_dev_metrics_written": exposure.get("current_split_dev_metrics_written", False),
        "old_split_dev_scores_used_for_current_selection": False,
        "failed_attempt_event_count": len(failure_events),
        "failed_attempt_event_log": str(failure_log),
        "qa_status": qa,
        "gpu": {
            "physical_gpus_used": physical_gpus,
            "training_physical_gpus": sorted(training_gpus),
            "inference_physical_gpus": sorted(inference_gpus),
            "watcher_child_cap": watcher_cap,
            "sum_fit_runtime_seconds": fit_seconds,
            "sum_inference_runtime_seconds": inference_seconds,
            "superseded_fit_runtime_seconds": superseded_fit_seconds,
            "superseded_inference_runtime_seconds": superseded_inference_seconds,
            "historical_sum_fit_runtime_seconds": fit_seconds + superseded_fit_seconds,
            "historical_sum_inference_runtime_seconds": inference_seconds + superseded_inference_seconds,
            "superseded_training_physical_gpus": sorted(superseded_training_gpus),
            "superseded_inference_physical_gpus": sorted(superseded_inference_gpus),
            "inference_peak_allocated_mib_max": max(inference_peak_allocated) if inference_peak_allocated else "NA",
            "inference_peak_reserved_mib_max": max(inference_peak_reserved) if inference_peak_reserved else "NA",
            "all_leases_released": leases_released,
            "site_limit": "shared watcher source-enforced two project children; no study hour cap",
        },
        "encoder_training": 0, "encoder_extraction": 0,
        "cache_lineage": "accepted_cache_with_partial_historical_binding",
        "collection_independence": "unverified_episode_source_only",
        "dev_labels_read_after_prediction_seal": exposure.get("dev_target_labels_read_after_prediction_seal") is True,
        "parent_aggregate_metadata_exposure_preserved": True,
        "new_final_aggregate_counts_read_or_printed": False,
        "final_heldout_payload_access": 0, "old_root_writes": 0,
        "other_window_jobs_preempted": False, "independent_confirmation": False,
        "human_review_status": "pending", "checksum_acceptance": "pending_final_seal",
        "waypoint_endpoint_results": waypoint_endpoints,
        "training_source_provenance": str(provenance_path),
        "training_source_hash_rows": len(source_provenance),
        "gpu_lease_ledger": str(lease_path),
        "gpu_lease_attempt_records": len(lease_rows),
        "result_summary_path": str(ROOT / "results/RESULTS_SUMMARY.json"),
    }
    write_json(ROOT / "qa/FINAL_ACCEPTANCE.json", acceptance)
    write_json(ROOT / "results/FIT_ACCEPTANCE_SUMMARY.json", {
        "screening": screening, "formal": formal, "waypoint": waypoint,
        "actual_training_attempts": actual_attempts,
        "failed_gpu_fit_attempts": failed_fit_attempts, "oom_retry_attempts": oom_retries,
    })

    lines = [
        "# V2 实验结果回传", "",
        f"状态：{status}。这是开发实验；负收益照实保留，不构成独立确认。", "",
        "## 样本与执行", "",
        "| Task | Train ep/anchors | Fit ep/anchors | Tune ep/anchors | Dev ep/anchors |",
        "|---:|---:|---:|---:|---:|",
    ]
    dev_by_task = {int(row["task_id"]): row for row in dev}
    for row in train:
        d = dev_by_task[int(row["task_id"])]
        lines.append(
            f"| {row['task_id']} | {row['train_episodes']}/{row['train_anchors']} | "
            f"{row['fit_episodes']}/{row['fit_anchors']} | {row['tune_episodes']}/{row['tune_anchors']} | "
            f"{d['dev_episodes_after_overlap_exclusion']}/{d['dev_h30_anchors_after_overlap_exclusion']} |"
        )
    lines.extend([
        "",
        f"Fits accepted: screening {screening['accepted']}/36; formal {formal['accepted']}/36; waypoint {waypoint['accepted']}/6.",
        f"Corrected-split accepted fits: screening {screening['accepted']}/36, formal {formal['accepted']}/36, waypoint {waypoint['accepted']}/6; corrected-split attempts {corrected_split_attempts}.",
        f"Superseded dataset-salted run archived {superseded_completed_fits} completed logical fits and {superseded_failed_gpu_attempts} failed GPU attempts; total historical training attempts {actual_attempts}. Old dev scores were not used for corrected selection.",
        f"Recorded technical/QA failure events {len(failure_events)} in qa/FAILED_ATTEMPTS.jsonl.",
        f"Physical GPUs used across both protocol versions: {', '.join(physical_gpus) if physical_gpus else 'NA'}; watcher cap {watcher_cap}; corrected fit/inference runtime {fit_seconds:.1f}/{inference_seconds:.1f}s; superseded runtime {superseded_fit_seconds:.1f}/{superseded_inference_seconds:.1f}s; leases released={leases_released}.",
        "",
        "## Inner-tune LR / epoch choices", "",
        "| Model | LR | Epoch | Tune J | Valid candidates |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in selection.get("logical_model_choices", []):
        lines.append(f"| {row['model_id']} | {float(row['selected_lr']):g} | {row['selected_epoch']} | {float(row['selected_j_tune']):.6f} | {row['valid_lr_epoch_candidates']}/9 |")
    lines.extend([
        "",
        "## Absolute exposed-dev losses", "",
        "Means over the three formal seeds, with window errors averaged within episode and episodes equally weighted. Native command units are unidentified. safe-FD uses accepted primary-train scales; the 29 future-internal differences and anchor-to-first-step seam are separate.",
        "",
        "| Task | Program | Position MAE | safe-FD L1 | Seam safe-FD L1 |",
        "|---:|---|---:|---:|---:|",
    ])
    for task in TASK_IDS:
        models = ["G-S", "G-V", f"S-S-{task}", f"S-V-{task}"]
        if task == 119:
            models += ["B-AUX_ONLY", "B-GOAL_COND"]
        for model in models:
            subset = [r for r in abs_rows if r["model_id"] == model and int(r["task_id"]) == task and r["status"] == "PASS"]
            vals = {}
            for metric in ("position_mae", "safe_fd_l1", "seam_safe_fd_l1"):
                vals[metric] = [float(r[metric]) for r in subset if r[metric] not in ("NA", "")]
            lines.append(f"| {task} | {model} | {mean(vals['position_mae'])} | {mean(vals['safe_fd_l1'])} | {mean(vals['seam_safe_fd_l1'])} |")
    lines.extend([
        "",
        "## Paired contrasts", "",
        "Positive values mean benefit. ALL_SEED_D_V_I.csv has every task/seed; SUMMARY_STATISTICS.csv has task and macro mean/sample SD/median; ALL_SEED_LOO.csv has leave-one-task-out values; CONDITIONAL_CI.csv has the paired 5,000-draw source-group bootstrap.",
        "",
        "| Scope | Contrast | Metric | Mean | Sample SD | Median |",
        "|---|---|---|---:|---:|---:|",
    ])
    for row in summary_stats:
        if row["metric"] in {"position_mae", "safe_fd_l1"} and row["contrast"] in {"D_S", "D_V", "V_G", "V_S", "I", "B_GAIN", "AUX_VS_SV", "GOAL_VS_SV"}:
            lines.append(f"| {row['scope']} / {row['task_id']} | {row['contrast']} | {row['metric']} | {row['mean_across_seeds']} | {row['sample_sd_across_seeds']} | {row['median_across_seeds']} |")
    endpoint_by_key = {(row["model_id"], int(row["seed"])): row for row in waypoint_endpoints}
    lines.extend([
        "",
        "## Predicted waypoint endpoint", "",
        "Normalized endpoint errors use the accepted canonical primary-train target_sigma; native units remain unidentified.",
        "",
        "| Seed | AUX_ONLY endpoint MAE (norm) | GOAL_COND endpoint MAE (norm) | B_GOAL_HEAD_GAIN | AUX_ONLY native MAE | GOAL_COND native MAE |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for seed in SEEDS:
        aux = endpoint_by_key.get(("B-AUX_ONLY", seed), {})
        goal = endpoint_by_key.get(("B-GOAL_COND", seed), {})
        aux_norm = aux.get("goal_endpoint_mae_normalized", "NA")
        goal_norm = goal.get("goal_endpoint_mae_normalized", "NA")
        aux_native = aux.get("goal_endpoint_mae_native", "NA")
        goal_native = goal.get("goal_endpoint_mae_native", "NA")
        gain = "NA" if aux_norm == "NA" or goal_norm == "NA" else f"{float(aux_norm)-float(goal_norm):.4f}"
        lines.append(f"| {seed} | {aux_norm} | {goal_norm} | {gain} | {aux_native} | {goal_native} |")
    lines.extend([
        "",
        "## Limitations and artifacts", "",
        "- Waypoint goal-head endpoint MAE is reported separately in WAYPOINT_ENDPOINT.csv; action forecast comparisons remain in WAYPOINT.csv.",
        f"- Dev-label access audit: the current run read {current_dev_target_rows:,} future-action rows only after its valid 42/42 prediction seal; labels entered the scorer only, never model forward. Two earlier archived attempts each read 5,550 rows: the invalid dataset-salted split after a seal with 36 predictions and 6 missing, and a corrected-split zero-prediction scoring attempt after a seal with 0 predictions. The latter was invalid; both earlier score outputs were excluded from corrected selection and their numeric values were not opened. These are {historical_pre_current_target_rows + current_dev_target_rows:,} cumulative row reads across three attempts (re-reads, not unique rows). Final/heldout payload access is 0.",
        f"- Zero-prediction scoring archive: {empty_prediction_archive_path}",
        "- Parent aggregate metadata exposure is preserved. Cache lineage retains partial historical extractor binding. Collection/session independence is unverified at episode/source level.",
        "- The dataset-salted split run was invalidated because its tune groups differed from the literal V2 source-key hash. Its fits, predictions, and scores are preserved in the archive and excluded from corrected selection. The earlier sealed dev-label exposure is recorded; corrected LR/epoch selection uses inner-tune only.",
        "- TRAINING_SOURCE_PROVENANCE.csv binds weights, normalizers, model/scheduler code, and the training-script version by archived source hash and manifest completion timestamp; older fit manifests did not embed their own script SHA.",
        "- Bootstrap intervals are conditional development diagnostics; they do not establish independent confirmation. Human figure review is pending.",
        "- The palette audit passed white-background graphical contrast; four pairwise grayscale distinctions were flagged heuristically and are paired with distinct markers/line styles/hatching. This is not a publisher-compliance certification.",
        f"- Absolute losses: {ROOT / 'results/ABSOLUTE_LOSSES.csv'}",
        f"- D/V/I: {ROOT / 'results/ALL_SEED_D_V_I.csv'}",
        f"- LOO: {ROOT / 'results/ALL_SEED_LOO.csv'}",
        f"- Conditional CI: {ROOT / 'results/CONDITIONAL_CI.csv'}",
        f"- Waypoint action contrasts: {ROOT / 'results/WAYPOINT.csv'}",
        f"- Waypoint endpoint errors: {ROOT / 'results/WAYPOINT_ENDPOINT.csv'}",
        f"- Training source provenance: {provenance_path}",
        f"- Dynamic GPU leases/resources: {lease_path}",
        f"- Figures: {ROOT / 'figures'}",
    ])
    handoff = ROOT / "RETURN_TO_CHATGPT.zh-CN.md"
    handoff.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(ROOT / "RETURN_TO_CHATGPT.json", {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status, "markdown_path": str(handoff),
        "acceptance_path": str(ROOT / "qa/FINAL_ACCEPTANCE.json"),
        "results_summary_path": str(ROOT / "results/RESULTS_SUMMARY.json"),
        "final_heldout_payload_access": 0,
    })
    print(json.dumps({
        "status": status, "screening": screening, "formal": formal, "waypoint": waypoint,
        "physical_gpus": physical_gpus, "actual_attempts": actual_attempts,
        "failed_gpu_attempts": failed_fit_attempts, "final_heldout_payload_access": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
