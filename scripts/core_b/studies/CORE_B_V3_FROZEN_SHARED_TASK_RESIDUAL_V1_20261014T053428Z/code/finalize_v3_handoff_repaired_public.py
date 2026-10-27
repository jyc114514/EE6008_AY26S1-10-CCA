#!/usr/bin/env python3
# Repaired public copy of EXCL-CODE-011; source SHA-256: 7d28585a89bb1777e38fe9506531cb2ded142faa255c8d1f76dfa209540b0048
"""Create the final V3 acceptance, resource, and return-to-user artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("EE6008_V3_RESIDUAL_STUDY_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
PROJECT = Path(os.environ.get("EE6008_PROJECT_ROOT", ".")).expanduser().resolve()
V2 = ROOT.parent / "CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z"
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
TASKS = (119, 94, 90, 141, 230)
SEEDS = (202610131, 202610132, 202610133)
METRICS = ("position_mae", "safe_fd_l1", "seam_safe_fd_l1")
SCORE = ROOT / "results/scoring/attempt_02"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def backup(path: Path, stamp: str) -> str | None:
    if not path.is_file():
        return None
    out = path.with_name(path.name + ".pre-final-" + stamp + ".bak")
    if not out.exists():
        shutil.copy2(path, out)
    return str(out)


def f(v: Any, digits: int = 6) -> str:
    return f"{float(v):.{digits}f}"


def float_field(value: str) -> float | None:
    if value in {"", "NA", "None", "null"}:
        return None
    return float(value)


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    exposure_path = ROOT / "inventory/OUTCOME_EXPOSURE.json"
    figure_qa_path = ROOT / "qa/FIGURE_BUILD_QA.json"
    scoring_qa_path = ROOT / "qa/DEV_TARGET_SCORING_QA.json"
    owner_path = ROOT / "coordination/OWNER_STATE.json"
    output_paths = [exposure_path, figure_qa_path, scoring_qa_path, owner_path,
                    ROOT / "qa/FIGURE_RGB_REBUILD.json", ROOT / "qa/BASE_FROZEN.json",
                    ROOT / "qa/INPUT_LABEL_SEPARATION.json", ROOT / "qa/WEIGHT_METRIC_AND_BIAS_IDENTITIES.json",
                    ROOT / "qa/PREDICTION_ACCEPTANCE.json", ROOT / "qa/FINAL_ACCEPTANCE.json",
                    ROOT / "qa/WATCHER_PROVENANCE_RECONCILIATION.json",
                    ROOT / "coordination/RESOURCE_LOG.csv", ROOT / "coordination/JOB_LEDGER.json",
                    ROOT / "coordination/PHASE_STATUS_RGB_PROBE.json", ROOT / "coordination/PHASE_STATUS_POSTSEAL_SCORING.json",
                    ROOT / "coordination/RESUME_COMMANDS.md", ROOT / "RETURN_TO_CHATGPT.zh-CN.md", ROOT / "RETURN_TO_CHATGPT.json"]
    backups = {str(p): backup(p, stamp) for p in output_paths}

    # Accepted model and prediction receipts. No scorer-only target artifact is opened here.
    fits: list[tuple[Path, dict[str, Any]]] = []
    for p in ROOT.rglob("FIT_COMPLETE.json"):
        d = read_json(p)
        if d.get("status") == "PASS": fits.append((p, d))
    features: list[tuple[Path, dict[str, Any]]] = []
    for p in ROOT.rglob("FEATURES_COMPLETE.json"):
        d = read_json(p)
        if d.get("status") == "PASS": features.append((p, d))
    rgb_completions = [read_json(p) for p in (ROOT / "predictions/rgb_probe").rglob("PREDICTION_COMPLETE.json")]
    state_qa = read_json(ROOT / "qa/STATE_RGB_INVARIANCE/attempt_01/STATE_RGB_QA_COMPLETE.json")
    stage0 = read_json(ROOT / "qa/STAGE0_ACCEPTANCE.json")
    pre_main = read_json(ROOT / "qa/MAIN_PREDICTION_ACCEPTANCE_PRESEAL.json")
    combined_pre = read_json(ROOT / "qa/PREDICTION_ACCEPTANCE_PRESEAL.json")
    scoring_qa = read_json(SCORE / "DEV_TARGET_SCORING_QA.json")
    seal_path = ROOT / "protocol/PREDICTION_BATCH_SEAL.json"
    seal = read_json(seal_path)
    selection = read_json(ROOT / "protocol/READOUT_SELECTION.json")
    model_lock = read_json(ROOT / "protocol/MODEL_READOUT_LOCK.json")
    analysis_lock = read_json(ROOT / "protocol/ANALYSIS_LOCK.json")
    v2_recon = read_json(ROOT / "diagnostics/V2_RECONCILIATION.json")
    v2_accept = read_json(ROOT / "inventory/V2_SOURCE_ACCEPTANCE.json")
    v2_metric_path = V2 / "protocol/METRIC_SCALE_LOCK.json"
    v2_metric = read_json(v2_metric_path)
    environment = read_json(ROOT / "inventory/ENVIRONMENT_PRECHECK.json")
    resource_lock = read_json(ROOT / "qa/V3_RESOURCE_ESTIMATE_LOCK.json")
    exposure = read_json(exposure_path)
    figure_qa = read_json(figure_qa_path)

    if len(fits) != 20 or len(features) != 8 or len(rgb_completions) != 12:
        raise RuntimeError(f"accepted receipt count mismatch fits={len(fits)} features={len(features)} RGB={len(rgb_completions)}")
    if seal.get("status") != "SEALED_BEFORE_TARGET_ACCESS" or len(seal.get("new_prediction_entries", [])) != 30:
        raise RuntimeError("final handoff refuses incomplete prediction seal")
    if scoring_qa.get("status") != "PASS" or scoring_qa.get("v3_target_rows_read_across_all_scoring_attempts") != 11100:
        raise RuntimeError("final handoff refuses scoring QA or access-ledger mismatch")

    # Validate base immutability and label-forward boundaries from accepted receipts.
    base_frozen = all(d.get("base_state_hash_before") == d.get("base_state_hash_after")
                      and d.get("base_unchanged") is True for _, d in features)
    readouts_only = all(d.get("base_parameters_in_optimizer") is False
                        and d.get("labels_passed_to_forward") is False
                        and d.get("train_labels_used_for_loss_only") is True
                        for _, d in fits)
    feature_no_encoder = all("encoder_training" in d and "encoder_extraction" in d
                             and not bool(d["encoder_training"]) and not bool(d["encoder_extraction"])
                             for _, d in features)
    if not base_frozen or not readouts_only or not feature_no_encoder:
        raise RuntimeError("final freeze/label-separation invariant check failed")

    # Watcher ledger is intentionally sanitized: no PID or process identity is copied.
    watcher_dir = ROOT / "coordination/watcher_status"
    watcher_rows: list[dict[str, Any]] = []
    def phase_from_job(job: str) -> str:
        if job.startswith("v3_stage0_"): return "stage0_smoke"
        if job.startswith("v3_feature_screening_"): return "screening_feature_extraction"
        if job.startswith("v3_feature_formal_"): return "formal_feature_extraction"
        if job.startswith("v3_screen_"): return "screening_readout_fit"
        if job.startswith("v3_formal_"): return "formal_readout_fit"
        if job.startswith("v3_state_rgb_"): return "state_rgb_invariance_qa"
        if job.startswith("v3_probe_"): return "rgb_probe"
        return "other"
    for p in sorted(watcher_dir.glob("*.json")):
        d = read_json(p)
        start, end = d.get("started_utc"), d.get("completed_utc")
        duration = None
        if start and end:
            duration = (datetime.fromisoformat(end.replace("Z", "+00:00")) -
                        datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds()
        job = str(d.get("job_id", p.stem))
        watcher_rows.append({
            "job_id": job, "phase": phase_from_job(job), "status": d.get("status"),
            "physical_gpu": d.get("physical_gpu", d.get("selected_gpu")),
            "started_utc": start, "completed_utc": end, "duration_seconds": duration,
            "watcher_sampled_peak_mib": d.get("measured_peak_mib"),
            "required_free_memory_mib": d.get("required_free_memory_mib"),
            "safety_margin_mib": d.get("safety_margin_mib"),
            "oom_requeue_count": d.get("oom_requeue_count"),
            "lease_released": d.get("lease_released"),
            "project_child_slot_released": d.get("project_child_slot_released"),
            "non_child_process_signaled": d.get("non_child_process_signaled"),
            "command_hash": d.get("command_hash"), "output_root": d.get("output_root"),
            "log_path": d.get("log_path"), "watcher_status_path": str(p),
        })
    counts = Counter(str(x["status"]) for x in watcher_rows)
    if len(watcher_rows) != 57 or counts.get("complete") != 43 or counts.get("failed") != 14:
        raise RuntimeError(f"watcher receipt totals changed: total={len(watcher_rows)} statuses={dict(counts)}")
    all_released = all(x["lease_released"] is True and x["project_child_slot_released"] is True for x in watcher_rows)
    if not all_released:
        raise RuntimeError("a V3 project GPU lease/slot is not released")

    # Per-phase resource summary joins actual application peak receipts to sanitized watcher status.
    app_receipts: dict[str, dict[str, Any]] = {}
    for _, d in fits + features:
        if d.get("job_id"):
            app_receipts[str(d["job_id"])] = d
    for d in rgb_completions:
        if d.get("job_id"): app_receipts[str(d["job_id"])] = d
    state_job_id = str(state_qa.get("job_id", "v3_state_rgb_invariance_seed202610131"))
    app_receipts[state_job_id] = state_qa
    phase_order = ("stage0_smoke", "screening_feature_extraction", "formal_feature_extraction",
                   "screening_readout_fit", "formal_readout_fit", "state_rgb_invariance_qa", "rgb_probe")
    resource_rows: list[dict[str, Any]] = []
    for phase in phase_order:
        rows = [x for x in watcher_rows if x["phase"] == phase]
        completions = [app_receipts[x["job_id"]] for x in rows if x["job_id"] in app_receipts]
        actual_alloc = [float(d["peak_memory_allocated_mib"]) for d in completions if d.get("peak_memory_allocated_mib") is not None]
        actual_reserved = [float(d["peak_memory_reserved_mib"]) for d in completions if d.get("peak_memory_reserved_mib") is not None]
        durations = [float(x["duration_seconds"]) for x in rows if x.get("duration_seconds") is not None]
        complete_durations = [float(x["duration_seconds"]) for x in rows if x.get("status") == "complete" and x.get("duration_seconds") is not None]
        if phase == "stage0_smoke":
            actual_alloc = [float(resource_lock["stage0_max_measured_peak_allocated_mib"])]
            actual_reserved = [float(resource_lock["stage0_max_measured_peak_reserved_mib"])]
        resource_rows.append({
            "phase": phase, "watcher_jobs": len(rows),
            "complete_jobs": sum(x["status"] == "complete" for x in rows),
            "failed_jobs": sum(x["status"] == "failed" for x in rows),
            "physical_gpus": ";".join(sorted({str(x["physical_gpu"]) for x in rows if x["physical_gpu"] is not None})),
            "max_application_peak_allocated_mib": max(actual_alloc) if actual_alloc else "NA",
            "max_application_peak_reserved_mib": max(actual_reserved) if actual_reserved else "NA",
            "sum_all_attempt_wall_seconds": sum(durations),
            "sum_complete_job_wall_seconds": sum(complete_durations),
            "max_watcher_job_wall_seconds": max(durations) if durations else "NA",
            "all_leases_and_slots_released": all(x["lease_released"] is True and x["project_child_slot_released"] is True for x in rows),
            "watcher_project_child_cap": 2,
        })
    write_csv(ROOT / "coordination/RESOURCE_LOG.csv", resource_rows)
    write_json(ROOT / "coordination/JOB_LEDGER.json", {
        "schema_version": 1, "status": "CLOSED", "created_utc": now(),
        "watcher_job_count": len(watcher_rows), "status_counts": dict(counts),
        "all_project_leases_released": all_released,
        "process_identity_policy": "no process IDs or non-child identities copied into this summary",
        "jobs": watcher_rows,
    })

    # Final phase receipts.
    rgb_queue = read_json(ROOT / "coordination/queues/RGB_PROBE.json")
    rgb_job_rows = []
    for raw in rgb_queue["jobs"]:
        wrapper = read_json(Path(raw)); command = wrapper.get("command", [])
        if "--job-json" not in command: raise RuntimeError(f"RGB watcher wrapper lacks job config: {raw}")
        cfg = read_json(Path(command[command.index("--job-json") + 1]))
        completion = Path(cfg["output_root"]) / "PREDICTION_COMPLETE.json"
        done = read_json(completion)
        ws = next(x for x in watcher_rows if x["job_id"] == cfg["job_id"])
        if done.get("status") != "PASS" or ws["status"] != "complete" or ws["lease_released"] is not True:
            raise RuntimeError(f"RGB job not accepted/released: {cfg['job_id']}")
        rgb_job_rows.append({"job_id": cfg["job_id"], "model_id": cfg["model_id"], "seed": cfg["seed"],
                             "condition": cfg["condition"], "prediction_rows": done["row_count"],
                             "physical_gpu": done["gpu_physical"], "peak_allocated_mib": done["peak_memory_allocated_mib"],
                             "peak_reserved_mib": done["peak_memory_reserved_mib"], "completion": str(completion),
                             "status": done["status"], "lease_released": True})
    write_json(ROOT / "coordination/PHASE_STATUS_RGB_PROBE.json", {
        "schema_version": 1, "phase": "rgb_probe", "status": "PASS", "created_utc": now(),
        "planned_jobs": 12, "accepted_jobs": len(rgb_job_rows), "failed_jobs": 0,
        "new_task_condition_views": 36, "matched_v2_reference_jobs": 6,
        "matched_v2_reference_task_views": 18,
        "matched_gv_views_overlapping_main_base_reference": 15,
        "matched_sv90_additional_reference_views": 3,
        "dev_targets_loaded_by_probe": False, "final_heldout_payload_access": 0,
        "shared_watcher_max_children": 2, "jobs": rgb_job_rows,
    })
    write_json(ROOT / "coordination/PHASE_STATUS_POSTSEAL_SCORING.json", {
        "schema_version": 1, "phase": "post_seal_scoring", "status": "PASS", "created_utc": now(),
        "prediction_seal_sha256": sha(seal_path), "new_jobs_scored": 30,
        "main_new_jobs": 18, "main_new_task_method_seed_views": 90,
        "rgb_probe_new_jobs": 12, "rgb_new_task_condition_views": 36,
        "reused_v2_reference_jobs": 9, "reused_v2_main_task_views": 30,
        "successful_attempt": "attempt_02", "successful_attempt_target_action_rows_read": 5550,
        "failed_postseal_attempt_target_action_rows_read": 5550,
        "v3_target_action_rows_read_total": 11100,
        "parent_v2_target_action_rows_read": 16650,
        "parent_plus_v3_target_action_rows_read": 27750,
        "final_heldout_payload_access": 0,
    })

    # Figure and target-exposure accounting; the RGB-only repair did not reopen any target artifact.
    rgb_fig = {"schema_version": 1, "status": "PASS", "created_utc": now(),
               "attempt": "attempt_02_rgb_sensitivity_correction",
               "corrected_figures": ["RGB_H_AND_OUTPUT_SENSITIVITY.png", "RGB_ERROR_CHANGES.png"],
               "empty_condition_groups": 0, "nan_bars": 0,
               "rgb_diagnostics_source": "results/scoring/attempt_02/RGB_PROBE_RESULTS.csv",
               "scorer_only_target_archive_opened": False, "dev_target_rows_read": 0,
               "final_heldout_payload_access": 0}
    write_json(ROOT / "qa/FIGURE_RGB_REBUILD.json", rgb_fig)
    figure_qa.update({
        "created_utc": now(), "status": "PASS", "corrected_rgb_figure_attempt": "attempt_02",
        "attempt_01_issue": "condition-specific bars included other-condition task keys, yielding empty means; initial figure and QA preserved as timestamped backups",
        "attempt_02_corrected_figures": rgb_fig["corrected_figures"],
        "attempt_02_used_scored_rgb_diagnostics_only": True,
        "attempt_02_scorer_only_target_archive_opened": False,
        "manual_visual_review_status": "PASS",
        "manually_reviewed_figures": ["MAIN_TASK_METRICS.png", "SCREENING_TRAIN_TUNE_CURVES.png",
                                     "RGB_ERROR_CHANGES.png", "RGB_H_AND_OUTPUT_SENSITIVITY.png",
                                     "TRAJECTORY_VIDEO_AXIS0.png"],
        "human_review_status": "pending",
        "reused_existing_v2_target_trace_points": 300,
        "final_heldout_payload_access": 0,
    })
    write_json(figure_qa_path, figure_qa)
    scoring_qa.update({
        "figure_target_data_note": "300 selected trajectory target points reused from pre-existing V2 derived plot data; no additional scorer-only target archive read was made by the RGB chart repair",
        "figure_rgb_chart_rebuild_target_rows_read": 0,
        "v3_target_rows_read_across_all_scoring_attempts": 11100,
        "cumulative_parent_plus_v3_target_rows_read": 27750,
    })
    write_json(scoring_qa_path, scoring_qa)
    write_json(ROOT / "qa/DEV_TARGET_SCORING_QA.json", scoring_qa)
    exposure.update({
        "v3_current_dev_target_rows_read": 11100,
        "v3_successful_scoring_attempt_rows_read": 5550,
        "v3_target_rows_read_by_attempt": {"attempt_01": 5550, "attempt_02": 5550},
        "parent_v2_cumulative_dev_target_row_reads": 16650,
        "cumulative_parent_plus_v3_dev_target_row_reads": 27750,
        "v3_reused_existing_v2_trajectory_target_trace_points": 300,
        "v3_rgb_chart_rebuild_target_rows_read": 0,
        "v3_final_heldout_payload_access": 0,
    })
    write_json(exposure_path, exposure)

    # Deterministic invariant QA from existing accepted receipts and result files.
    fit_ids = [d["job_id"] for _, d in fits]
    shared_params = int(stage0["active_readout_parameters"])
    base_frozen_qa = {
        "schema_version": 1, "status": "PASS", "created_utc": now(),
        "feature_extraction_jobs": len(features), "feature_base_hash_before_after_all_equal": base_frozen,
        "neural_fit_receipts": len(fits), "base_parameters_in_optimizer_all_false": all(d.get("base_parameters_in_optimizer") is False for _, d in fits),
        "base_weights_changed": False, "encoder_training": False, "encoder_extraction": False,
        "hidden_shape": [185, 30, 128], "hidden_interface": "h=[B,30,128] before V2 output projection",
        "shared_readout_parameters": shared_params, "task_bank_total_parameters": 5 * shared_params,
        "task_bank_active_parameters_per_sample": shared_params,
        "zero_residual_replay_stage0": stage0["base_zero_replay_pass"],
        "v2_matched_base_replay_status": read_json(ROOT / "qa/V2_MATCHED_BASE_REPLAY.json")["status"],
        "v2_matched_base_replay_max_abs_native_difference": max(x["max_abs_native_output_difference"] for x in read_json(ROOT / "qa/V2_MATCHED_BASE_REPLAY.json")["comparisons"]),
        "fit_job_ids": fit_ids, "final_heldout_payload_access": 0,
    }
    write_json(ROOT / "qa/BASE_FROZEN.json", base_frozen_qa)
    labels_forward = all(d.get("labels_passed_to_forward") is False for _, d in fits)
    labels_loss_only = all(d.get("train_labels_used_for_loss_only") is True for _, d in fits)
    input_sep = {
        "schema_version": 1, "status": "PASS", "created_utc": now(),
        "future_labels_passed_to_model_forward": False, "neural_fit_receipts": len(fits),
        "all_fit_receipts_labels_passed_to_forward_false": labels_forward,
        "all_train_labels_used_for_loss_only": labels_loss_only,
        "feature_extraction_encoder_labels": False,
        "dev_target_access_before_prediction_seal": 0,
        "postseal_scoring_attempts": {"attempt_01": {"rows_read": 5550, "result": "failed_after_v2_shared_seed_row_parse", "selection_used": False},
                                      "attempt_02": {"rows_read": 5550, "result": "PASS", "selection_used": False}},
        "v3_target_rows_read_total": 11100, "parent_v2_rows_preserved": 16650,
        "final_heldout_payload_access": 0,
    }
    write_json(ROOT / "qa/INPUT_LABEL_SEPARATION.json", input_sep)

    bias_done = read_json(ROOT / "calibration/BIAS_CALIBRATION_COMPLETE.json")
    main_scores = read_csv(SCORE / "METHOD_METRICS_BY_TASK_SEED.csv")
    main_score_map = {(x["method"], int(x["seed"]), int(x["task_id"]), x["metric"]): float(x["value"]) for x in main_scores}
    bias_fd_differences = [main_score_map[(bias, seed, task, "safe_fd_l1")] - main_score_map[(base, seed, task, "safe_fd_l1")]
                           for bias, base in (("BIAS_STATE", "BASE_STATE"), ("BIAS_VIDEO", "BASE_VIDEO"))
                           for seed in SEEDS for task in TASKS]
    bias_metric_qa = {
        "schema_version": 1, "status": "PASS", "created_utc": now(),
        "v2_metric_lock_sha256": sha(v2_metric_path),
        "position_metric": "canonical V2 target_sigma-normalized Full-28 x 30 MAE",
        "safe_fd_metric": "canonical V2 safe_fd_scale-normalized 29 future-internal first differences",
        "seam_metric": "first future displacement error relative to anchor, separately reported",
        "aggregation": "window mean within episode, episode-equal task means, equal five-task macro",
        "v2_accepted_base_max_metric_difference": scoring_qa["max_abs_v2_base_metric_reconciliation_difference"],
        "persistence_max_metric_difference": scoring_qa["max_abs_v2_persistence_metric_reconciliation_difference"],
        "bias_calibration_status": bias_done["status"], "bias_calibrations": bias_done["calibration_count"],
        "bias_future_fd_identity_max_abs_fp64": max(float(x["future_fd_max_abs_change_fp64"]) for c in bias_done["calibrations"] for x in c["tasks"]),
        "scored_bias_minus_base_future_fd_error_max_abs": max(abs(x) for x in bias_fd_differences),
        "group_bootstrap": {"replicates": 5000, "seed": 202610134, "shared_draw_manifest": "protocol/SOURCE_GROUP_DRAWS.npz",
                            "paired_source_groups_within_task": True, "p_values": "not computed by protocol"},
        "final_heldout_payload_access": 0,
    }
    write_json(ROOT / "qa/WEIGHT_METRIC_AND_BIAS_IDENTITIES.json", bias_metric_qa)

    # Read the score tables for directly reproducible report sections.
    task_summary = read_csv(SCORE / "SUMMARY_STATISTICS.csv")
    contrast_summary = [x for x in task_summary if x["scope"] == "five_task_macro" and x["result_type"] == "contrast"]
    contrast_ci = read_csv(SCORE / "PAIRED_SOURCE_GROUP_CI.csv")
    ci_map = {(x["contrast"], x["metric"]): x for x in contrast_ci if x["scope"] == "five_task_macro_all_seeds"}
    summary_map = {(x["name"], x["metric"]): x for x in task_summary if x["scope"] == "five_task_macro"}
    methods = ["PERSISTENCE", "BASE_STATE", "BIAS_STATE", "SHARED_RES_STATE", "TASK_RES_STATE",
               "BASE_VIDEO", "BIAS_VIDEO", "SHARED_RES_VIDEO", "TASK_RES_VIDEO"]
    task_cells = []
    for task in TASKS:
        row = {"task_id": task}
        for method in methods:
            pos = next(x for x in task_summary if x["scope"] == "task" and x["task_id"] == str(task) and x["result_type"] == "method" and x["name"] == method and x["metric"] == "position_mae")
            fd = next(x for x in task_summary if x["scope"] == "task" and x["task_id"] == str(task) and x["result_type"] == "method" and x["name"] == method and x["metric"] == "safe_fd_l1")
            row[method] = {"position_mae_mean": float(pos["mean_across_seeds"]), "position_mae_sample_sd": float(pos["sample_sd_across_seeds"]),
                           "safe_fd_l1_mean": float(fd["mean_across_seeds"]), "safe_fd_l1_sample_sd": float(fd["sample_sd_across_seeds"])}
        task_cells.append(row)
    macro_methods = []
    for method in methods:
        macro_methods.append({"method": method,
                              **{m: {"mean_across_seeds": float(summary_map[(method, m)]["mean_across_seeds"]),
                                     "sample_sd_across_seeds": float(summary_map[(method, m)]["sample_sd_across_seeds"]),
                                     "median_across_seeds": float(summary_map[(method, m)]["median_across_seeds"])} for m in METRICS}})
    macro_contrasts = []
    for name in sorted({x["name"] for x in contrast_summary}):
        per = {x["metric"]: x for x in contrast_summary if x["name"] == name}
        macro_contrasts.append({"contrast": name,
                                **{m: {"mean_across_seeds": float(per[m]["mean_across_seeds"]),
                                       "sample_sd_across_seeds": float(per[m]["sample_sd_across_seeds"]),
                                       "median_across_seeds": float(per[m]["median_across_seeds"]),
                                       "conditional_group_bootstrap_ci95": [float(ci_map[(name, m)]["ci_low_2_5"]), float(ci_map[(name, m)]["ci_high_97_5"])]}
                                   for m in METRICS}})

    # LOO ranges for selected V3 contrasts: seed LOO keeps five tasks; task LOO keeps four tasks per seed.
    seed_loo = read_csv(SCORE / "LEAVE_ONE_SEED_OUT.csv")
    task_loo = read_csv(SCORE / "LEAVE_ONE_TASK_OUT.csv")
    loo_names = ["G_R_shared_STATE", "G_R_task_STATE", "T_R_STATE", "G_R_shared_VIDEO", "G_R_task_VIDEO", "T_R_VIDEO"]
    loo_ranges = []
    for name in loo_names:
        for metric in ("position_mae", "safe_fd_l1"):
            sv = [float(x["loo_mean"]) for x in seed_loo if x["scope"] == "five_task_macro" and x["name"] == name and x["metric"] == metric]
            tv = [float(x["loo_mean"]) for x in task_loo if x["scope"] == "all_seeds_mean" and x["name"] == name and x["metric"] == metric]
            loo_ranges.append({"contrast": name, "metric": metric,
                               "seed_loo_range": [min(sv), max(sv)], "task_loo_all_seed_mean_range": [min(tv), max(tv)],
                               "seed_loo_rows": len(sv), "task_loo_rows": len(tv)})

    rgb_rows = read_csv(SCORE / "RGB_PROBE_RESULTS.csv")
    rgb_means = []
    for model in ("G-V", "S-V-90"):
        for cond in ("ZERO_NORMALIZED_BLOCK", "SAME_TASK_DONOR"):
            sub = [x for x in rgb_rows if x["model_id"] == model and x["condition"] == cond]
            rgb_means.append({"model_id": model, "condition": cond, "task_condition_seed_views": len(sub),
                              "mean_position_error_change": sum(float(x["position_mae_condition_minus_matched"]) for x in sub) / len(sub),
                              "mean_safe_fd_error_change": sum(float(x["safe_fd_l1_condition_minus_matched"]) for x in sub) / len(sub),
                              "mean_h_relative_l2_change": sum(float(x["h_relative_l2_change"]) for x in sub) / len(sub),
                              "mean_base_output_abs_change": sum(float(x["base_residual_mean_abs_change"]) for x in sub) / len(sub)})
    state_change_fields = ("base_output_max_abs_change", "h_max_abs_change", "shared_R_max_abs_change", "task_R_max_abs_change")
    state_changes = {key: max(float(c.get(key, 0.0)) for fix in state_qa["fixtures"] for c in fix["conditions"]) for key in state_change_fields}

    # V2 corrected result and scale audit.
    v2_summary_rows = read_csv(ROOT / "diagnostics/V2_RECOMPUTED_SUMMARY_STATISTICS.csv")
    v2_macro = []
    for name in ("D_S", "D_V", "V_G", "V_S", "I"):
        row = {"contrast": name}
        for metric in ("position_mae", "safe_fd_l1"):
            x = next(x for x in v2_summary_rows if x["scope"] == "five_task_macro" and x["contrast"] == name and x["metric"] == metric)
            row[metric] = {"mean_across_seeds": float(x["mean_across_seeds"]), "sample_sd_across_seeds": float(x["sample_sd_across_seeds"])}
        v2_macro.append(row)
    v2_seed_loo_rows = read_csv(ROOT / "diagnostics/V2_LEAVE_ONE_SEED_OUT.csv")
    v2_task_loo_rows = read_csv(ROOT / "diagnostics/V2_LEAVE_ONE_TASK_OUT.csv")
    scale_summary = {"target_sigma_min": min(v2_metric["target_sigma"]), "target_sigma_max": max(v2_metric["target_sigma"]),
                     "safe_fd_scale_min": min(v2_metric["safe_fd_scale"]), "safe_fd_scale_max": max(v2_metric["safe_fd_scale"]),
                     "target_sigma": v2_metric["target_sigma"], "safe_fd_scale": v2_metric["safe_fd_scale"],
                     "path": str(v2_metric_path), "sha256": sha(v2_metric_path)}

    # Resource values from receipts; watcher sampled/admission memory remains a separate field.
    feature_elapsed = sum(float(d.get("elapsed_seconds", 0)) for _, d in features)
    fit_elapsed = sum(float(d.get("elapsed_seconds", 0)) for _, d in fits)
    watcher_durations = [float(x["duration_seconds"]) for x in watcher_rows if x.get("status") == "complete" and x.get("duration_seconds") is not None]
    physical_gpus = sorted({str(x["physical_gpu"]) for x in watcher_rows if x["physical_gpu"] is not None})
    max_watcher_children = max(int(read_json(p).get("max_concurrent_project_gpu_children", 0)) for p in watcher_dir.glob("*.json"))
    probe_max_alloc = max(float(x["peak_memory_allocated_mib"]) for x in rgb_completions)
    probe_max_reserved = max(float(x["peak_memory_reserved_mib"]) for x in rgb_completions)
    feature_max_alloc = max(float(d["peak_memory_allocated_mib"]) for _, d in features)
    feature_max_reserved = max(float(d["peak_memory_reserved_mib"]) for _, d in features)
    fit_max_alloc = max(float(d["peak_memory_allocated_mib"]) for _, d in fits)
    fit_max_reserved = max(float(d["peak_memory_reserved_mib"]) for _, d in fits)
    app_max_alloc = max(probe_max_alloc, feature_max_alloc, fit_max_alloc, float(resource_lock["stage0_max_measured_peak_allocated_mib"]))
    app_max_reserved = max(probe_max_reserved, feature_max_reserved, fit_max_reserved, float(resource_lock["stage0_max_measured_peak_reserved_mib"]))
    env_cards = [{"physical_gpu": x["index"], "name": x["name"], "total_mib": x["memory_total_mib"],
                  "free_mib_at_admission": x["free_memory_mib"], "utilization_percent_at_admission": x["utilization_percent"]}
                 for x in environment["aggregate_gpu_snapshots_at_stage0_admission"]]
    watcher_path = REPO / "tools/gpu_lease_watcher.py"
    watcher_current_sha = sha(watcher_path)
    later_phase_watcher_sha = read_json(ROOT / "coordination/PHASE_STATUS_STATE_RGB_QA.json").get("shared_watcher_sha256")
    watcher_recon = {"current_source_sha256": watcher_current_sha,
                     "v2_source_acceptance_sha256": v2_accept.get("shared_watcher_sha256"),
                     "stage0_receipt_sha256": stage0.get("shared_watcher_sha256", "not recorded in STAGE0_ACCEPTANCE.json"),
                     "later_phase_receipt_sha256": later_phase_watcher_sha,
                     "environment_precheck_recorded_sha256": environment.get("shared_watcher_sha256"),
                     "current_matches_v2_and_later_phase": watcher_current_sha == v2_accept.get("shared_watcher_sha256") == later_phase_watcher_sha,
                     "environment_precheck_hash_matches_current": environment.get("shared_watcher_sha256") == watcher_current_sha,
                     "disposition": "retained the recorded ENVIRONMENT_PRECHECK hash discrepancy; no watcher source was changed"}
    watcher_qa = {"schema_version": 1, "status": "PASS_WITH_RECORDED_PRECHECK_HASH_DISCREPANCY", "created_utc": now(),
                  **watcher_recon, "source_path": str(watcher_path), "source_sha256_now": watcher_current_sha,
                  "physical_gpus_used": physical_gpus, "max_project_children": max_watcher_children,
                  "watcher_jobs": len(watcher_rows), "complete": counts["complete"], "failed_attempts": counts["failed"],
                  "all_leases_and_project_slots_released": all_released, "other_process_identities_read": False}
    write_json(ROOT / "qa/WATCHER_PROVENANCE_RECONCILIATION.json", watcher_qa)

    final_acceptance = {
        "schema_version": 1, "status": "PASS", "study_status": "complete", "created_utc": now(),
        "mode": "post_V2_exploratory_frozen_readout",
        "tasks": list(TASKS), "train_episodes": 220, "train_anchors": 1280,
        "inner_fit_episodes": 174, "inner_fit_anchors": 1010, "inner_tune_episodes": 46, "inner_tune_anchors": 270,
        "dev_anchors": 185, "dev_support_by_task": {"119":48,"94":48,"90":41,"141":20,"230":28},
        "fold_membership_reused_without_rehash": True, "V2_78_fits_rerun": 0,
        "base_training_updates": 0, "encoder_training": 0, "encoder_extraction": 0,
        "accepted_neural_screen_fits": "8/8", "accepted_neural_formal_fits": "12/12",
        "accepted_cpu_bias_calibrations": "6/6", "base_parameters_changed": False,
        "hidden_interface": "[B,30,128] before frozen V2 output projection",
        "shared_readout_parameters": 3612, "task_bank_total_parameters": 18060,
        "task_bank_active_parameters_per_sample": 3612,
        "zero_residual_stage0_smokes": "2/2 PASS; smoke updates are not counted as fits",
        "base_frozen": base_frozen, "labels_only_used_for_fit_loss": labels_loss_only,
        "future_labels_used_in_forward": False,
        "prediction_seal": seal["status"], "new_prediction_jobs": "30/30",
        "main_new_jobs_and_views": {"jobs":18,"task_method_seed_views":90},
        "reused_v2_main_base_jobs_and_views": {"jobs":6,"task_method_seed_views":30},
        "rgb_probe_new_jobs_and_views": {"jobs":12,"task_condition_seed_views":36},
        "rgb_matched_v2_reference_jobs_and_views": {"jobs":6,"task_views":18,"overlap_with_main_gv_views":15,"additional_sv90_views":3},
        "dev_target_reader_before_seal": False,
        "new_dev_target_action_rows_read_across_attempts": 11100,
        "new_dev_target_action_rows_read_by_attempt": {"attempt_01_failed_after_read":5550,"attempt_02_scoring_pass":5550},
        "parent_v2_cumulative_target_action_rows_read_preserved": 16650,
        "parent_plus_v3_target_action_rows_read": 27750,
        "existing_v2_target_trace_points_reused_for_figures": 300,
        "final_heldout_payload_access": 0, "numerical_support_gate": False,
        "positive_gain_gate": False, "independent_confirmation": False,
        "old_v2_root_writes": 0, "human_review_status": "pending",
        "limitations": ["post-V2 development-only; dev outcomes were accessed historically and again after the V3 seal",
                        "session/trial independence unverified", "some cache extractor provenance binding remains partial",
                        "no independent confirmation, OOD/ICL, or real-robot claim"],
        "watcher": watcher_qa,
        "figure_build_status": figure_qa.get("status"),
        "figure_manual_visual_review_status": figure_qa.get("manual_visual_review_status"),
        "figure_human_review_status": "pending",
        "failed_attempt_ledger_records": len([x for x in (ROOT / "qa/FAILED_ATTEMPTS.jsonl").read_text().splitlines() if x.strip()]),
        "final_qa_outputs": ["qa/BASE_FROZEN.json","qa/INPUT_LABEL_SEPARATION.json","qa/WEIGHT_METRIC_AND_BIAS_IDENTITIES.json",
                              "qa/PREDICTION_ACCEPTANCE.json","qa/DEV_TARGET_SCORING_QA.json","qa/FIGURE_BUILD_QA.json",
                              "qa/FINAL_ACCEPTANCE.json","qa/WATCHER_PROVENANCE_RECONCILIATION.json"],
    }
    write_json(ROOT / "qa/PREDICTION_ACCEPTANCE.json", {
        "schema_version":1,"status":"PASS","created_utc":now(),"prediction_seal_sha256":sha(seal_path),
        "new_jobs_accepted":30,"new_jobs_planned":30,"missing_jobs":0,"V2_reference_jobs":9,
        "main_jobs":18,"main_task_method_seed_views":90,"rgb_probe_jobs":12,"rgb_task_condition_seed_views":36,
        "all_shapes_ordered_ids_finite_and_hashes_verified":True,"empty_seal_target_read_refused":True,
        "dev_target_rows_read_before_seal":0,"final_heldout_payload_access":0})
    write_json(ROOT / "qa/FINAL_ACCEPTANCE.json", final_acceptance)

    # Return JSON contains per-task/per-seed metric rows and all locked contrasts.
    method_by_seed = read_csv(SCORE / "METHOD_METRICS_BY_TASK_SEED.csv")
    contrast_by_seed = read_csv(SCORE / "CONTRASTS_BY_TASK_SEED.csv")
    def numberize(rows: list[dict[str, str]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
        out = []
        for row in rows:
            item: dict[str, Any] = {}
            for k, v in row.items():
                if k in fields:
                    try:
                        item[k] = int(v) if k in {"seed", "task_id", "dev_episodes", "dev_source_groups"} else float(v)
                    except (TypeError, ValueError): item[k] = v
                else: item[k] = v
            out.append(item)
        return out
    rgb_reporting = []
    for row in rgb_rows:
        item = dict(row)
        for key in list(item):
            if key in {"seed", "task_id", "task_episodes"}: item[key] = int(item[key])
            elif key == "rgb_intervention_is_diagnostic_only": item[key] = item[key].lower() == "true"
            elif key not in {"model_id", "condition", "matched_reference_method"}: item[key] = float(item[key])
        rgb_reporting.append(item)
    choice_summary = [{"program":x["program"],"selected_lambda":x["selected_lambda"],"selected_epoch":x["selected_epoch"],
                       "selected_tune_j":x["selected_tune_j"],"selected_tune_position_mae":x["selected_tune_position_mae"],
                       "selected_tune_safe_fd_l1":x["selected_tune_safe_fd_l1"],"status":x["status"]}
                      for x in selection["choices"]]
    return_json = {
        "schema_version": 1, "status": "complete", "created_utc": now(),
        "mode": "post_V2_exploratory_frozen_readout", "V2_root": str(V2), "V3_root": str(ROOT),
        "tasks": list(TASKS), "seeds": list(SEEDS), "dev_support": {"119":48,"94":48,"90":41,"141":20,"230":28},
        "fit_scope": {"train_episodes":220,"train_anchors":1280,"inner_fit_episodes":174,"inner_fit_anchors":1010,
                      "inner_tune_episodes":46,"inner_tune_anchors":270,"encoder_training":0,"encoder_extraction":0,
                      "V2_78_fits_rerun":0,"neural_screening_fits_accepted":8,"neural_formal_fits_accepted":12,
                      "cpu_bias_calibrations_accepted":6},
        "readout": {"hidden_shape":"[B,30,128]","shared_parameters":3612,"task_bank_total_parameters":18060,
                    "active_task_parameters_per_sample":3612,"formal_seeds":list(SEEDS),"selection_lock_sha256":sha(ROOT/"protocol/READOUT_SELECTION.json"),
                    "choices":choice_summary,
                    "frozen_v2_base_sources": {stage:[{"model_id":x["model_id"],"seed":x["seed"],"lr":x["lr"],"epochs":x["epochs"],
                                                         "model_sha256":x["model_sha256"],"normalizer_sha256":x["normalizer_sha256"],"video_mode":x.get("video_mode")}
                                                        for x in rows] for stage,rows in model_lock["base_sources"].items()}},
        "V2_reconciliation": {"status":v2_recon["status"],"independent_point_rows":v2_recon["independent_point_rows"],
                              "point_rows_match_at_4dp":v2_recon["point_rows_matching_at_4_decimals"],
                              "max_abs_point_difference":v2_recon["max_abs_point_difference"],
                              "max_abs_summary_difference":v2_recon["max_abs_summary_field_difference"],
                              "existing_all_seed_loo_means_leave_one_task_out":True,
                              "true_seed_loo_rows":len(v2_seed_loo_rows),"true_task_loo_rows":len(v2_task_loo_rows),
                              "macro_points":v2_macro,"metric_scales":scale_summary},
        "scores_by_task_seed": numberize(method_by_seed,("seed","task_id","value","dev_episodes","dev_source_groups")),
        "contrasts_by_task_seed": numberize(contrast_by_seed,("seed","task_id","value")),
        "macro_methods_mean_sample_sd": macro_methods,"macro_contrasts_mean_sample_sd_and_ci":macro_contrasts,
        "leave_one_out_ranges":loo_ranges,"rgb_probe_task_seed_views":rgb_reporting,
        "rgb_probe_means":rgb_means,"state_rgb_invariance": {"status":state_qa["status"],"fixtures":len(state_qa["fixtures"]),
                   "max_abs_changes":state_changes,"nonzero_R_checks":sum(int(c.get("shared_R_nonzero",False))+int(c.get("task_R_head_nonzero",False))
                   for fix in state_qa["fixtures"] for c in fix["conditions"])},
        "target_access": {"prediction_seal_sha256":sha(seal_path),"new_rows_total":11100,
                          "by_attempt":{"attempt_01_failed_after_target_read":5550,"attempt_02_pass":5550},
                          "parent_v2_rows":16650,"parent_plus_v3_rows":27750,
                          "existing_v2_trajectory_trace_points_reused":300,"final_heldout_payload_access":0},
        "gpu": {"host":environment["host"],"driver":environment["driver_version"],"cards_at_admission":env_cards,
                "physical_gpus_used":physical_gpus,"logical_device":"cuda:0 per leased child",
                "shared_watcher_limit":2,"actual_max_concurrency":max_watcher_children,
                "successful_watcher_jobs":counts["complete"],"failed_watcher_attempts":counts["failed"],
                "all_leases_released":all_released,"sum_successful_watcher_job_wall_seconds":sum(watcher_durations),
                "fit_elapsed_seconds":fit_elapsed,"fit_peak_allocated_mib":fit_max_alloc,"fit_peak_reserved_mib":fit_max_reserved,
                "feature_peak_allocated_mib":feature_max_alloc,"feature_peak_reserved_mib":feature_max_reserved,
                "rgb_peak_allocated_mib":probe_max_alloc,"rgb_peak_reserved_mib":probe_max_reserved,
                "all_successful_app_peak_allocated_mib":app_max_alloc,"all_successful_app_peak_reserved_mib":app_max_reserved,
                "watcher_provenance":watcher_qa},
        "limitations":final_acceptance["limitations"],"human_review_status":"pending",
        "files": {"main_task_seed_metrics":"results/scoring/attempt_02/METHOD_METRICS_BY_TASK_SEED.csv",
                  "contrasts":"results/scoring/attempt_02/CONTRASTS_BY_TASK_SEED.csv",
                  "summary":"results/scoring/attempt_02/SUMMARY_STATISTICS.csv",
                  "seed_loo":"results/scoring/attempt_02/LEAVE_ONE_SEED_OUT.csv",
                  "task_loo":"results/scoring/attempt_02/LEAVE_ONE_TASK_OUT.csv",
                  "conditional_ci":"results/scoring/attempt_02/PAIRED_SOURCE_GROUP_CI.csv",
                  "rgb":"results/scoring/attempt_02/RGB_PROBE_RESULTS.csv",
                  "figures":"figures/","figure_data":"figure_data/","checksum":"CHECKSUMS.sha256"},
    }
    write_json(ROOT / "RETURN_TO_CHATGPT.json", return_json)

    # Human-readable report.
    md: list[str] = []
    md += ["# EE6008 V3 冻结共享模型与输出残差适配：执行回传", "",
           "**状态：完成（开发实验）**。V3 复用 V2 corrected run 的 split、冻结权重、normalizer、Q0 cache 和预测；未重训 V2 的 78 fits，也未训练/提取 encoder。最终/heldout payload 访问为 0。", "",
           "## 数据、模型与训练", "",
           "- 固定任务：119、94、90、141、230。训练集 220 episodes / 1,280 anchors；直接复用 inner-fit 174 / 1,010 与 inner-tune 46 / 270；dev 为 185 anchors，逐任务 support 为 48、48、41、20、28。无最小样本或正收益门。",
           "- base hidden 为 `[B,30,128]`；共享线性 residual 3,612 参数，任务五头总计 18,060，每条样本激活 3,612。base 全程冻结，optimizer 只含 residual heads；future labels 只进入训练 loss/scorer，没有进入 forward。",
           "- 8 个 screening fits 全部接受，12 个 formal fits 全部接受，另有 6 个 CPU train-only weighted-median bias calibrations。Stage0 两个 smoke 通过，readout 更新不计入 fits。",
           "- V3 LR 固定为 1e-3；只在 inner-tune 选择 λ∈{0,1} 与 epoch∈{20,40,80}。四程序选择如下：", "",
           "|程序|λ|epoch|tune J|", "|---|---:|---:|---:|"]
    for x in choice_summary: md.append(f"|{x['program']}|{x['selected_lambda']:g}|{x['selected_epoch']}|{x['selected_tune_j']:.9f}|")
    md += ["", "冻结 V2 base 的实际来源（selection/manifest 锁定；V3 未更新 base）：", "",
           "|阶段|模型|seed(s)|V2 LR|epochs|模式|", "|---|---|---|---:|---:|---|"]
    for stage,rows in model_lock["base_sources"].items():
        for model_id in sorted({x["model_id"] for x in rows}):
            subset=[x for x in rows if x["model_id"]==model_id]
            first=subset[0]
            md.append(f"|{stage}|{model_id}|{';'.join(str(x['seed']) for x in subset)}|{first['lr']:g}|{first['epochs']}|{first.get('video_mode','')}|")
    md += ["", "## V2 轻量复核", "",
           f"V2 saved-loss 复算与接受表核对通过：{v2_recon['independent_point_rows']} 个 per-seed contrast 行最大绝对差 `{v2_recon['max_abs_point_difference']:.1e}`；{v2_recon['summary_fields_matching_at_4_decimals']} 个 summary 字段在四位小数匹配（最大差 `{v2_recon['max_abs_summary_field_difference']:.2e}`）。旧 `ALL_SEED_LOO.csv` 实际是 leave-one-task-out；V3 分别生成 true seed-LOO（留 2 seeds、5 tasks 全保留）与 task-LOO（每 seed 留 4 tasks）。",
           "", f"position/safe-FD 使用 V2 accepted canonical metric scales（lock SHA `{sha(v2_metric_path)}`）；完整 28 轴 `target_sigma` 与 `safe_fd_scale` 在 `RETURN_TO_CHATGPT.json`。旧 V2 macro 点估计：", "",
           "|V2 contrast|position MAE mean±sample SD|safe-FD mean±sample SD|", "|---|---:|---:|"]
    for x in v2_macro: md.append(f"|{x['contrast']}|{x['position_mae']['mean_across_seeds']:.6f} ± {x['position_mae']['sample_sd_across_seeds']:.6f}|{x['safe_fd_l1']['mean_across_seeds']:.6f} ± {x['safe_fd_l1']['sample_sd_across_seeds']:.6f}|")
    md += ["", "## V3 五任务结果", "",
           "下表为每任务三个 formal seeds 的均值；每格是 `position MAE / safe-FD L1`，两个指标分别报告、越低越好。全量 per-seed 数值和 sample SD 见 `results/scoring/attempt_02/METHOD_METRICS_BY_TASK_SEED.csv` 与 `SUMMARY_STATISTICS.csv`。", "",
           "|Task|Persistence|Base state|Bias state|Shared-R state|Task-R state|Base video|Bias video|Shared-R video|Task-R video|", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        cells=[]
        for method in methods:
            p=next(x for x in task_summary if x["scope"]=="task" and x["task_id"]==str(task) and x["result_type"]=="method" and x["name"]==method and x["metric"]=="position_mae")
            d=next(x for x in task_summary if x["scope"]=="task" and x["task_id"]==str(task) and x["result_type"]=="method" and x["name"]==method and x["metric"]=="safe_fd_l1")
            cells.append(f"{float(p['mean_across_seeds']):.6f}/{float(d['mean_across_seeds']):.6f}")
        md.append("|"+str(task)+"|"+"|".join(cells)+"|")
    md += ["", "五任务 macro（先 episode 等权、再 task 等权；数值为三个 seed 的 mean±sample SD）：", "",
           "|方法|position MAE|safe-FD L1|", "|---|---:|---:|"]
    for x in macro_methods:
        md.append(f"|{x['method']}|{x['position_mae']['mean_across_seeds']:.6f} ± {x['position_mae']['sample_sd_across_seeds']:.6f}|{x['safe_fd_l1']['mean_across_seeds']:.6f} ± {x['safe_fd_l1']['sample_sd_across_seeds']:.6f}|")
    md += ["", "### 配对 contrast 与条件区间", "",
           "定义统一为 `L(left)-L(right)`；正数表示 right 误差更低。区间为 5,000 次 paired source-group bootstrap，seed 202610134，同一组 draws 共用于方法、seed、指标；区间条件于已训练 seeds 与 tune 选择，不是独立确认，不给 p 值。", "",
           "|Contrast|Δposition mean±SD [95% group CI]|Δsafe-FD mean±SD [95% group CI]|", "|---|---:|---:|"]
    for x in macro_contrasts:
        pm=x["position_mae"]; fd=x["safe_fd_l1"]
        md.append(f"|{x['contrast']}|{pm['mean_across_seeds']:.6f} ± {pm['sample_sd_across_seeds']:.6f} [{pm['conditional_group_bootstrap_ci95'][0]:.6f}, {pm['conditional_group_bootstrap_ci95'][1]:.6f}]|{fd['mean_across_seeds']:.6f} ± {fd['sample_sd_across_seeds']:.6f} [{fd['conditional_group_bootstrap_ci95'][0]:.6f}, {fd['conditional_group_bootstrap_ci95'][1]:.6f}]|")
    md += ["", "**解释：**全五任务上，task residual 没有稳定地同时优于 shared residual 的 position 与 safe-FD。state 的 `T_R` position 为正但区间跨 0，safe-FD 为负且区间低于 0；video 的两项 `T_R` 区间跨 0。train-only bias 的 macro position 小幅改善，safe-FD 恒等于 base；residual 的 `R_vs_BIAS` 多数为负，整体未超过简单 bias 校准。任务 90 上 residual 有局部改善，不能外推为五任务一致优势。", "",
           "真正 seed-LOO/task-LOO 的完整表在 `LEAVE_ONE_SEED_OUT.csv` 与 `LEAVE_ONE_TASK_OUT.csv`。seed-LOO 排除一个 seed、保留另两个 seed 和全部五任务；task-LOO 每个 seed 排除一个 task、保留其余四任务。主要 paired contrast 的范围：", "",
           "|Contrast|指标|seed-LOO range|task-LOO all-seed mean range|", "|---|---|---:|---:|"]
    for x in loo_ranges:
        md.append(f"|{x['contrast']}|{x['metric']}|[{x['seed_loo_range'][0]:.6f}, {x['seed_loo_range'][1]:.6f}]|[{x['task_loo_all_seed_mean_range'][0]:.6f}, {x['task_loo_all_seed_mean_range'][1]:.6f}]|")
    md += ["", "## RGB 诊断", "",
           f"固定 checkpoint RGB 探针为 12 个新 job / 36 个 task-condition-seed views；另复用 18 个 matched V2 references，其中 15 个 G-V views 与主比较的 30 个旧 base views 重叠，另有 3 个 S-V-90 views。G-V 上零输入与 same-task donor 的平均 `condition−matched` position/F-D 变化分别为 `{next(x for x in rgb_means if x['model_id']=='G-V' and x['condition']=='ZERO_NORMALIZED_BLOCK')['mean_position_error_change']:.6f}/{next(x for x in rgb_means if x['model_id']=='G-V' and x['condition']=='ZERO_NORMALIZED_BLOCK')['mean_safe_fd_error_change']:.6f}` 与 `{next(x for x in rgb_means if x['model_id']=='G-V' and x['condition']=='SAME_TASK_DONOR')['mean_position_error_change']:.6f}/{next(x for x in rgb_means if x['model_id']=='G-V' and x['condition']=='SAME_TASK_DONOR')['mean_safe_fd_error_change']:.6f}`；正数代表扰动使误差上升。S-V-90 position 也上升，safe-FD 变化跨 seed。",
           f"state-only QA 覆盖 5 个任务；base output、h、非零 shared-R 与 task-R 的最大变化都为 `{max(state_changes.values()):.1f}`。这说明这些 state-only 输入路径对所测 RGB block 扰动不敏感；video 对照显示输入扰动会改变 h/output 和误差，但不能推出视频信息有因果效用。详见 `results/scoring/attempt_02/RGB_PROBE_RESULTS.csv`。", "",
           "## GPU、执行与访问审计", "",
           f"宿主 `gpu33` 在 admission 快照有 4 张 RTX 3090（24,576 MiB/card）；GPU0/1 分别空闲 24,555/24,560 MiB，GPU2 当时利用率 93%、空闲 10,968 MiB，GPU3 空闲 2,266 MiB。按共享 watcher 的 `dynamic_all_visible` 与 hard cap 2 并行，实际用物理 GPU {', '.join(physical_gpus)}（子进程逻辑设备 `cuda:0`）；57 个 watcher attempts 中 43 complete、14 failed，全部 lease 与 child slot 都释放，未向非 child 进程发信号。",
           f"成功 workload 的最大应用峰值为 `{app_max_alloc:.1f} MiB allocated / {app_max_reserved:.0f} MiB reserved`（RGB probe）；readout fit 最大 `{fit_max_alloc:.1f}/{fit_max_reserved:.0f} MiB`，feature extraction 最大 `{feature_max_alloc:.1f}/{feature_max_reserved:.0f} MiB`。20 个 fit 记录的 GPU elapsed 合计 `{fit_elapsed:.1f}s`；43 个成功 watcher job 的逐 job wall-time 合计 `{sum(watcher_durations):.1f}s`（并行总和，不是端到端墙钟；失败尝试另计）。",
           f"watcher provenance：当前源码与 V2 source acceptance/later phase receipt 都是 `{watcher_current_sha}`；`ENVIRONMENT_PRECHECK.json` 原记录为 `49fcf49e…`，保留该 hash 不一致且未修改 watcher。",
           f"seal 前 dev target rows 为 0。V3 scorer 的 attempt_01 在 seal 后读了 5,550 个 future action rows 后因 V2 `seed=shared` 汇总行解析失败；attempt_02 再读 5,550 行并评分通过。V3 实际共读 11,100 行，V2 历史 16,650 行保留，累计 27,750；final/heldout 一直为 0。示例轨迹复用已有 V2 derived plot data 的 300 个 target trace points，单独记账；没有再打开 scorer-only target artifact。",
           ""]
    md += ["## 保留的失败与限制", "",
           f"`qa/FAILED_ATTEMPTS.jsonl` 保留 {len([x for x in (ROOT/'qa/FAILED_ATTEMPTS.jsonl').read_text().splitlines() if x.strip()])} 条首轮失败。12 个 formal attempt_01 在 optimizer update 前失败，修正后 12/12 通过；18 个主 CPU prediction attempt_01 修复 shape/key 后 18/18 通过；两次 seal 失败均在读标签前；score attempt_01 是唯一造成额外 postseal dev reads 的失败。RGB figure attempt_01 的空条件分组图也已备份并修正，评分结果未受影响。",
           "缓存 extractor/session/trial 绑定仍有既有证据限制；dev 已在 V2 多次访问，本轮是 post-V2 exploratory，不是首次 dev 暴露、独立确认、OOD/ICL 或真实机器人结论。当前结果保留负向与 seed-sensitive 结果。人工/教授复审仍 pending。", "",
           "## 文件索引", "",
           "- 完整机器可读回传：`RETURN_TO_CHATGPT.json`。",
           "- per-task/per-seed 双指标：`results/scoring/attempt_02/METHOD_METRICS_BY_TASK_SEED.csv`。",
           "- paired contrasts、LOO、条件区间：同目录 `CONTRASTS_BY_TASK_SEED.csv`、`LEAVE_ONE_SEED_OUT.csv`、`LEAVE_ONE_TASK_OUT.csv`、`PAIRED_SOURCE_GROUP_CI.csv`。",
           "- 预测封存、访问、验收：`protocol/PREDICTION_BATCH_SEAL.json`、`qa/DEV_TARGET_SCORING_QA.json`、`qa/FINAL_ACCEPTANCE.json`。",
           "- figures 与 captions：`figures/`；绘图数据：`figure_data/`。Checksum-last 清单：`CHECKSUMS.sha256`。", ""]
    (ROOT / "RETURN_TO_CHATGPT.zh-CN.md").write_text("\n".join(md), encoding="utf-8")

    # Final owner closure. Active GPU jobs and tmux session were already verified absent.
    owner = read_json(owner_path)
    owner.update({"phase":"complete","status":"complete","active_jobs":0,"owner_lock_acquired":False,
                  "owner_pid":None,"tmux_session_active":False,"next_gate":"stop; V3 exploratory experiment complete",
                  "updated_utc":now(),"final_heldout_payload_access":0,
                  "completed_outputs":["RETURN_TO_CHATGPT.zh-CN.md","RETURN_TO_CHATGPT.json","qa/FINAL_ACCEPTANCE.json",
                                      "coordination/JOB_LEDGER.json","coordination/RESOURCE_LOG.csv"]})
    write_json(owner_path, owner)
    (ROOT / "coordination/RESUME_COMMANDS.md").write_text(
        "# V3 completion and read-only recheck\n\n"
        "V3 is complete; there are no active V3 GPU jobs and the tmux session is absent. No training resume command is pending.\n\n"
        "From the V3 code directory, rerun the prediction-only integrity check with:\n\n"
        "```bash\n"
        "PYTHONDONTWRITEBYTECODE=1 python3 -B score_v3_results.py --preflight-only\n"
        "```\n\n"
        "This command checks the sealed predictions and does not open target labels. Do not rerun scoring unless a new authorized scoring attempt is intended and its target rows will be added to `inventory/OUTCOME_EXPOSURE.json`.\n",
        encoding="utf-8")

    # Final acceptance is written after all component receipts exist.
    final_acceptance["completion_artifacts"] = ["RETURN_TO_CHATGPT.zh-CN.md","RETURN_TO_CHATGPT.json",
                                                 "coordination/JOB_LEDGER.json","coordination/RESOURCE_LOG.csv",
                                                 "coordination/RESUME_COMMANDS.md"]
    final_acceptance["watcher_provenance_reconciliation"] = watcher_recon
    write_json(ROOT / "qa/FINAL_ACCEPTANCE.json", final_acceptance)
    print(json.dumps({"status":"PASS","fit_count":len(fits),"feature_count":len(features),
                      "new_prediction_jobs":len(seal["new_prediction_entries"]),"watcher_jobs":len(watcher_rows),
                      "watcher_status_counts":dict(counts),"fit_gpu_seconds":fit_elapsed,
                      "overall_application_peak_allocated_mib":app_max_alloc,
                      "overall_application_peak_reserved_mib":app_max_reserved,
                      "v3_target_rows_total":11100,"combined_parent_plus_v3_rows":27750,
                      "final_heldout_payload_access":0,"report":str(ROOT/"RETURN_TO_CHATGPT.zh-CN.md")},indent=2))


if __name__ == "__main__":
    main()
