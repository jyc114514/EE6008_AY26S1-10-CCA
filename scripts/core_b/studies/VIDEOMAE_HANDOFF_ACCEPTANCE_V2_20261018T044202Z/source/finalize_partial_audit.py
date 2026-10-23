#!/usr/bin/env python3
"""Finalize the auditable CPU-only portion while GPU approval is pending."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


SELECTED_METRICS = {
    "standardized_mae", "native_mae", "standardized_rmse", "native_rmse",
    "native_future_internal_fd", "safe_fd_normalized", "tv_pred_mean", "tv_gt_mean",
    "tv_ratio_group_weighted_means", "tv_ratio_group_mean_distribution_mean",
    "tv_ratio_window_p50", "direction_accuracy_active", "active_step_count_total",
    "near_zero_gt_step_count_total", "small_nonzero_gt_step_count_total",
    "seam_error_native", "endpoint_native_mae", "n_windows", "n_episodes", "n_source_groups"
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x
    tmp = path.with_name("." + path.name + ".part")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_csv(path: Path):
    return list(csv.DictReader(path.open(newline="", encoding="utf-8")))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def semantics(field: str) -> tuple[str, str]:
    low = field.lower()
    if field in {"n_windows", "n_episodes", "n_source_groups"} or "count" in low or low.startswith("zero_gt_tv_"):
        return "count/support", "exact"
    if "standardized" in low:
        return "standardized; train-only target_sigma", "window_to_episode_to_source_group_equal"
    if "native" in low or "anchor" in low or "seam" in low or "endpoint" in low:
        return "native action coordinate; physical units unresolved", "window_to_episode_to_source_group_equal"
    if "ratio" in low:
        return "dimensionless ratio", "statistic-specific"
    if "direction" in low:
        return "fraction", "active-step support"
    if field.endswith("_p25") or field.endswith("_p50") or field.endswith("_p75"):
        return "metric statistic", "window_or_source_group_quantile"
    return "metric statistic", "window_to_episode_to_source_group_equal"


def compare(old, new):
    def finite_or_none(v):
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v
    old, new = finite_or_none(old), finite_or_none(new)
    if old is None and new is None:
        return old, new, 0.0, None, "both_NA"
    if old is None or new is None:
        return old, new, None, None, "NA_support_mismatch"
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        delta = float(new) - float(old)
        return old, new, abs(delta), None if float(old) == 0 else delta / abs(float(old)), "compared"
    return old, new, None, None, "metadata"


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("project-root", "v2-root", "v1-root", "team-root", "audit-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    project, v2, v1, team, audit = [p.resolve() for p in
        (args.project_root, args.v2_root, args.v1_root, args.team_root, args.audit_root)]

    expected = {r["cell"]: r for r in read_csv(audit / "EXPECTED_CELLS.csv")}
    archive = read_json(v2 / "ARCHIVE.json")
    archive_cells = {c["name"]: c for c in archive["cells"]}
    cpu_root = v2 / "replay/cpu_zero_20261013T065954Z"
    metric_rows, seed_rows = [], []
    for cell, erow in expected.items():
        acell = archive_cells[cell]
        old_run = v2 / acell["run"]
        cpu_cell = cpu_root / cell
        old_json = read_json(v2 / acell["metrics"])
        cpu_json = read_json(cpu_cell / "evaluation/METRICS.json")
        mode, horizon, seed = erow["mode"], int(erow["horizon"]), int(erow["seed"])
        model_label = "VideoMAE" if mode == "TS11-D" else "STATE-CTRL"
        for split in ("train", "dev"):
            for region in ("action_0_14", "action_14_28", "action_full28"):
                old_models = old_json["panels"][split][region]
                cpu_models = cpu_json["panels"][split][region]
                for model_name in sorted(set(old_models) | set(cpu_models)):
                    old_panel, cpu_panel = old_models.get(model_name, {}), cpu_models.get(model_name, {})
                    support = {
                        "windows_old": old_panel.get("n_windows"), "windows_new": cpu_panel.get("n_windows"),
                        "episodes_old": old_panel.get("n_episodes"), "episodes_new": cpu_panel.get("n_episodes"),
                        "source_groups_old": old_panel.get("n_source_groups"),
                        "source_groups_new": cpu_panel.get("n_source_groups")
                    }
                    for field in sorted(set(old_panel) | set(cpu_panel)):
                        old, new, absd, reld, status = compare(old_panel.get(field), cpu_panel.get(field))
                        unit, aggregation = semantics(field)
                        metric_rows.append({
                            "cell": cell, "mode": mode, "horizon": horizon, "seed": seed,
                            "split": split, "region": region, "model": model_name,
                            "metric": field, "statistic": field, "unit_scale": unit, "aggregation": aggregation,
                            "backend_comparison": "historical_GPU_vs_historical_CPU",
                            "old": old, "new": new, "abs_delta": absd, "relative_delta": reld,
                            "support": json.dumps(support, sort_keys=True), "status": status
                        })
                old_panel, cpu_panel = old_models[model_label], cpu_models[model_label]
                for field in sorted(SELECTED_METRICS & (set(old_panel) | set(cpu_panel))):
                    old, new, absd, reld, status = compare(old_panel.get(field), cpu_panel.get(field))
                    seed_rows.append({
                        "cell": cell, "mode": mode, "horizon": horizon, "seed": seed,
                        "split": split, "region": region, "metric": field,
                        "historical_GPU": old, "historical_CPU": new, "abs_delta": absd,
                        "relative_delta": reld, "unit_scale": semantics(field)[0],
                        "aggregation": semantics(field)[1],
                        "support_windows": old_panel.get("n_windows"),
                        "support_episodes": old_panel.get("n_episodes"),
                        "support_source_groups": old_panel.get("n_source_groups"),
                        "comparison_status": status, "new_GPU": "not_tested"
                    })
    write_csv(audit / "METRIC_REPLAY_DIFFS.csv", metric_rows)
    write_csv(audit / "CORE_B_PER_SEED_METRIC_SUMMARY.csv", seed_rows)

    paired_path = v1 / "CORE_B_PAIRED_CONTRASTS.csv"
    paired = [r for r in read_csv(paired_path) if r["split"] == "dev" and r["horizon"] == "30"
              and r["region_raw"] == "action_full28" and r["metric"] == "standardized_mae"
              and r["contrast"] in {"Delta_backbone", "G_VM", "G_VJ"}]
    if len(paired) != 3:
        raise RuntimeError("expected three already accepted V1 H30 Full28 sMAE contrast rows")

    team_sd_path = team / "analysis/MODEL_METRICS_BY_SEED.csv"
    team_rows = read_csv(team_sd_path)
    sd_rows = [r for r in team_rows if r["split"] == "dev" and r["variant"] == "corrected_scheduler"
               and r["config"] == "STATE-CTRL" and r["horizon_steps"] == "30"
               and r["region_raw"] == "action_full28"]
    if len(sd_rows) != 3:
        raise RuntimeError(f"expected three corrected TEAM STATE-CTRL H30 Full28 rows; found {len(sd_rows)}")
    sd_values = {r["seed"]: float(r["standardized_mae"]) for r in sd_rows}
    sd_mean = statistics.mean(sd_values.values())
    sd_value = statistics.stdev(sd_values.values())

    erratum_hits = []
    for rel in ("02_SCIENTIFIC_RESULTS.zh-CN.md", "RETURN_TO_CHATGPT.zh-CN.md",
                "FINAL_ACCEPTANCE.json", "FINAL_REPORT_QA.json"):
        path = team / rel
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            if any(token in text for token in ("erratum", "errata", "勘误")):
                erratum_hits.append(str(path))
    team_check = {
        "status": "current_table_verified_erratum_link_not_found" if not erratum_hits else "current_table_and_erratum_reference_found",
        "current_corrected_TEAM_STATE_CTRL_H30_Full28_dev_standardized_MAE": {
            "source": str(team_sd_path), "source_sha256": digest(team_sd_path),
            "seed_values": sd_values, "mean": sd_mean, "sample_sd_ddof1": sd_value,
            "approximation_target": 0.00083946, "matches_target": abs(sd_value - 0.00083946) < 1e-7
        },
        "old_sealed_result_erratum_link": {
            "status": "not_found_in_declared_TEAM_REFERENCE_documents" if not erratum_hits else "found",
            "checked_documents": [str(team / rel) for rel in ("02_SCIENTIFIC_RESULTS.zh-CN.md",
                                   "RETURN_TO_CHATGPT.zh-CN.md", "FINAL_ACCEPTANCE.json", "FINAL_REPORT_QA.json")],
            "hits": erratum_hits,
            "action": "leave old sealed result untouched; report owner should attach erratum reference"
        },
        "bootstrap_recomputed": False, "final_test_access": 0
    }
    write_json(audit / "TEAM_REFERENCE_CHECK.json", team_check)
    write_json(audit / "SCIENTIFIC_CONTEXT_REUSED.json", {
        "status": "accepted_reused", "source": str(paired_path), "source_sha256": digest(paired_path),
        "scope": "fixed three seeds; exposed development split; 603 source groups; 10000 paired group draws already accepted",
        "h30_full28_standardized_mae_contrasts": paired,
        "interpretation": "The H30 VideoMAE-vs-V-JEPA mean contrast is concentrated in seed 20261002. V1's accepted per-model mean contrasts against each model's own STATE-CTRL do not establish an average visual gain. Fixed seeds and exposed development data do not support pretrained-objective causality.",
        "bootstrap_recomputed": False, "final_test_access": 0
    })

    registry_path = audit / "SOURCE_REGISTRY.csv"
    registry = read_csv(registry_path)
    known_paths = {r.get("path") for r in registry}
    source_specs = [
        ("audit_entrypoint", audit / "REPRODUCE_V2.sh", "root-parameterized CPU/GPU entrypoint; GPU branch not run"),
        ("audit_preflight_code", audit / "source/preflight.py", "read-only archive, safe model and historical CPU verifier"),
        ("audit_prediction_child", audit / "source/replay_one_cell.py", "safe tensor-only fresh-process predictor"),
        ("audit_gpu_orchestrator", audit / "source/gpu_orchestrator.py", "shared watcher orchestration; live branch not run"),
        ("audit_host_preflight_code", audit / "source/host_preflight.py", "aggregate-only GPU preflight code; not executed"),
        ("audit_prediction_validator", audit / "source/validate_predictions.py", "prediction seal and numeric gate; not executed"),
        ("audit_postseal_scorer", audit / "source/score_predictions.py", "V1 plus team postseal scorer; live branch not executed"),
        ("audit_cpu_fixture_validator", audit / "source/validate_cpu_fixtures.py", "four CPU archive integration fixture QA"),
        ("audit_partial_finalizer", audit / "source/finalize_partial_audit.py", "CPU-only partial evidence report generator"),
        ("copied_v1_independent_scorer", audit / "source/independent_core_b_metrics.py", "byte-identical V1 scorer copy"),
        ("copied_v1_scorer_fixtures", audit / "source/v1_red_capable_metric_fixtures.py", "byte-identical V1 fixture copy"),
        ("team_corrected_sd_table", team_sd_path, "current corrected H30 state-control Full28 per-seed values"),
        ("team_scientific_report", team / "02_SCIENTIFIC_RESULTS.zh-CN.md", "team scientific interpretation and erratum lookup"),
        ("archive_cpu_fixture_acceptance", audit / "ARCHIVE_CPU_FIXTURE_ACCEPTANCE.csv", "four of four CPU fixtures"),
        ("watcher_dry_run_receipt", audit / "GPU_WATCHER_DRY_RUN.json", "four of four shared watcher job schema dry-runs"),
        ("replay_contract", audit / "REPLAY_CONTRACT.json", "predeclared gate and 24 reference hashes"),
        ("profile_lock_pending", audit / "PROFILE_LOCK.json", "historical profile and approval gate")
    ]
    added = []
    for role, path, note in source_specs:
        current = {"role": role, "path": str(path), "exists": True, "bytes": path.stat().st_size,
                   "sha256": digest(path), "status": "PASS", "expected_bytes": "",
                   "expected_sha256": "", "note": note}
        if str(path) in known_paths:
            for row in registry:
                if row.get("path") == str(path):
                    row.update(current)
                    break
        else:
            added.append(current)
    fields = list(registry[0])
    write_csv(registry_path, registry + added, fields)

    v1_final = read_json(v1 / "FINAL_ACCEPTANCE.json")
    v1_core_b = v1_final["acceptance"]["core_b_metrics"]
    reused_dev_qa = read_json(v1 / "metrics/core_b/CORE_B_METRIC_QA.json")
    reused_train_qa = read_json(v1 / "metrics/core_b/CORE_B_TRAIN_ARTIFACT_SCORE_QA.json")
    new_scorer_qa = {
        "status": "accepted_reused_for_existing_saved_predictions",
        "existing_dev_scoring": {
            "status": reused_dev_qa.get("status"),
            "prediction_bundles_prevalidated": reused_dev_qa.get("prediction_bundles_prevalidated_before_targets"),
            "rows": reused_dev_qa.get("dev_rows"), "episodes": reused_dev_qa.get("dev_episodes"),
            "source_groups": reused_dev_qa.get("dev_source_groups"),
            "max_abs_vs_handoff_metrics": reused_dev_qa.get("max_abs_vs_handoff_saved_metrics"),
            "scorer_sha256": reused_dev_qa.get("independent_scorer_sha256")
        },
        "existing_train_scoring": {
            "status": reused_train_qa.get("status"),
            "train_prediction_artifacts_scored": reused_train_qa.get("train_prediction_artifacts_scored"),
            "combined_saved_train_dev_prediction_bundles_scored": reused_train_qa.get("combined_saved_train_dev_prediction_bundles_scored"),
            "metric_comparisons": reused_train_qa.get("metric_comparisons"),
            "max_abs_vs_handoff_metrics": reused_train_qa.get("max_abs_vs_saved_handoff_train_metrics"),
            "checkpoint_replayed": False, "scorer_sha256": reused_train_qa.get("scorer_sha256")
        },
        "new_gpu_prediction_scoring": {"status": "not_tested", "reason": "no 24-output GPU prediction seal exists"},
        "final_test_access": 0, "bootstrap_rerun": False
    }
    write_json(audit / "INDEPENDENT_SCORER_QA.json", new_scorer_qa)
    write_csv(audit / "INDEPENDENT_SCORER_QA.csv", [
        {"item": "existing_dev_prediction_scoring", "status": reused_dev_qa.get("status"),
         "bundles": reused_dev_qa.get("prediction_bundles_prevalidated_before_targets"),
         "max_abs": reused_dev_qa.get("max_abs_vs_handoff_saved_metrics"), "source": str(v1 / "metrics/core_b/CORE_B_METRIC_QA.json")},
        {"item": "existing_train_prediction_scoring", "status": reused_train_qa.get("status"),
         "bundles": reused_train_qa.get("train_prediction_artifacts_scored"),
         "max_abs": reused_train_qa.get("max_abs_vs_saved_handoff_train_metrics"), "source": str(v1 / "metrics/core_b/CORE_B_TRAIN_ARTIFACT_SCORE_QA.json")},
        {"item": "new_gpu_prediction_scoring", "status": "not_tested", "bundles": 0,
         "max_abs": "", "source": "requires the sealed 24-output GPU replay"}
    ])

    reuse_path = audit / "ACCEPTED_REUSE.csv"
    reuse = read_csv(reuse_path)
    if not any(r.get("item") == "TEAM STATE-CTRL H30 Full28 corrected per-seed sample SD" for r in reuse):
        reuse.append({"item": "TEAM STATE-CTRL H30 Full28 corrected per-seed sample SD",
                      "status": "accepted_reused", "evidence": str(team_sd_path),
                      "detail": f"3 fixed seed values; mean={sd_mean:.12g}; sample_sd_ddof1={sd_value:.12g}; no bootstrap rerun"})
    if not any(r.get("item") == "erratum link for old sealed team result" for r in reuse):
        reuse.append({"item": "erratum link for old sealed team result",
                      "status": "needs_owner_follow_up", "evidence": str(team / "02_SCIENTIFIC_RESULTS.zh-CN.md"),
                      "detail": "no erratum/errata/勘误 reference found in current declared team documents; old result untouched"})
    write_csv(reuse_path, reuse)

    dependencies = read_json(audit / "DEPENDENCIES.json")
    dependencies.update({
        "schema_version": "1.0",
        "portability_scope": "frozen_feature_predictor_replay",
        "root_parameterized_entrypoint": str(audit / "REPRODUCE_V2.sh"),
        "verified_from_different_cwd": {
            "status": "PASS_CPU_FIXTURES_4_OF_4",
            "working_directory": str(audit / "tmp")
        },
        "archive_inventory": {
            "registered_files": 538, "verified_file_sha_and_size": "538/538",
            "applicable_source_origin_sha_and_size": "510/510",
            "registered_bytes": 5831063648, "symlinks": 0, "broken_links": 0,
            "absolute_archive_file_paths": 0, "unregistered_bytecode_files": 3,
            "post_archive_replay_subtree_files": 229
        },
        "runtime_versions_from_existing_safe_metadata": {
            "python": "3.12.14", "torch": "2.12.1+cu130", "numpy": "2.5.2",
            "pyarrow": "25.0.1", "safetensors": "0.8.0",
            "current_gpu_driver_cudnn_architecture": "UNKNOWN until explicit host approval and live profile"
        },
        "copied_scoring_dependencies": {
            "v1_independent_scorer": {"path": str(audit / "source/independent_core_b_metrics.py"),
                                      "sha256": digest(audit / "source/independent_core_b_metrics.py")},
            "v1_metric_fixtures": {"path": str(audit / "source/v1_red_capable_metric_fixtures.py"),
                                   "sha256": digest(audit / "source/v1_red_capable_metric_fixtures.py")},
            "team_evaluator": {"path_in_v2_archive": str(v2 / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/code/evaluate_corrected_scheduler.py"),
                               "sha256": digest(v2 / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z/code/evaluate_corrected_scheduler.py")}
        },
        "output_and_cache_policy": {
            "audit_root": str(audit),
            "predictions_reports_status_temp_python_bytecode_and_torch_cuda_triton_caches": "under AUDIT_ROOT",
            "gpu_lease_directory": str(project / "reports/locks/gpu_lease"),
            "shared_lease_files": "operational flock state only; no model outputs or logs",
            "old_root_writes": 0
        },
        "archive_path_resolution": {
            "source_of_safe_weights_config_cache": "V2_ROOT",
            "cache_manifest_original_project_field": "used only to strip source prefix and resolve each shard inside V2_ROOT",
            "runtime_raw_video_access": False,
            "runtime_old_tmp_data_access": False
        },
        "shared_gpu_admission": {
            "watcher": str(project / "repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py"),
            "policy": str(project / "repo/ee6008-humanoid-vjepa/docs/operations/GPU_WATCHER_POLICY.md"),
            "pool": "dynamic_all_visible; no historical physical GPU index inherited",
            "dry_run_status": "PASS_4_OF_4", "host_preflight_status": "approval_blocked_not_executed",
            "tmux": {"available": True, "version": "3.2a", "long_gpu_run_required": True}
        },
        "not_performed": ["raw video decoding", "VideoMAE encoder feature extraction", "retraining",
                          "optimizer updates", "unrestricted pickle deserialization", "final/held-out access"]
    })
    write_json(audit / "DEPENDENCIES.json", dependencies)

    fixture_summary = read_json(audit / "ARCHIVE_CPU_FIXTURE_SUMMARY.json")
    dry_run = read_json(audit / "GPU_WATCHER_DRY_RUN.json")
    preflight = read_json(audit / "PREFLIGHT_SUMMARY.json")
    cpu_diagnostic = read_json(audit / "HISTORICAL_CPU_DIAGNOSIS.json")
    preflight.update({
        "status": "CPU_PREFLIGHT_PASS_GPU_APPROVAL_PENDING",
        "archive_cpu_fixtures": {"status": fixture_summary["status"], "passed": fixture_summary["fixtures_passed"],
                                 "expected": fixture_summary["fixtures_expected"]},
        "watcher_dry_run": {"status": dry_run["status"], "jobs": len(dry_run["stage0_jobs"]), "gpu_queried": False},
        "gpu_host_approval": "not_obtained",
        "nvidia_smi_host_preflight_auto_review": "rejected_explicit_user_approval_missing",
        "nvidia_smi_host_preflight_executed": False,
        "new_gpu_predictions": 0,
        "audit_work_complete": False, "handoff_acceptance_complete": False,
        "current_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    })
    write_json(audit / "PREFLIGHT_SUMMARY.json", preflight)

    write_json(audit / "ARCHIVE_INTEGRATION_ACCEPTANCE.json", {
        "status": "accepted_with_caveats", "portability_scope": "frozen_feature_predictor_replay",
        "registered_archive_files_verified": "538/538", "origin_sources_verified": "510/510",
        "old_handoff_artifacts_rechecked": "169/169", "v1_checksum_entries_rechecked": "81/81",
        "safe_model_structure": "accepted 12/12", "cpu_fresh_process_fixture": "PASS 4/4",
        "shared_watcher_job_dry_run": "PASS 4/4", "gpu_stage0": "approval_blocked_not_tested",
        "gpu_full_replay": "approval_blocked_not_tested", "raw_video_end_to_end": "not_tested",
        "unregistered_post_archive_cpu_replay_files": 229, "unregistered_bytecode_files": 3,
        "symlinks": 0, "broken_links": 0, "absolute_archive_file_paths": 0,
        "source_binding": "hash_bound_exporter_attested", "final_test_access": 0
    })
    write_json(audit / "GPU_RESOURCE_STATUS.json", {
        "status": "approval_blocked", "host_preflight_executed": False,
        "live_gpu_resource_receipts": [], "watcher_child_count": 0,
        "external_process_identity_recorded": False, "non_child_process_signaled": False,
        "tmux_available": True, "tmux_version": "3.2a",
        "reason": "Server AGENTS.md requires explicit host approval before nvidia-smi/CUDA/GPU resource checks; no GPU command executed",
        "auto_review_rejection": {
            "action": "aggregate nvidia-smi host GPU preflight",
            "reviewer_reason_exact": "该命令会实际访问 GPU/NVIDIA 设备；服务器策略要求用户明确批准宿主 GPU 预检，而当前没有此项明确批准。",
            "executed": False, "bypass_attempted": False
        },
        "final_test_access": 0
    })
    write_json(audit / "ERRATA_AND_REMAINING_GAPS.json", {
        "status": "partial",
        "gaps": [
            {"id": "GPU_HOST_APPROVAL", "owner": "user", "status": "auto_review_rejected_waiting_explicit_user_approval",
             "reviewer_reason_exact": "该命令会实际访问 GPU/NVIDIA 设备；服务器策略要求用户明确批准宿主 GPU 预检，而当前没有此项明确批准。",
             "action": "Explicitly approve aggregate host GPU preflight followed by four Stage0 and twelve fixed-model train/dev GPU replays through the shared watcher in tmux."},
            {"id": "TEAM_ERRATUM_LINK", "owner": "team report owner", "status": "needs_owner_follow_up",
             "action": "Provide the erratum link for the old sealed TEAM STATE-CTRL H30 summary. The current declared Team Reference documents contain no erratum reference; the old result remains untouched."},
            {"id": "VJEPA_REFERENCE_PROVENANCE", "owner": "V-JEPA cache/extractor owner", "status": "unresolved",
             "action": "Supply the missing historical extractor source SHA and intended 64-row cache bridge reference. This is separate from VideoMAE package acceptance."}
        ],
        "new_training": 0, "optimizer_updates": 0, "new_encoder_extraction": 0,
        "unrestricted_pickle_loads": 0, "final_test_access": 0, "old_root_writes": 0
    })

    final_acceptance = {
        "schema_version": "1.0", "status": "approval_blocked",
        "audit_work_complete": False, "handoff_acceptance_complete": False,
        "AUDIT_WORK_COMPLETE": False, "HANDOFF_ACCEPTANCE_COMPLETE": False,
        "scope": "safe model load, source binding, fixed epoch-40 train/dev predictor replay, independent scoring, archive integration",
        "CORE_A_STATUS": "accepted_reused",
        "SAFE_MODELS_STRUCTURALLY_ACCEPTED": "12/12",
        "SOURCE_BINDING_EVIDENCE_LEVEL": "hash_bound_exporter_attested",
        "REPLAY_CONTRACT_ORIGIN": "predeclared_V2_engineering_gate",
        "ORIGINAL_PROFILE_MATCH_STATUS": "not_verifiable_historical_gpu_UUID_name_architecture_and_cudnn_missing; live current profile not tested",
        "TRAIN_REPLAY_EXECUTED": "0/12", "DEV_REPLAY_EXECUTED": "0/12",
        "TRAIN_NUMERIC_ACCEPTED": "0/12", "DEV_NUMERIC_ACCEPTED": "0/12",
        "CPU_ZERO_TOLERANCE_STATUS": "historical_fail_preserved",
        "INDEPENDENT_SCORER_QA": "accepted_reused_for_existing_saved_predictions; new GPU scorer not_tested",
        "METRIC_MAX_DIFF_FIELD_IDENTIFIED": "STATE-CTRL_H60_S20261003 dev action_14_28 STATE-CTRL.tv_ratio_window_p50 abs_delta=0.00011648237705230713",
        "ARCHIVE_INVENTORY_ACCEPTANCE": "538/538 registered size+SHA pass; 510 applicable source origins pass; 169/169 old handoff and 81/81 V1 checksum entries pass",
        "ROOT_PARAMETERIZED_ENTRY_ACCEPTANCE": "CPU archive fixtures 4/4 PASS from different cwd; watcher dry-run 4/4 PASS; GPU replay approval_blocked",
        "PORTABILITY_SCOPE": "frozen_feature_predictor_replay",
        "RAW_VIDEO_END_TO_END_REPLAY": "not_tested",
        "STATE_CTRL_CAUSE_STATUS": "unresolved",
        "NEW_TRAINING": 0, "OPTIMIZER_UPDATES": 0, "NEW_ENCODER_EXTRACTION": 0,
        "UNRESTRICTED_PICKLE_LOADS": 0, "FINAL_HELDOUT_ACCESS_THIS_AUDIT": 0, "OLD_ROOT_WRITES": 0,
        "GPU_RESOURCE_RECEIPTS": "none; nvidia-smi aggregate host preflight was rejected before execution pending explicit user approval",
        "GPU_HOST_PREFLIGHT_REVIEW_STATUS": "auto_review_rejected_waiting_explicit_user_approval",
        "NEW_GPU_PREDICTIONS": 0, "PREDICTION_SEAL": "not_created",
        "TEAM_STATE_CTRL_H30_SD": {"status": "accepted_reused", "mean": sd_mean, "sample_sd_ddof1": sd_value},
        "TEAM_OLD_SEALED_ERRATUM_LINK": "not_found_in_declared_TEAM_REFERENCE_documents; old result left untouched",
        "VJEPA_REFERENCE_PROVENANCE": "unresolved_separate_owner_follow_up",
        "CPU_MAX_PREDICTION_DIFF": {"train": 0.0030961036682128906, "dev": 0.0020438432693481445,
                                    "cell": "STATE-CTRL_H60_S20261001", "historical_zero_tolerance": "FAIL"},
        "CPU_MAX_PANEL_DIFF": cpu_diagnostic["metric_summary_field_identification"],
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "audit_root": str(audit),
        "component_status": {
            "core_a_reuse": "accepted_reused", "safe_format": "accepted", "safe_structure": "accepted",
            "source_binding": "accepted_with_caveats", "original_profile_replay": "approval_blocked",
            "current_profile_gpu_replay": "not_tested", "new_gpu_scoring": "not_tested",
            "video_mae_archive_integration": "accepted_with_caveats",
            "team_reference_provenance": "accepted_with_caveats", "raw_video_replay": "not_tested"
        },
        "errors": [],
        "next_owner_actions": read_json(audit / "ERRATA_AND_REMAINING_GAPS.json")["gaps"]
    }
    write_json(audit / "FINAL_ACCEPTANCE.json", final_acceptance)

    report = f"""# VideoMAE V2 安全权重与归档独立复验

状态：approval_blocked。CPU、源身份与归档核验已完成；本次新 GPU 固定权重重放、原 profile 对照和新预测评分尚未执行。服务器级 AGENTS.md 要求 GPU/CUDA 命令先申请宿主权限。自动审查器先拒绝 cudnn 运行时版本探测，随后拒绝了 aggregate nvidia-smi host preflight，因为当前没有明确用户授权；两项命令都没有执行，也没有运行其他 GPU 设备检查。

## 已完成核验

- V2 ARCHIVE 登记的 538/538 项文件 size/SHA 通过；510 个适用源副本与项目源一致。另有 3 个未登记 __pycache__ 文件和 229 个归档后 CPU replay 子树文件，分别列明。没有 symlink、broken link 或登记为绝对路径的文件项。
- 原 handoff 的 169/169 文件及 V1 的 81/81 checksum 条目通过复验。
- 十二份 safetensors 均通过 weights hash、safe metadata、57 tensors/cell、strict key/shape/dtype reload。STATE-CTRL 有 890,780 个参数、768D projection；TS11-D 有 841,628 个参数、384D projection。
- 来源等级为 hash_bound_exporter_attested。现有 exporter 源码含逐tensor roundtrip 检查，但没有独立绑定的导出前 tensor catalogue，也没有单独的 exporter 执行日志。
- Forward state 检查确认 state/video/anchor 三个 persistent modality buffers；没有 nonpersistent buffer。temporal positions、spatial quadrant encoding 和 future queries 在 forward 中生成。
- root 参数化入口从 audit/tmp 工作目录运行四个 train-only CPU fixture，4/4 通过。共享 watcher 的四个 Stage0 job JSON dry-run，4/4 通过。这些结果只证明 CPU archive integration 和 admission schema。

## 历史 CPU 重放

历史 CPU 零容差仍为 FAIL：12/12 cell 的 train 与 dev，共 24/24 split，没有任何一行通过 atol=0、rtol=0。预测最大差为 train 0.003096103668、dev 0.002043843269，均位于 STATE-CTRL_H60_S20261001。

CPU panel 最大差 0.00011648237705230713 已从保存的旧/new metric panels 定位到 STATE-CTRL_H60_S20261003 的 dev action_14_28 STATE-CTRL.tv_ratio_window_p50。旧值 0.19099212437868118，CPU 值 0.19087564200162888。这不是预测数组最大差。

新合同已预先锁定为 abs(new-ref) <= 1e-6 + 1e-6*abs(ref)，equal_nan=false，并同时报告 bitwise equality。该工程诊断门不会用于重签历史 CPU 失败，也不表示科学等效。

## 复用结果与报告缺口

V1 已接受的 Core A train/validation replay、保存 Core B predictions 独立评分和 10,000-draw paired development analysis 标为 accepted_reused；没有重跑 bootstrap。V1 对三 seed、603 dev groups 的条件范围保持原样。

TEAM corrected STATE-CTRL H30 Full28 dev standardized-MAE 的三 seed mean 为 {sd_mean:.12f}，sample SD(ddof=1) 为 {sd_value:.12f}。TEAM_REFERENCE 当前声明的主报告/回传/验收文档中没有找到旧 sealed result 的 erratum 链接；旧 sealed 文件未改写。V-JEPA extractor 历史 SHA 和 64-row bridge reference 仍按原 owner 缺口记录。

## GPU 阶段

PROFILE_LOCK 保留了可恢复的历史 profile：Python 3.12.14、Torch 2.12.1+cu130、NumPy 2.5.2、FP32、eval、inference_mode、batch 64、canonical split order 和 train-only normalizers。历史记录只提供 physical GPU index 2，没有 UUID、型号、架构或 cuDNN。当前 GPU UUID、driver/cuDNN 与 CVD 尚未检查。

获得宿主批准后，入口将先运行 aggregate-only host preflight，再由 shared gpu_lease_watcher 在 tmux 中运行四个 64-row Stage0、锁定当前 profile/UUID，随后逐 cell fresh process 完成十二模型 train/dev replay。24 份预测全部通过精确 ID/shape/input/finite/hash 门并封存后，评分器才读取 train/dev targets。全过程不访问 final/held-out，不训练，不做 encoder extraction，不反序列化 full checkpoint。

详细表、逐项裁决与 owner action 见 FINAL_ACCEPTANCE.json、CHECKPOINT_REPLAY.csv、METRIC_REPLAY_DIFFS.csv、ACCEPTED_REUSE.csv 和 ERRATA_AND_REMAINING_GAPS.json。
"""
    (audit / "AUDIT_REPORT.zh-CN.md").write_text(report, encoding="utf-8")

    return_note = f"""# 回传摘要

- Audit root: {audit}
- 状态：approval_blocked。safe models 12/12；V2 inventory 538/538；旧handoff 169/169；V1 checksums 81/81。
- CPU archive fixtures 4/4 PASS；shared watcher dry-run 4/4 PASS。
- 新 GPU replay：train 0/12、dev 0/12，数值接受各 0/12。原 profile 无法完全验证：历史 GPU UUID、架构、cuDNN 未记录；当前 profile 尚未检查。
- 历史 CPU zero-tolerance FAIL 保留。预测最大差 train 0.003096103668、dev 0.002043843269。CPU panel 最大差字段为 STATE-CTRL_H60_S20261003 / dev / action_14_28 / tv_ratio_window_p50，abs delta 0.00011648237705230713。
- 新工程诊断门为 predeclared_V2_engineering_gate：abs(new-ref) <= 1e-6 + 1e-6*abs(ref)。没有对旧 CPU 结果重签。
- 已复用 V1 Core A、已保存预测评分和 10,000-draw development analysis；未重算统计。
- TEAM corrected STATE-CTRL H30 Full28 dev sMAE sample SD 为 {sd_value:.12f}。指定 TEAM_REFERENCE 中没有找到旧 sealed result erratum link；旧结果未覆盖。
- 新训练、optimizer updates、encoder extraction、unrestricted pickle loads、final/held-out access、old-root writes 均为 0。
- Frozen feature predictor replay 是本次 portability 范围；raw-video end-to-end replay 未测试。

自动审批审查器拒绝了 aggregate nvidia-smi host preflight；其理由是需要明确用户授权。请明确批准宿主 GPU 预检及后续 watcher-managed tmux inference。最多三项 owner action 见 ERRATA_AND_REMAINING_GAPS.json。
"""
    (audit / "RETURN_TO_CHATGPT.zh-CN.md").write_text(return_note, encoding="utf-8")

    print(json.dumps({"status": "partial_report_generated", "metric_rows": len(metric_rows),
                      "seed_metric_rows": len(seed_rows), "team_state_ctrl_h30_sd": sd_value,
                      "new_source_registry_rows": len(added)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

