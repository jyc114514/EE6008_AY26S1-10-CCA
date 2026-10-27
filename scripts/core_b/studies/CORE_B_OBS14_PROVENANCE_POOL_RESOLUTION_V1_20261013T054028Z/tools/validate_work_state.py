#!/usr/bin/env python3
"""Evidence-bound state validator for the isolated R1/R2 prospective study."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import resource
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'reports/benchmarks/CORE_B_RAW_OBS_0_14_PROSPECTIVE_REPLICATION_V1_20261012T044150Z'))
EXPECTED_MODEL = "e6ffb76907634342fe32f6809f12cc9643f5a46a12d085d4196ae946ce4fb2ef"
EXPECTED_VIEWS = ["OBS_PERSISTENCE", "OBS_CONSTANT_VELOCITY", "ARX_HOLD_COMMAND"]
ALLOWED_STAGES = {"provenance_working", "prepared", "metadata_admitted", "sample_sealed", "forecast_sealed", "score_once_complete"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ref_is_valid(ref: dict | None, required_keys: tuple[str, ...]) -> bool:
    if not isinstance(ref, dict):
        return False
    return all(ref.get(key) not in (None, "", False) for key in required_keys)


def assess_state(state: dict, admission: dict, signature: dict, scope: dict,
                 coverage: dict, sample: dict | None = None,
                 forecast: dict | None = None, score: dict | None = None,
                 hash_drift: bool = False) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    stage = state.get("stage")
    if stage not in ALLOWED_STAGES:
        errors.append("unknown_work_stage")
    if state.get("n_required_clusters") != 270:
        errors.append("n_required_must_remain_270")
    if state.get("parent_model_sha256") != EXPECTED_MODEL:
        errors.append("frozen_model_sha256_changed")
    if state.get("new_model_fits") != 0:
        errors.append("new_model_fit_count_nonzero")
    if state.get("gpu_actions") != 0:
        errors.append("gpu_action_count_nonzero")
    if state.get("final_heldout_access") != 0:
        errors.append("protected_payload_access_nonzero")
    if state.get("oracle_or_teacher_used") is not False:
        errors.append("oracle_or_teacher_forcing_forbidden")
    if hash_drift:
        errors.append("evidence_hash_drift")

    active_scope = state.get("active_scope")
    if active_scope not in {"R1_same_reference", "R2_recorded_schema_external", "not_admitted"}:
        errors.append("invalid_active_scope")
    sig_ok = (
        state.get("recorded_schema_verified") is True
        and signature.get("dataset_id") == "USC-PSI-Lab/Humanoid-Everyday-G1"
        and signature.get("dataset_revision") == "6b6b599def75e3558b2736062e79bdbeec6cd224"
        and signature.get("model_input_order") == "[s43,v43,u28]"
        and coverage.get("record_schema", {}).get("status") == "verified_at_model_facing_stored_record_boundary_with_physical_semantics_limited"
    )
    if active_scope == "R1_same_reference":
        if state.get("physical_configuration_verified") is not True:
            errors.append("r1_requires_verified_physical_configuration")
        if not sig_ok:
            errors.append("r1_requires_verified_record_signature")
    elif active_scope == "R2_recorded_schema_external":
        if not sig_ok:
            errors.append("r2_requires_full_model_facing_record_signature_not_shape_only")
        if scope.get("amendment_id") != "RAW_OBS_0_14_FROZEN_MODEL_RECORDED_SCHEMA_EXTERNAL_EVALUATION":
            errors.append("r2_scope_amendment_missing_or_wrong")
        if scope.get("created_before_new_target_access") is not True:
            errors.append("r2_amendment_not_pre_outcome")
        if state.get("old_r1_same_hardware_confirmation_complete") is not False:
            errors.append("r2_cannot_mark_old_r1_complete")
        if state.get("physical_configuration_verified") is True:
            warnings.append("physical_configuration_evidence_present_but_active_scope_remains_r2")

    admission_n = admission.get("strict_independent_clusters")
    if not isinstance(admission_n, int) or admission_n < 0:
        errors.append("invalid_metadata_admission_cluster_count")
        admission_n = 0
    if state.get("strict_registered_local_clusters") != admission_n:
        errors.append("state_and_metadata_admission_cluster_count_mismatch")

    if stage in {"provenance_working", "prepared"}:
        if state.get("sample_sealed") is not False or state.get("forecast_sealed") is not False:
            errors.append("prepared_state_must_not_have_sample_or_forecast_seal")
        if state.get("outcome_read") is not False or state.get("score_once_complete") is not False:
            errors.append("prepared_state_must_not_have_outcome_access_or_score")
        if sample or forecast or score:
            errors.append("prepared_state_has_downstream_evidence_files")
        if admission_n < 270:
            warnings.append("strict_pool_below_fixed_n; remain data_blocked")
        elif active_scope == "not_admitted":
            warnings.append("candidate pool exists but scope is not admitted")

    if stage in {"metadata_admitted", "sample_sealed", "forecast_sealed", "score_once_complete"}:
        if admission_n < 270:
            errors.append("at_least_270_strict_independent_clusters_required")
        if not sig_ok:
            errors.append("metadata_admission_requires_verified_record_signature")
        if active_scope not in {"R1_same_reference", "R2_recorded_schema_external"}:
            errors.append("metadata_admission_requires_active_scope")
        if state.get("metadata_admitted") is not True:
            errors.append("metadata_admission_flag_missing")

    if stage in {"sample_sealed", "forecast_sealed", "score_once_complete"}:
        if state.get("sample_sealed") is not True or not isinstance(sample, dict):
            errors.append("sample_seal_missing")
        else:
            ids = sample.get("selected_cluster_ids")
            episodes = sample.get("selected_episode_ids")
            if sample.get("n_required_clusters") != 270 or sample.get("selected_clusters") != 270:
                errors.append("sample_size_not_fixed_270")
            if not isinstance(ids, list) or len(ids) != 270 or len(set(ids)) != 270:
                errors.append("sample_cluster_ids_not_270_unique")
            if not isinstance(episodes, list) or len(episodes) != 270 or len(set(episodes)) != 270:
                errors.append("sample_episode_ids_not_270_unique")
            if sample.get("one_episode_per_cluster") is not True:
                errors.append("sample_must_select_one_episode_per_cluster")
            if sample.get("selection_seed") != 20261012 or sample.get("selection_rule") != "sha256(seed|canonical_cluster_id)":
                errors.append("sample_selection_rule_or_seed_changed")
            if sample.get("model_sha256") != EXPECTED_MODEL:
                errors.append("sample_not_bound_to_frozen_model")
            if sample.get("sample_manifest_sha256") in (None, ""):
                errors.append("sample_manifest_hash_missing")
        if state.get("outcome_read") is not False:
            errors.append("outcomes_must_remain_unread_through_sample_seal")

    if stage in {"forecast_sealed", "score_once_complete"}:
        if state.get("forecast_sealed") is not True or not isinstance(forecast, dict):
            errors.append("forecast_seal_missing")
        else:
            if forecast.get("views") != EXPECTED_VIEWS:
                errors.append("forecast_views_do_not_match_three_locked_views")
            if forecast.get("n_clusters") != 270 or forecast.get("model_sha256") != EXPECTED_MODEL:
                errors.append("forecast_sample_or_model_binding_mismatch")
            if forecast.get("all_finite") is not True:
                errors.append("forecast_contains_missing_or_nonfinite_values")
            if forecast.get("prefix_only_read_log_sha256") in (None, ""):
                errors.append("prefix_only_access_log_missing")
            if forecast.get("native_conversion_sha256") in (None, ""):
                errors.append("native_conversion_hash_missing")
            if forecast.get("future_command_used") is not False or forecast.get("teacher_forcing_used") is not False:
                errors.append("forbidden_future_command_or_teacher_forcing")
            if forecast.get("sample_manifest_sha256") != (sample or {}).get("sample_manifest_sha256"):
                errors.append("forecast_sample_hash_mismatch")
        if state.get("outcome_read") is not False:
            errors.append("outcomes_must_remain_unread_until_forecast_seal")

    if stage == "score_once_complete":
        if state.get("outcome_read") is not True or state.get("score_once_complete") is not True:
            errors.append("score_state_does_not_record_single_outcome_access")
        if not isinstance(score, dict):
            errors.append("independent_score_receipt_missing")
        else:
            if score.get("score_attempt_count") != 1 or state.get("score_attempt_count") != 1:
                errors.append("score_once_gate_violation")
            if score.get("scorer_code_sha256") in (None, "") or score.get("metric_lock_sha256") in (None, ""):
                errors.append("scorer_or_metric_lock_hash_missing")
            if score.get("analysis_lock_sha256") in (None, ""):
                errors.append("analysis_ci_lock_hash_missing")
            if score.get("forecast_manifest_sha256") != (forecast or {}).get("forecast_manifest_sha256"):
                errors.append("score_forecast_hash_mismatch")
            if score.get("sample_manifest_sha256") != (sample or {}).get("sample_manifest_sha256"):
                errors.append("score_sample_hash_mismatch")
        if state.get("sample_reselection_after_outcome") is True:
            errors.append("sample_reselection_after_outcome_forbidden")
        if state.get("sample_revision_after_outcome", 0) != 0:
            errors.append("sample_revision_after_outcome_forbidden")
    else:
        if state.get("outcome_read") is True:
            errors.append("outcome_access_outside_score_once_stage")
        if state.get("score_attempt_count", 0) != 0 or state.get("score_once_complete") is True:
            errors.append("score_attempt_outside_score_once_stage")

    if state.get("parent_root_modified") is True:
        errors.append("parent_root_modified")
    if state.get("parent_snapshot_status") != "PASS":
        errors.append("parent_snapshot_not_passed")

    valid = not errors
    if valid and admission_n < 270:
        disposition = "VALID_BLOCKED_DATA"
        allowed_next = "obtain and metadata-register a permitted, unseen, compatible pool; do not score"
    elif valid and stage == "prepared":
        disposition = "VALID_PREPARED"
        allowed_next = "metadata admission"
    elif valid and stage == "metadata_admitted":
        disposition = "VALID_METADATA_ADMITTED"
        allowed_next = "single fixed-sample seal"
    elif valid and stage == "sample_sealed":
        disposition = "VALID_SAMPLE_SEALED"
        allowed_next = "three-view forecast seal"
    elif valid and stage == "forecast_sealed":
        disposition = "VALID_FORECAST_SEALED"
        allowed_next = "one independent scorer access"
    elif valid and stage == "score_once_complete":
        disposition = "VALID_SCORE_ONCE_COMPLETE"
        allowed_next = "closed; no reselection or second score"
    else:
        disposition = "INVALID_STATE"
        allowed_next = "repair evidence before continuing; do not read outcomes"
    return {"valid": valid, "disposition": disposition, "stage": stage,
            "errors": sorted(set(errors)), "warnings": sorted(set(warnings)),
            "allowed_next_action": allowed_next, "scope": active_scope,
            "strict_registered_local_clusters": admission_n,
            "n_required": 270, "old_r1_same_hardware_confirmation_complete": state.get("old_r1_same_hardware_confirmation_complete")}


def verify_root(root: Path) -> tuple[dict, list[str]]:
    errors: list[str] = []
    state = read_json(root / "protocol/WORK_STATE.json")
    admission = read_json(root / "catalog/ADMISSION_RESULT_R2.json")
    signature = read_json(root / "provenance/LEGACY_RECORD_SIGNATURE.json")
    scope = read_json(root / "protocol/SCOPE_AMENDMENT_R2.json")
    coverage = read_json(root / "provenance/REFERENCE_COVERAGE.json")
    parent_qa = read_json(root / "qa/PARENT_SNAPSHOT_VALIDATION.json")
    inherited = read_json(root / "protocol/INHERITED_PROTOCOL_LOCK.json")
    sample_path = root / "protocol/SAMPLE_SEAL.json"
    forecast_path = root / "protocol/FORECAST_SEAL.json"
    score_path = root / "results/INDEPENDENT_SCORE_RECEIPT.json"
    sample = read_json(sample_path) if sample_path.is_file() else None
    forecast = read_json(forecast_path) if forecast_path.is_file() else None
    score = read_json(score_path) if score_path.is_file() else None

    hash_refs = [
        ("legacy_record_signature_sha256", root / "provenance/LEGACY_RECORD_SIGNATURE.json"),
        ("scope_amendment_sha256", root / "protocol/SCOPE_AMENDMENT_R2.json"),
        ("r2_target_lock_sha256", root / "protocol/R2_TARGET_DOMAIN_LOCK.json"),
        ("reference_coverage_sha256", root / "provenance/REFERENCE_COVERAGE.json"),
        ("admission_result_sha256", root / "catalog/ADMISSION_RESULT_R2.json"),
        ("parent_snapshot_validation_sha256", root / "qa/PARENT_SNAPSHOT_VALIDATION.json"),
    ]
    hash_drift = False
    for field, path in hash_refs:
        expected = state.get(field)
        try:
            actual = sha256(path)
        except OSError:
            actual = None
        if not expected or expected != actual:
            errors.append(f"{field}_missing_or_drifted")
            hash_drift = True
    if state.get("parent_model_sha256") != inherited.get("frozen_model_sha256"):
        errors.append("inherited_model_hash_mismatch")
        hash_drift = True
    if parent_qa.get("status") != "PASS" or parent_qa.get("parent_modified") is not False:
        errors.append("parent_snapshot_validation_failed_or_modified")
    for item in inherited.get("parent_files", []):
        path = Path(item["absolute_path"])
        try:
            actual = sha256(path)
        except OSError:
            actual = None
        if actual != item.get("sha256"):
            errors.append(f"inherited_parent_hash_mismatch:{item.get('relative_path')}")
            hash_drift = True
    if score is None and state.get("score_once_complete") is True:
        errors.append("state_claims_score_without_receipt")
    result = assess_state(state, admission, signature, scope, coverage, sample, forecast, score, hash_drift)
    result["errors"] = sorted(set(result["errors"] + errors))
    result["valid"] = not result["errors"]
    if not result["valid"]:
        result["disposition"] = "INVALID_STATE"
        result["allowed_next_action"] = "repair evidence before continuing; do not read outcomes"
    result["parent_snapshot_sha256"] = sha256(root / "context/PARENT_SNAPSHOT.json")
    result["work_state_sha256"] = sha256(root / "protocol/WORK_STATE.json")
    result["admission_result_sha256"] = sha256(root / "catalog/ADMISSION_RESULT_R2.json")
    return result, errors


def load_admission_module(work_root: Path):
    spec = importlib.util.spec_from_file_location("metadata_admission_r2", ROOT / "tools/metadata_admission_r2.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    module.ROOT = work_root.resolve()
    return module


def test_candidate_row(base: Path, status: str | None = None,
                       exposure_fit: str = "false", source_group: str = "new-group") -> tuple[dict, list[str]]:
    ev = base / "intake_evidence.json"
    ev.write_text(json.dumps({
        "r2_compatibility_verified": True,
        "nonfinal_authority_verified": True,
        "prefix_label_custody": "separated_prefix_only_and_scorer_only",
        "session_independence_verified": True,
    }), encoding="utf-8")
    row = {
        "candidate_record_id": "candidate-1", "dataset_id": "registered-dataset",
        "dataset_revision": "rev-sha", "dataset_role": "prospective_nonfinal",
        "proposed_split": "prospective_eval", "allowed_use": "prospective_eval",
        "dataset_registry_entry_id": "registry-1", "registry_entry_sha256": "a" * 64,
        "explicit_nonfinal_pool_id": "pool-1", "registration_evidence_ref": str(ev),
        "source_group_id": source_group, "source_group_aliases": "[]", "session_id": "session-1",
        "session_aliases": "[]", "independence_evidence_ref": "independence.json",
        "session_independence_verified": "true", "exposure_fit": exposure_fit,
        "exposure_normalizer": "false", "exposure_cv": "false", "exposure_dev": "false",
        "related_dev_selection": "false", "final_test": "false", "heldout": "false",
        "record_signature_sha256": "b" * 64, "producer_code_sha256": "c" * 64,
        "source_schema_sha256": "a5754b47bb819533b2268a38159416f9e19d4e0a6c6f4a47254315d2fbb42cb0",
        "conversion_sha256": "9afc1f6b143e6e25c59423be6056ffd196a87f61de8fd35e10af991d7a3b5bed",
        "timestamp_basis": "nominal_30hz_frame_grid", "timestamp_evidence_ref": "clock.json",
        "physical_config_status": "unknown", "frame_index_path": "meta/frame.json",
        "frame_index_sha256": "d" * 64, "prefix_capture_path": "prefix/prefix.npz",
        "prefix_capture_sha256": "e" * 64, "future_labels_path": "labels/scorer.npz",
        "future_labels_sha256": "f" * 64, "hard_qa_anchor_count": "2",
    }
    if status:
        row["dataset_role"] = status
    module = load_admission_module(base)
    reasons = module.row_reasons(row, "b" * 64, {"old-alias", "old-source"})
    return row, reasons


def run_synthetic_tests() -> dict:
    checks: list[dict] = []

    def record(name: str, passed: bool, expected: str) -> None:
        checks.append({"case": name, "passed": bool(passed), "expected": expected})

    sig = {"dataset_id": "USC-PSI-Lab/Humanoid-Everyday-G1",
           "dataset_revision": "6b6b599def75e3558b2736062e79bdbeec6cd224",
           "model_input_order": "[s43,v43,u28]"}
    cover = {"record_schema": {"status": "verified_at_model_facing_stored_record_boundary_with_physical_semantics_limited"}}
    scope = {"amendment_id": "RAW_OBS_0_14_FROZEN_MODEL_RECORDED_SCHEMA_EXTERNAL_EVALUATION",
             "created_before_new_target_access": True}

    def base_state(stage: str = "prepared", n: int = 0, r2: bool = True) -> dict:
        return {"stage": stage, "n_required_clusters": 270, "strict_registered_local_clusters": n,
                "parent_model_sha256": EXPECTED_MODEL, "new_model_fits": 0, "gpu_actions": 0,
                "final_heldout_access": 0, "oracle_or_teacher_used": False,
                "active_scope": "R2_recorded_schema_external" if r2 else "R1_same_reference",
                "recorded_schema_verified": True, "physical_configuration_verified": False,
                "old_r1_same_hardware_confirmation_complete": False,
                "metadata_admitted": stage in {"metadata_admitted", "sample_sealed", "forecast_sealed", "score_once_complete"},
                "sample_sealed": stage in {"sample_sealed", "forecast_sealed", "score_once_complete"},
                "forecast_sealed": stage in {"forecast_sealed", "score_once_complete"},
                "outcome_read": stage == "score_once_complete",
                "score_once_complete": stage == "score_once_complete",
                "score_attempt_count": 1 if stage == "score_once_complete" else 0,
                "sample_reselection_after_outcome": False, "sample_revision_after_outcome": 0,
                "parent_root_modified": False, "parent_snapshot_status": "PASS"}

    empty_admission = {"strict_independent_clusters": 0}
    state = base_state("prepared", 0)
    result = assess_state(state, empty_admission, sig, scope, cover)
    record("prepared_resolved_reference_pool0_is_valid_blocked", result["valid"] and result["disposition"] == "VALID_BLOCKED_DATA", "valid blocked state; no score")

    state = base_state("metadata_admitted", 270, r2=False)
    state["strict_registered_local_clusters"] = 270
    result = assess_state(state, {"strict_independent_clusters": 270}, sig, scope, cover)
    record("r1_unknown_physical_configuration_rejected", "r1_requires_verified_physical_configuration" in result["errors"], "R1 needs exact physical evidence")

    state = base_state("metadata_admitted", 270)
    result = assess_state(state, {"strict_independent_clusters": 270}, sig, scope, cover)
    record("r2_full_record_signature_allows_physical_unknown", result["valid"], "R2 model-facing signature passes while R1 remains false")

    shape_cover = {"record_schema": {"status": "dimension_only"}}
    state = base_state("metadata_admitted", 270)
    result = assess_state(state, {"strict_independent_clusters": 270}, sig, scope, shape_cover)
    record("r2_shape_only_signature_rejected", "r2_requires_full_model_facing_record_signature_not_shape_only" in result["errors"], "shape-only match rejected")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _, reasons = test_candidate_row(tmp, status="final_test")
        record("protected_role_candidate_rejected", any("dataset_role_not" in x for x in reasons), "final role rejected")
        _, reasons = test_candidate_row(tmp, exposure_fit="unknown")
        record("unknown_exposure_candidate_rejected", "exposure_fit_not_verified_clear" in reasons, "unknown exposure rejected")
        _, reasons = test_candidate_row(tmp, source_group="old-alias")
        record("prior_alias_overlap_rejected", "prior_exposure_or_alias_overlap" in reasons, "old alias blocks candidate")
        _, reasons = test_candidate_row(tmp, status="unknown_role")
        record("unknown_role_candidate_rejected", any("dataset_role_not" in x for x in reasons), "unknown role rejected")

    with tempfile.TemporaryDirectory() as root_td, tempfile.TemporaryDirectory() as outside_td:
        root_tmp, outside_tmp = Path(root_td), Path(outside_td)
        row, _ = test_candidate_row(root_tmp)
        external_evidence = outside_tmp / "external_evidence.json"
        external_evidence.write_text(json.dumps({"r2_compatibility_verified": True}), encoding="utf-8")
        row["registration_evidence_ref"] = str(external_evidence)
        module = load_admission_module(root_tmp)
        reasons = module.row_reasons(row, "b" * 64, set())
        record("registration_evidence_path_escape_rejected", "registration_evidence_outside_work_root" in reasons, "evidence JSON references stay inside isolated work root")

    state = base_state("sample_sealed", 269)
    state["strict_registered_local_clusters"] = 269
    result = assess_state(state, {"strict_independent_clusters": 269}, sig, scope, cover, sample={})
    record("undersized_pool_rejects_sample", "at_least_270_strict_independent_clusters_required" in result["errors"], "N fixed at 270")

    state = base_state("forecast_sealed", 270)
    state["outcome_read"] = True
    result = assess_state(state, {"strict_independent_clusters": 270}, sig, scope, cover, sample=None, forecast={})
    record("unsealed_sample_blocks_label_read", "sample_seal_missing" in result["errors"], "no labels before sample seal")

    state = base_state("score_once_complete", 270)
    sample = {"n_required_clusters": 270, "selected_clusters": 270,
              "selected_cluster_ids": [f"c{i}" for i in range(270)],
              "selected_episode_ids": [f"e{i}" for i in range(270)],
              "one_episode_per_cluster": True, "selection_seed": 20261012,
              "selection_rule": "sha256(seed|canonical_cluster_id)",
              "model_sha256": EXPECTED_MODEL, "sample_manifest_sha256": "samplehash"}
    result = assess_state(state, {"strict_independent_clusters": 270}, sig, scope, cover, sample, None, {})
    record("missing_forecast_blocks_score", "forecast_seal_missing" in result["errors"], "no score without forecast seal")

    state = base_state("score_once_complete", 270)
    state["score_attempt_count"] = 2
    result = assess_state(state, {"strict_independent_clusters": 270}, sig, scope, cover, sample, {}, {"score_attempt_count": 2})
    record("repeat_score_attempt_rejected", "score_once_gate_violation" in result["errors"], "single scorer access")

    state = base_state("score_once_complete", 270)
    state["sample_reselection_after_outcome"] = True
    result = assess_state(state, {"strict_independent_clusters": 270}, sig, scope, cover, sample, {}, {"score_attempt_count": 1})
    record("sample_reselection_after_outcome_rejected", "sample_reselection_after_outcome_forbidden" in result["errors"], "outcomes cannot select a new sample")

    state = base_state("score_once_complete", 270)
    state["sample_revision_after_outcome"] = 1
    result = assess_state(state, {"strict_independent_clusters": 270}, sig, scope, cover, sample, {}, {"score_attempt_count": 1})
    record("sample_revision_after_outcome_rejected", "sample_revision_after_outcome_forbidden" in result["errors"], "sealed sample is immutable")

    state = base_state("metadata_admitted", 270)
    state["parent_model_sha256"] = "0" * 64
    result = assess_state(state, {"strict_independent_clusters": 270}, sig, scope, cover)
    record("model_hash_drift_rejected", "frozen_model_sha256_changed" in result["errors"], "model/scalers frozen")

    state = base_state("prepared", 0)
    state["n_required_clusters"] = 269
    result = assess_state(state, empty_admission, sig, scope, cover)
    record("n_reduction_rejected", "n_required_must_remain_270" in result["errors"], "do not lower N")

    state = base_state("metadata_admitted", 270)
    state["old_r1_same_hardware_confirmation_complete"] = True
    result = assess_state(state, {"strict_independent_clusters": 270}, sig, scope, cover)
    record("r2_does_not_upgrade_r1", "r2_cannot_mark_old_r1_complete" in result["errors"], "R2 never completes R1")

    state = base_state("prepared", 0)
    state["parent_root_modified"] = True
    result = assess_state(state, empty_admission, sig, scope, cover)
    record("parent_mutation_rejected", "parent_root_modified" in result["errors"], "parent is immutable")

    state = base_state("prepared", 0)
    state["final_heldout_access"] = 1
    result = assess_state(state, empty_admission, sig, scope, cover)
    record("protected_payload_access_rejected", "protected_payload_access_nonzero" in result["errors"], "no final/heldout access")

    state = base_state("forecast_sealed", 270)
    forecast = {"views": EXPECTED_VIEWS, "n_clusters": 270, "model_sha256": EXPECTED_MODEL,
                "all_finite": True, "prefix_only_read_log_sha256": "loghash",
                "native_conversion_sha256": "nativehash", "future_command_used": True,
                "teacher_forcing_used": False, "sample_manifest_sha256": "samplehash"}
    state["outcome_read"] = False
    result = assess_state(state, {"strict_independent_clusters": 270}, sig, scope, cover, sample, forecast)
    record("future_command_view_rejected", "forbidden_future_command_or_teacher_forcing" in result["errors"], "only three locked non-oracle views")

    state = base_state("forecast_sealed", 270)
    forecast["future_command_used"] = False
    forecast["views"] = ["OBS_PERSISTENCE", "ARX_ORACLE_COMMAND", "ARX_HOLD_COMMAND"]
    result = assess_state(state, {"strict_independent_clusters": 270}, sig, scope, cover, sample, forecast)
    record("oracle_view_name_rejected", "forecast_views_do_not_match_three_locked_views" in result["errors"], "oracle is forbidden")

    state = base_state("prepared", 0)
    result = assess_state(state, {"strict_independent_clusters": 0}, sig, scope, cover, hash_drift=True)
    record("hash_drift_rejected", "evidence_hash_drift" in result["errors"], "evidence hashes must stay pinned")

    state = base_state("score_once_complete", 270)
    state["outcome_read"] = True
    result = assess_state(state, {"strict_independent_clusters": 270}, sig, scope, cover,
                          sample, {}, {"score_attempt_count": 1})
    record("score_without_complete_forecast_receipt_rejected",
           any(x in result["errors"] for x in ("forecast_seal_missing", "forecast_views_do_not_match_three_locked_views", "prefix_only_access_log_missing")),
           "score follows a complete forecast receipt")

    state = base_state("metadata_admitted", 270)
    state["metadata_admitted"] = False
    result = assess_state(state, {"strict_independent_clusters": 270}, sig, scope, cover)
    record("missing_metadata_admission_flag_rejected", "metadata_admission_flag_missing" in result["errors"], "metadata admission is explicit")

    return {"status": "PASS" if all(x["passed"] for x in checks) else "FAIL",
            "synthetic_cases": len(checks), "passed": sum(x["passed"] for x in checks),
            "failed": [x for x in checks if not x["passed"]], "checks": checks,
            "data_payloads_read": 0, "new_labels_read": 0, "gpu_actions": 0,
            "new_model_fits": 0, "method": "pure synthetic dictionaries/temp JSON custody receipts; no project data files opened"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    if args.self_test:
        report = run_synthetic_tests()
    else:
        report, _ = verify_root(args.root.resolve())
    report["process_cpu_seconds"] = time.process_time() - started_cpu
    report["wall_seconds"] = time.perf_counter() - started_wall
    report["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    report["effective_cpu_limit"] = min(8, os.cpu_count() or 1)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("valid", report.get("status") == "PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
