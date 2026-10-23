"""Score the sealed six-cell Core A final predictions and paired cluster interval."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds
from sklearn.metrics import confusion_matrix, f1_score

from analysis_metrics import core_a_classification_metrics, sample_sd

OUT_ROOT = Path(__file__).resolve().parents[1]
PROJECT = OUT_ROOT.parents[2]
RUN_ROOT = PROJECT / os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z')
PROTOCOL = OUT_ROOT / "protocol"
LABELS = PROJECT / "repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/final_test_label_manifest.parquet"
LABEL_SHA256 = "72cb3c9b9a758193c19ab7099e09826383a190d37ecbe9ef4c45a46255857942"
BOOTSTRAP_SEED = 2026101901
BOOTSTRAP_DRAWS = 5000
SEEDS = (20260825, 20260826, 20260827)
PROGRAMS = ("A_DYNAMIC", "A_STATIC")
CLASS_COUNT = 120


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.part")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temp = path.with_name(f".{path.name}.{os.getpid()}.part")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_seal() -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]], str]:
    seal_path = OUT_ROOT / "predictions/CORE_A_PREDICTION_SEAL.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    digest = sha256_file(seal_path)
    if seal.get("status") != "SEALED_BEFORE_CORE_A_FINAL_LABEL_ACCESS" or seal.get("labels_or_targets_read") is not False:
        raise PermissionError("Core A final predictions are not sealed before labels")
    if seal.get("core") != "A" or seal.get("prediction_count") != 6 or seal.get("eligible_rows") != 602:
        raise ValueError("Core A final seal has the wrong matrix or population")
    records = {}
    for record in seal.get("prediction_records", []):
        key = (str(record["program"]), int(record["seed"]))
        if key in records:
            raise ValueError(f"duplicate Core A seal cell: {key}")
        records[key] = record
    expected = {(program, seed) for program in PROGRAMS for seed in SEEDS}
    if set(records) != expected:
        raise ValueError("Core A final seal does not contain exactly six fixed cells")
    manifest_path = Path(seal["prediction_manifest_path"])
    if sha256_file(manifest_path) != seal.get("prediction_manifest_sha256"):
        raise ValueError("prediction manifest hash differs from the seal")
    manifest_rows = read_csv(manifest_path)
    if len(manifest_rows) != 6:
        raise ValueError("prediction manifest does not contain six rows")
    manifest_by_cell = {(row["program"], int(row["seed"])): row for row in manifest_rows}
    for key, record in records.items():
        path = Path(record["prediction_path"])
        sidecar = Path(record["sidecar_path"])
        if sha256_file(path) != record["prediction_sha256"]:
            raise ValueError(f"sealed prediction changed: {path}")
        if sha256_file(sidecar) != record["sidecar_sha256"]:
            raise ValueError(f"sealed prediction sidecar changed: {sidecar}")
        row = manifest_by_cell.get(key)
        if row is None or row["prediction_sha256"] != record["prediction_sha256"] or row["sidecar_sha256"] != record["sidecar_sha256"]:
            raise ValueError(f"prediction manifest and seal disagree: {key}")
    return seal, records, digest


def read_locked_labels(labels_source: Path, expected_ids: np.ndarray) -> list[dict[str, Any]]:
    dataset = ds.dataset(str(labels_source), format="parquet")
    table = dataset.to_table(
        columns=["episode_index", "split", "contiguous_label_index"],
        filter=ds.field("episode_index").isin(np.asarray(expected_ids, dtype=np.int64).tolist())
        & (ds.field("split") == "final_test"),
    )
    return table.to_pylist()


def _classify(logits: np.ndarray, truth: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    result = core_a_classification_metrics(logits, truth, class_count=CLASS_COUNT)
    prediction = np.argmax(np.asarray(logits, dtype=np.float64), axis=1)
    return result, prediction


def _macro_f1_fixed_classes(truth: np.ndarray, prediction: np.ndarray, classes: np.ndarray) -> float:
    encoded = np.asarray(truth, dtype=np.int64) * CLASS_COUNT + np.asarray(prediction, dtype=np.int64)
    confusion = np.bincount(encoded, minlength=CLASS_COUNT * CLASS_COUNT).reshape(CLASS_COUNT, CLASS_COUNT)
    true_support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    denominator = true_support + predicted_support
    f1 = np.divide(2.0 * np.diag(confusion), denominator, out=np.zeros(CLASS_COUNT, dtype=np.float64), where=denominator != 0)
    return float(f1[classes].mean())


def _bootstrap_contrast(
    labels: np.ndarray,
    predictions: dict[tuple[str, int], np.ndarray],
    cluster_ids: list[str],
) -> dict[str, dict[str, float]]:
    indices_by_cluster: dict[str, list[int]] = {}
    for index, cluster in enumerate(cluster_ids):
        indices_by_cluster.setdefault(cluster, []).append(index)
    cluster_names = np.asarray(sorted(indices_by_cluster), dtype=str)
    if len(cluster_names) < 2:
        raise ValueError("paired cluster bootstrap requires at least two source-assignment groups")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    top1_draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    f1_draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    classes = np.arange(CLASS_COUNT, dtype=np.int64)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(cluster_names, size=len(cluster_names), replace=True)
        rows = np.fromiter((i for cluster in sampled.tolist() for i in indices_by_cluster[cluster]), dtype=np.int64)
        per_seed_top1 = []
        per_seed_f1 = []
        truth = labels[rows]
        for seed in SEEDS:
            dynamic = predictions[("A_DYNAMIC", seed)][rows]
            static = predictions[("A_STATIC", seed)][rows]
            per_seed_top1.append(float(np.mean(dynamic == truth) - np.mean(static == truth)))
            dynamic_f1 = _macro_f1_fixed_classes(truth, dynamic, classes)
            static_f1 = _macro_f1_fixed_classes(truth, static, classes)
            per_seed_f1.append(float(dynamic_f1 - static_f1))
        top1_draws[draw] = float(np.mean(per_seed_top1))
        f1_draws[draw] = float(np.mean(per_seed_f1))
    def interval(values: np.ndarray) -> dict[str, float]:
        low, high = np.quantile(values, [0.025, 0.975])
        return {"ci95_low": float(low), "ci95_high": float(high)}
    return {"top1_accuracy": interval(top1_draws), "macro_f1": interval(f1_draws)}


def main() -> int:
    seal, records, seal_sha = verify_seal()
    lock_path = PROTOCOL / "FINAL_STUDY_LOCK.json"
    freeze_path = PROTOCOL / "FREEZE_RECEIPT.json"
    access_path = PROTOCOL / "FINAL_ACCESS_OPENED.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    access = json.loads(access_path.read_text(encoding="utf-8"))
    lock_sha = sha256_file(lock_path)
    freeze_sha = sha256_file(freeze_path)
    if seal.get("study_lock_sha256") != lock_sha:
        raise PermissionError("prediction seal does not bind the current final study lock")
    if freeze.get("status") != "FROZEN_BEFORE_FINAL_ACCESS" or freeze.get("study_lock_sha256") != lock_sha:
        raise PermissionError("freeze receipt is invalid or differs from the study lock")
    if access.get("status") != "OPENED_BY_USER_AUTHORIZED_TASK" or access.get("core_a_access_authorized") is not True or access.get("core_b_access_authorized") is not False:
        raise PermissionError("Core A access must be open and Core B access must remain closed")
    if access.get("study_lock_sha256") != lock_sha or access.get("freeze_receipt_sha256") != freeze_sha:
        raise PermissionError("final access receipt does not bind the frozen protocol")
    if seal.get("final_access_receipt_sha256") != sha256_file(access_path):
        raise PermissionError("prediction seal does not bind the current final access receipt")
    for relative, expected in freeze.get("code_sha256", {}).items():
        path = OUT_ROOT / "code" / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise PermissionError(f"frozen code hash mismatch: {relative}")

    expected_rows = read_csv(PROTOCOL / "CORE_A_FINAL_IDENTITY.csv")
    expected_ids = np.asarray([int(row["episode_id"]) for row in expected_rows], dtype=np.int64)
    canonical_ids = [row["canonical_id"] for row in expected_rows]
    if len(expected_ids) != 602 or len(set(expected_ids.tolist())) != 602 or len(set(canonical_ids)) != 602:
        raise ValueError("locked Core A population identity is invalid")
    input_hashes = freeze.get("input_sha256", {})
    if input_hashes.get("core_a_final_video_index") != sha256_file(PROTOCOL / "CORE_A_FINAL_VIDEO_INDEX.csv"):
        raise ValueError("final video index differs from the frozen input hash")
    cluster_path = PROTOCOL / "CORE_A_FINAL_BOOTSTRAP_CLUSTER_MAP.csv"
    if input_hashes.get("core_a_bootstrap_cluster_map") != sha256_file(cluster_path):
        raise ValueError("bootstrap cluster map differs from the frozen input hash")
    if sha256_file(cluster_path) != lock.get("core_a", {}).get("bootstrap_cluster_map_sha256"):
        raise ValueError("bootstrap cluster map differs from the study lock")
    cluster_rows = read_csv(cluster_path)
    cluster_lookup = {int(row["episode_index"]): row["cluster_id_sha256"] for row in cluster_rows}
    if len(cluster_rows) != 602 or len(cluster_lookup) != 602 or set(cluster_lookup) != set(expected_ids.tolist()):
        raise ValueError("bootstrap cluster map does not match the locked Core A row set")
    cluster_ids = [cluster_lookup[int(value)] for value in expected_ids]
    if len(set(cluster_ids)) != 602:
        raise ValueError("locked Core A source-assignment cluster map must retain 602 distinct groups")

    class_map_path = OUT_ROOT / "configs/CORE_A_CLASS_MAP.json"
    class_map_sha = sha256_file(class_map_path)
    if class_map_sha != lock.get("core_a", {}).get("label_map_sha256") or class_map_sha != lock.get("core_a", {}).get("primary_class_order_sha256"):
        raise ValueError("training-derived Core A class map/output order differs from the study lock")
    class_map = json.loads(class_map_path.read_text(encoding="utf-8"))
    class_indices = [int(row["class_index"]) for row in class_map.get("classes", [])]
    if class_map.get("class_count") != CLASS_COUNT or class_indices != list(range(CLASS_COUNT)):
        raise ValueError("Core A class map does not define the locked 0..119 output order")
    run_receipt_path = Path(seal["run_receipt_path"])
    extraction_receipt_path = Path(seal["extraction_receipt_path"])
    if sha256_file(run_receipt_path) != seal.get("run_receipt_sha256"):
        raise ValueError("Core A run receipt differs from the prediction seal")
    if sha256_file(extraction_receipt_path) != seal.get("extraction_receipt_sha256"):
        raise ValueError("Core A extraction receipt differs from the prediction seal")
    run_receipt = json.loads(run_receipt_path.read_text(encoding="utf-8"))
    extraction_receipt = json.loads(extraction_receipt_path.read_text(encoding="utf-8"))
    if extraction_receipt.get("row_count") != 602 or extraction_receipt.get("video_decoder_invocations") != 602:
        raise ValueError("Core A final extraction did not process exactly 602 videos")
    if extraction_receipt.get("selected_source_frames_returned") != 602 * 32:
        raise ValueError("Core A final extraction did not return exactly 32 source frames per video")
    if sum(extraction_receipt.get("decoder_invocations_by_backend", {}).values()) != 602:
        raise ValueError("Core A decoder backend counts do not sum to 602")
    if run_receipt.get("classifier_model_cells") != 6 or run_receipt.get("classifier_batch_forward_calls") != 78 or run_receipt.get("classifier_episode_predictions") != 3612:
        raise ValueError("Core A fixed model forward and prediction counts differ from the six-cell contract")

    # Load and validate every sealed output before touching the final label file.
    logits_by_cell: dict[tuple[str, int], np.ndarray] = {}
    for program in PROGRAMS:
        for seed in SEEDS:
            record = records[(program, seed)]
            with np.load(record["prediction_path"], allow_pickle=False) as bundle:
                logits = np.asarray(bundle["logits"], dtype=np.float64)
                ids = bundle["episode_index"].astype(np.int64)
            if not np.array_equal(ids, expected_ids) or logits.shape != (602, CLASS_COUNT) or not np.isfinite(logits).all():
                raise ValueError(f"sealed Core A logits changed or misaligned: {program}/{seed}")
            logits_by_cell[(program, seed)] = logits

    labels_source = LABELS
    if Path(lock.get("final_label_manifest_path", "")).resolve() != labels_source.resolve():
        raise ValueError("final label manifest path differs from the study lock")
    exposure_path = OUT_ROOT / "qa/EXPOSURE_LEDGER.json"
    exposure_path.parent.mkdir(parents=True, exist_ok=True)
    byte_read_utc = utc_now()
    byte_count = labels_source.stat().st_size
    exposure = {
        "schema_version": 1,
        "protocol_id": lock["protocol_id"],
        "status": "CORE_A_FINAL_LABEL_MANIFEST_HASH_IN_PROGRESS",
        "prior_development_exposure": {
            "core_a_label_vector_source": "accepted Batch-48 predictions_best.npz files, one fixed dynamic-seed label vector per scoring pass",
            "core_a_label_vector_read_runs": 3,
            "core_a_rows_per_read": 617,
            "core_a_label_rows_read_total": 1851,
            "core_b_target_archive_path": str(PROJECT / "reports/benchmarks/CORE_B_TASK_SPECIALIZATION_WAYPOINT_V2_20261013T064245Z/data/DEV_TARGETS_SCORER_ONLY.npz"),
            "core_b_target_archive_sha256": "233a0cdb31d1ddd7cea6ecfaa5e9d02e9c42e9822af7fe488c7fe5a40febee27",
            "core_b_target_archive_byte_reads": 3,
            "core_b_target_archive_size_bytes": 435083,
            "core_b_target_array_decodes": 2,
            "core_b_target_rows_materialized_total": 370,
            "core_b_target_values_decoded_successful_scoring_passes": 310800,
            "core_b_failed_hash_gate_target_values_decoded": 0,
            "core_b_final_target_access": 0,
        },
        "final_access": {
            "core_a_final_video_index_rows": 602,
            "core_a_final_video_decoder_invocations": extraction_receipt["video_decoder_invocations"],
            "core_a_decoder_invocations_by_backend": extraction_receipt["decoder_invocations_by_backend"],
            "core_a_selected_source_frames_returned": extraction_receipt["selected_source_frames_returned"],
            "core_a_opencv_sequential_frame_reads": extraction_receipt["opencv_sequential_frame_reads"],
            "core_a_decord_internal_decoded_frame_count": extraction_receipt["decord_internal_decoded_frame_count"],
            "core_a_frozen_encoder_clip_forwards": run_receipt["feature_extractor_forwards"],
            "core_a_final_classifier_cells": run_receipt["classifier_model_cells"],
            "core_a_classifier_batch_forward_calls": run_receipt["classifier_batch_forward_calls"],
            "core_a_final_prediction_rows": run_receipt["classifier_episode_predictions"],
            "core_a_final_label_manifest_path": str(labels_source),
            "core_a_final_label_manifest_expected_sha256": LABEL_SHA256,
            "core_a_final_label_manifest_size_bytes": byte_count,
            "core_a_final_label_manifest_byte_hash_read_utc": byte_read_utc,
            "core_a_final_label_rows_materialized": 0,
            "core_a_final_label_values_decoded": 0,
            "core_a_final_score_exposed": False,
            "core_b_final_video_or_target_access": 0,
            "core_b_final_target_values_decoded": 0,
            "core_b_final_predictions": 0,
            "first_final_payload_access_utc": seal.get("first_final_payload_access_utc"),
            "first_final_score_exposure_utc": None,
        },
        "retries": {
            "dev_replay_attempts": ["attempt_01", "attempt_02", "attempt_03", "attempt_04", "attempt_05", "attempt_06"],
            "dev_score_passes": [
                {"attempt":"attempt_06_score_retry_00","outcome":"target archive hash gate failed after Core A dev-label read; compressed bytes read; no Core B target arrays decoded"},
                {"attempt":"attempt_06_score_retry_01","outcome":"Core A dev scored; Core B target arrays decoded once with model residual normalizer scales; metric acceptance failed"},
                {"attempt":"attempt_06_score_retry_02","outcome":"Core A dev scored; Core B target arrays decoded once with locked metric scales; metric reconciliation passed"}
            ],
            "final_score_attempts": 0,
        },
        "prediction_seal_sha256": seal_sha,
    }
    atomic_json(exposure_path, exposure)
    observed_label_sha = sha256_file(labels_source)
    exposure["final_access"]["core_a_final_label_manifest_observed_sha256"] = observed_label_sha
    if observed_label_sha != LABEL_SHA256 or lock.get("expected_final_label_manifest_sha256") != LABEL_SHA256:
        exposure["status"] = "FINAL_LABEL_MANIFEST_HASH_MISMATCH"
        atomic_json(exposure_path, exposure)
        raise ValueError("final label manifest SHA-256 differs from the locked source")
    if byte_count != lock.get("final_label_manifest_size_bytes"):
        exposure["status"] = "FINAL_LABEL_MANIFEST_SIZE_MISMATCH"
        atomic_json(exposure_path, exposure)
        raise ValueError("final label manifest size differs from the locked source")
    first_score_utc = utc_now()
    exposure["status"] = "CORE_A_FINAL_SCORE_IN_PROGRESS"
    exposure["final_access"]["first_final_score_exposure_utc"] = first_score_utc
    atomic_json(exposure_path, exposure)
    label_rows = read_locked_labels(labels_source, expected_ids)
    returned_ids = np.asarray([int(row["episode_index"]) for row in label_rows], dtype=np.int64)
    if len(label_rows) != 602 or len(set(returned_ids.tolist())) != 602 or set(returned_ids.tolist()) != set(expected_ids.tolist()):
        exposure["status"] = "FINAL_LABEL_JOIN_MISMATCH_AFTER_ACCESS"
        exposure["final_access"]["label_rows_returned_before_error"] = len(label_rows)
        atomic_json(exposure_path, exposure)
        raise ValueError("final label join does not return the exact locked 602 Core A rows")
    if any(row["split"] != "final_test" for row in label_rows):
        exposure["status"] = "FINAL_LABEL_SPLIT_MISMATCH_AFTER_ACCESS"
        atomic_json(exposure_path, exposure)
        raise ValueError("final label join returned a non-final split row")
    labels_by_id = {int(row["episode_index"]): int(row["contiguous_label_index"]) for row in label_rows}
    labels = np.asarray([labels_by_id[int(value)] for value in expected_ids], dtype=np.int64)
    if np.any((labels < 0) | (labels >= CLASS_COUNT)):
        exposure["status"] = "FINAL_LABEL_CLASS_RANGE_MISMATCH_AFTER_ACCESS"
        exposure["final_access"]["core_a_final_label_rows_materialized"] = len(label_rows)
        exposure["final_access"]["core_a_final_label_values_decoded"] = len(labels)
        atomic_json(exposure_path, exposure)
        raise ValueError("final labels fall outside the fixed 120-class map")
    exposure["final_access"]["core_a_final_label_rows_materialized"] = len(label_rows)
    exposure["final_access"]["core_a_final_label_values_decoded"] = len(labels)
    exposure["final_access"]["core_a_final_score_exposed"] = True
    atomic_json(exposure_path, exposure)

    predictions: dict[tuple[str, int], np.ndarray] = {}
    per_seed: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    classes = np.arange(CLASS_COUNT, dtype=np.int64)
    for program in PROGRAMS:
        for seed in SEEDS:
            logits = logits_by_cell[(program, seed)]
            metrics, predicted = _classify(logits, labels)
            predictions[(program, seed)] = predicted
            per_seed.append({"program": program, "seed": seed, "support": 602, **metrics})
            matrix = confusion_matrix(labels, predicted, labels=classes)
            for true_class in range(CLASS_COUNT):
                support = int(matrix[true_class].sum())
                class_rows.append({
                    "program": program,
                    "seed": seed,
                    "class_index": true_class,
                    "support": support,
                    "recall": float(matrix[true_class, true_class] / support) if support else 0.0,
                })
                for predicted_class in range(CLASS_COUNT):
                    count = int(matrix[true_class, predicted_class])
                    if count:
                        confusion_rows.append({"program": program, "seed": seed, "true_class": true_class, "predicted_class": predicted_class, "count": count})

    summaries = []
    for program in PROGRAMS:
        rows = [row for row in per_seed if row["program"] == program]
        summaries.append({
            "program": program,
            "seed_count": 3,
            "support_per_seed": 602,
            "top1_accuracy_mean": float(np.mean([row["top1_accuracy"] for row in rows])),
            "top1_accuracy_sample_sd": sample_sd([row["top1_accuracy"] for row in rows]),
            "top5_accuracy_mean": float(np.mean([row["top5_accuracy"] for row in rows])),
            "top5_accuracy_sample_sd": sample_sd([row["top5_accuracy"] for row in rows]),
            "macro_f1_mean": float(np.mean([row["macro_f1"] for row in rows])),
            "macro_f1_sample_sd": sample_sd([row["macro_f1"] for row in rows]),
        })
    seed_contrasts = []
    for seed in SEEDS:
        dynamic = predictions[("A_DYNAMIC", seed)]
        static = predictions[("A_STATIC", seed)]
        seed_contrasts.append({
            "seed": seed,
            "top1_accuracy_difference": float(np.mean(dynamic == labels) - np.mean(static == labels)),
            "macro_f1_difference": float(f1_score(labels, dynamic, labels=classes, average="macro", zero_division=0) - f1_score(labels, static, labels=classes, average="macro", zero_division=0)),
            "support": 602,
        })
    intervals = _bootstrap_contrast(labels, predictions, cluster_ids)
    contrast_rows = []
    for metric, field in (("top1_accuracy", "top1_accuracy_difference"), ("macro_f1", "macro_f1_difference")):
        values = [row[field] for row in seed_contrasts]
        contrast_rows.append({
            "core": "A",
            "contrast": "A_DYNAMIC_minus_A_STATIC",
            "metric": metric,
            "mean_difference": float(np.mean(values)),
            "sample_sd_across_seeds": sample_sd(values),
            **intervals[metric],
            "interval_type": "paired_cluster_bootstrap_percentile_95_conditional_on_fitted_models",
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "cluster_count": len(set(cluster_ids)),
            "support_per_seed": 602,
        })
    loo_rows = []
    for omitted in SEEDS:
        retained = [row for row in seed_contrasts if row["seed"] != omitted]
        loo_rows.append({
            "analysis": "core_a_dynamic_minus_static_leave_one_seed_out",
            "omitted_seed": omitted,
            "included_seeds": ";".join(str(seed) for seed in SEEDS if seed != omitted),
            "top1_accuracy_difference_mean": float(np.mean([row["top1_accuracy_difference"] for row in retained])),
            "macro_f1_difference_mean": float(np.mean([row["macro_f1_difference"] for row in retained])),
            "episodes_per_seed": 602,
            "cluster_count": len(set(cluster_ids)),
        })

    results_root = OUT_ROOT / "results"
    atomic_csv(results_root / "CORE_A_FINAL_PER_SEED.csv", per_seed)
    atomic_csv(results_root / "CORE_A_FINAL_SUMMARY.csv", summaries)
    atomic_csv(results_root / "FINAL_PAIRED_CONTRASTS.csv", contrast_rows)
    atomic_csv(results_root / "CORE_A_FINAL_SEED_CONTRASTS.csv", seed_contrasts)
    atomic_csv(results_root / "CORE_A_FINAL_CLASS_RECALL.csv", class_rows)
    atomic_csv(results_root / "CORE_A_FINAL_CONFUSION_NONZERO.csv", confusion_rows or [{"program": "none", "seed": "none", "true_class": "none", "predicted_class": "none", "count": 0}])
    atomic_csv(results_root / "LOO_AND_SUPPORT.csv", loo_rows)
    result = {
        "schema_version": 1,
        "protocol_id": lock["protocol_id"],
        "status": "CORE_A_FINAL_SCORED",
        "scored_utc": utc_now(),
        "prediction_seal_path": str((OUT_ROOT / "predictions/CORE_A_PREDICTION_SEAL.json").resolve()),
        "prediction_seal_sha256": seal_sha,
        "study_lock_sha256": lock_sha,
        "label_manifest_path": str(labels_source.resolve()),
        "label_manifest_sha256": observed_label_sha,
        "label_manifest_byte_hash_read_utc": byte_read_utc,
        "video_decoder_invocations_by_backend": extraction_receipt["decoder_invocations_by_backend"],
        "selected_source_frames_returned": extraction_receipt["selected_source_frames_returned"],
        "opencv_sequential_frame_reads": extraction_receipt["opencv_sequential_frame_reads"],
        "decord_internal_decoded_frame_count": extraction_receipt["decord_internal_decoded_frame_count"],
        "classifier_batch_forward_calls": run_receipt["classifier_batch_forward_calls"],
        "classifier_episode_predictions": run_receipt["classifier_episode_predictions"],
        "first_final_score_exposure_utc": first_score_utc,
        "label_rows_materialized": len(label_rows),
        "labels_decoded": len(labels),
        "fixed_class_count": CLASS_COUNT,
        "seed_count": len(SEEDS),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "cluster_count": len(set(cluster_ids)),
        "cluster_map_sha256": sha256_file(PROTOCOL / "CORE_A_FINAL_BOOTSTRAP_CLUSTER_MAP.csv"),
        "result_files": [
            str((results_root / name).resolve()) for name in (
                "CORE_A_FINAL_PER_SEED.csv", "CORE_A_FINAL_SUMMARY.csv", "FINAL_PAIRED_CONTRASTS.csv",
                "CORE_A_FINAL_SEED_CONTRASTS.csv", "CORE_A_FINAL_CLASS_RECALL.csv",
                "CORE_A_FINAL_CONFUSION_NONZERO.csv", "LOO_AND_SUPPORT.csv",
            )
        ],
        "per_seed": per_seed,
        "summary": summaries,
        "seed_contrasts": seed_contrasts,
        "paired_intervals": intervals,
        "heldout_confirmation": False,
        "evidence_mode": "COURSE_FINAL_SAME_SOURCE_WITH_METADATA_DEVIATION",
    }
    receipt_path = OUT_ROOT / "results/CORE_A_FINAL_SCORE_RECEIPT.json"
    atomic_json(receipt_path, result)
    exposure["status"] = "CORE_A_FINAL_SCORE_COMPLETE"
    exposure["final_access"]["core_a_final_score_exposed"] = True
    exposure["final_access"]["core_a_final_label_rows_materialized"] = len(label_rows)
    exposure["final_access"]["core_a_final_label_values_decoded"] = len(labels)
    exposure["retries"]["final_score_attempts"] = 1
    exposure["final_result_receipt_path"] = str(receipt_path.resolve())
    atomic_json(exposure_path, exposure)
    print(json.dumps({"status": result["status"], "support": 602, "bootstrap_draws": BOOTSTRAP_DRAWS, "paired_intervals": intervals}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
