"""Validate and seal all six Core A final logits before label access."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

OUT_ROOT = Path(__file__).resolve().parents[1]
PROJECT = OUT_ROOT.parents[2]
RUN_ROOT = PROJECT / os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z')
PROTOCOL = OUT_ROOT / "protocol"
PROTOCOL_ID = "EE6008_FINAL_TEST_V1_20261019T041642Z"
SEEDS = (20260825, 20260826, 20260827)


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_feature_manifests(extraction: dict[str, Any], extraction_dir: Path) -> dict[str, tuple[str, Path]]:
    feature_manifest_by_program = {
        "A_DYNAMIC": ("dynamic", Path(extraction_dir) / "feature_manifest_dynamic.csv"),
        "A_STATIC": ("static_repeated_first_frame", Path(extraction_dir) / "feature_manifest_static_repeated_first_frame.csv"),
    }
    for program, (condition, manifest_path) in feature_manifest_by_program.items():
        expected_manifest_sha = extraction.get("feature_manifest_sha256", {}).get(condition)
        if not expected_manifest_sha or sha256_file(manifest_path) != expected_manifest_sha:
            raise ValueError(f"Core A {program} feature manifest differs from extraction receipt")
    return feature_manifest_by_program


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", default="attempt_01")
    args = parser.parse_args()
    prediction_root = RUN_ROOT / "predictions/core_a/final" / args.attempt
    lock_path = PROTOCOL / "FINAL_STUDY_LOCK.json"
    freeze_path = PROTOCOL / "FREEZE_RECEIPT.json"
    access_path = PROTOCOL / "FINAL_ACCESS_OPENED.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    access = json.loads(access_path.read_text(encoding="utf-8"))
    if lock.get("protocol_id") != PROTOCOL_ID or freeze.get("protocol_id") != PROTOCOL_ID or access.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("protocol ID mismatch")
    lock_sha = sha256_file(lock_path)
    freeze_sha = sha256_file(freeze_path)
    if freeze.get("status") != "FROZEN_BEFORE_FINAL_ACCESS" or freeze.get("study_lock_sha256") != lock_sha:
        raise ValueError("freeze receipt does not bind current study lock")
    if access.get("status") != "OPENED_BY_USER_AUTHORIZED_TASK" or access.get("core_a_access_authorized") is not True or access.get("core_b_access_authorized") is not False:
        raise ValueError("Core A open/Core B closed access state is invalid")
    if access.get("study_lock_sha256") != lock_sha or access.get("freeze_receipt_sha256") != freeze_sha:
        raise ValueError("final access receipt does not bind frozen protocol")
    for relative, expected in freeze.get("code_sha256", {}).items():
        path = OUT_ROOT / "code" / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen code hash mismatch: {relative}")

    run_path = prediction_root / "CORE_A_FINAL_RUN_RECEIPT.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("status") != "ALL_SIX_CORE_A_PREDICTIONS_COMPLETE_LABELS_UNREAD" or run.get("labels_or_targets_read") is not False or run.get("final_score_exposed") is not False:
        raise ValueError("final run receipt does not attest label-free inference")
    extraction_path = Path(run["extraction_receipt_path"])
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    first_access_utc = access.get("first_final_payload_access_utc")
    if not first_access_utc or extraction.get("first_final_payload_access_utc") != first_access_utc:
        raise ValueError("first final payload access timestamp is not bound consistently")
    if extraction.get("mode") != "final" or extraction.get("row_count") != 602:
        raise ValueError("Core A extraction receipt has the wrong mode or row count")
    extraction_dir = extraction_path.parent
    feature_manifest_by_program = verify_feature_manifests(extraction, extraction_dir)

    identities = read_csv(PROTOCOL / "CORE_A_FINAL_IDENTITY.csv")
    episode_ids = np.asarray([int(row["episode_id"]) for row in identities], dtype=np.int64)
    canonical_ids = [row["canonical_id"] for row in identities]
    if len(identities) != 602 or len(set(episode_ids.tolist())) != 602 or len(set(canonical_ids)) != 602:
        raise ValueError("locked Core A identity map is not exactly 602 unique rows")
    matrix = read_csv(PROTOCOL / "FINAL_MODEL_MATRIX.csv")
    cells = [row for row in matrix if row["core"] == "A"]
    records = run.get("predictions", [])
    by_cell = {(row["program"], int(row["seed"])): row for row in records}
    expected = {(program, seed) for program in ("A_DYNAMIC", "A_STATIC") for seed in SEEDS}
    if len(records) != 6 or set(by_cell) != expected:
        raise ValueError("Core A final run does not contain the fixed six cells")
    matrix_by_cell = {(row["program"], int(row["seed"])): row for row in cells}
    if set(matrix_by_cell) != expected:
        raise ValueError("frozen model matrix differs from the Core A six-cell contract")

    prediction_manifest: list[dict[str, Any]] = []
    for key in sorted(expected):
        record = by_cell[key]
        matrix_row = matrix_by_cell[key]
        path = Path(record["prediction_path"])
        sidecar_path = Path(record["sidecar_path"])
        if sha256_file(path) != record["prediction_sha256"] or sha256_file(sidecar_path) != record["sidecar_sha256"]:
            raise ValueError(f"prediction or sidecar checksum mismatch: {key}")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        condition, expected_manifest_path = feature_manifest_by_program[key[0]]
        expected_manifest_sha = extraction["feature_manifest_sha256"][condition]
        if sidecar.get("checkpoint_sha256") != matrix_row["checkpoint_sha256"] or sidecar.get("feature_rows") != 602 or sidecar.get("logits_shape") != [602, 120]:
            raise ValueError(f"prediction provenance/shape mismatch: {key}")
        if Path(sidecar.get("feature_manifest_path", "")).resolve() != expected_manifest_path.resolve() or sidecar.get("feature_manifest_sha256") != expected_manifest_sha:
            raise ValueError(f"prediction input manifest differs from the sealed extraction source: {key}")
        with np.load(path, allow_pickle=False) as bundle:
            if set(bundle.files) != {"episode_index", "logits"}:
                raise ValueError(f"prediction bundle has unexpected fields: {key}")
            ids = bundle["episode_index"].astype(np.int64)
            logits = np.asarray(bundle["logits"])
        if not np.array_equal(ids, episode_ids) or logits.shape != (602, 120) or logits.dtype != np.dtype("<f8") or not np.isfinite(logits).all():
            raise ValueError(f"prediction IDs, dtype, shape, or finite checks failed: {key}")
        prediction_manifest.append({
            "core": "A",
            "program": key[0],
            "seed": key[1],
            "checkpoint_sha256": matrix_row["checkpoint_sha256"],
            "feature_manifest_path": str(expected_manifest_path.resolve()),
            "feature_manifest_sha256": sidecar["feature_manifest_sha256"],
            "prediction_path": str(path.resolve()),
            "prediction_sha256": record["prediction_sha256"],
            "sidecar_path": str(sidecar_path.resolve()),
            "sidecar_sha256": record["sidecar_sha256"],
            "rows": 602,
            "shape": "[602,120]",
            "dtype": "float64",
            "canonical_ids_sha256": hashlib.sha256("\n".join(canonical_ids).encode()).hexdigest(),
            "labels_read": False,
        })

    manifest_path = OUT_ROOT / "predictions/CORE_A_PREDICTION_MANIFEST.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.part")
    fields = list(prediction_manifest[0])
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(prediction_manifest)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, manifest_path)
    seal = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "SEALED_BEFORE_CORE_A_FINAL_LABEL_ACCESS",
        "sealed_utc": utc_now(),
        "split": "final_test",
        "attempt": args.attempt,
        "core": "A",
        "prediction_count": len(prediction_manifest),
        "expected_model_cells": 6,
        "eligible_rows": 602,
        "prediction_manifest_path": str(manifest_path.resolve()),
        "prediction_manifest_sha256": sha256_file(manifest_path),
        "run_receipt_path": str(run_path.resolve()),
        "run_receipt_sha256": sha256_file(run_path),
        "extraction_receipt_path": str(extraction_path.resolve()),
        "extraction_receipt_sha256": sha256_file(extraction_path),
        "study_lock_sha256": lock_sha,
        "freeze_receipt_sha256": freeze_sha,
        "final_access_receipt_sha256": sha256_file(access_path),
        "first_final_payload_access_utc": first_access_utc,
        "prediction_records": prediction_manifest,
        "labels_or_targets_read": False,
        "final_score_exposed": False,
    }
    seal_path = OUT_ROOT / "predictions/CORE_A_PREDICTION_SEAL.json"
    atomic_json(seal_path, seal)
    print(json.dumps({"status": seal["status"], "cells": len(prediction_manifest), "rows": 602, "seal_sha256": sha256_file(seal_path)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
