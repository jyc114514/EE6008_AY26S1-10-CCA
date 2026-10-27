"""Run the frozen six-cell Core A final inference without reading final labels."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

import core_a_extract
import core_a_predict

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


def verify_watcher_output_root(path: Path) -> None:
    path = Path(path)
    if not path.exists():
        return
    if not path.is_dir():
        raise FileExistsError(f"watcher output root is not a directory: {path}")
    entries = list(path.iterdir())
    if len(entries) != 1 or entries[0].name != "watcher_child.log" or not entries[0].is_file() or entries[0].is_symlink():
        raise FileExistsError(f"final output root contains prior or unexpected data: {path}")


def verify_runtime_environment() -> None:
    required = (PROJECT / "envs/jyc/lib").resolve()
    configured = [
        Path(value).resolve()
        for value in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if value
    ]
    if required not in configured:
        raise RuntimeError(f"LD_LIBRARY_PATH must include the project runtime library directory: {required}")


def verify_receipts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    lock_path = PROTOCOL / "FINAL_STUDY_LOCK.json"
    freeze_path = PROTOCOL / "FREEZE_RECEIPT.json"
    access_path = PROTOCOL / "FINAL_ACCESS_OPENED.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    access = json.loads(access_path.read_text(encoding="utf-8"))
    if lock.get("protocol_id") != PROTOCOL_ID or freeze.get("protocol_id") != PROTOCOL_ID or access.get("protocol_id") != PROTOCOL_ID:
        raise PermissionError("Core A final protocol IDs do not match")
    lock_sha = sha256_file(lock_path)
    freeze_sha = sha256_file(freeze_path)
    if lock.get("selection_status") != "locked_by_task_protocol":
        raise PermissionError("final matrix is not locked by task protocol")
    if freeze.get("status") != "FROZEN_BEFORE_FINAL_ACCESS" or freeze.get("study_lock_sha256") != lock_sha:
        raise PermissionError("freeze receipt is incomplete or does not bind the current study lock")
    if access.get("status") != "OPENED_BY_USER_AUTHORIZED_TASK" or access.get("core_a_access_authorized") is not True:
        raise PermissionError("Core A final access receipt is not open")
    if access.get("core_b_access_authorized") is not False:
        raise PermissionError("Core B final access must remain closed in this branch")
    if access.get("study_lock_sha256") != lock_sha or access.get("freeze_receipt_sha256") != freeze_sha:
        raise PermissionError("final access receipt does not bind the frozen protocol")
    expected_code = freeze.get("code_sha256", {})
    for relative, expected in expected_code.items():
        path = OUT_ROOT / "code" / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise PermissionError(f"frozen code hash mismatch: {relative}")
    return lock, freeze, access


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--attempt", default="attempt_01")
    parser.add_argument("--expected-physical-gpu", type=int, default=3)
    args = parser.parse_args()
    verify_runtime_environment()
    if args.device == "cuda":
        observed = os.environ.get("GPU_WATCHER_PHYSICAL_GPU")
        if observed != str(args.expected_physical_gpu):
            raise RuntimeError(f"watcher physical GPU={observed!r}; expected {args.expected_physical_gpu}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
    lock, freeze, access = verify_receipts()
    matrix = read_csv(PROTOCOL / "FINAL_MODEL_MATRIX.csv")
    cells = [row for row in matrix if row["core"] == "A"]
    expected_cells = {(program, seed) for program in ("A_DYNAMIC", "A_STATIC") for seed in SEEDS}
    actual_cells = {(row["program"], int(row["seed"])) for row in cells}
    if actual_cells != expected_cells or len(cells) != 6:
        raise ValueError("Core A model matrix is not the frozen six-cell set")

    attempt_root = RUN_ROOT / "features/core_a/final" / args.attempt
    verify_watcher_output_root(attempt_root)
    extraction_dir = attempt_root / "extracted_features"
    if extraction_dir.exists():
        raise FileExistsError(f"final extraction path already exists: {extraction_dir}")
    config_row = next(row for row in cells if row["program"] == "A_DYNAMIC" and int(row["seed"]) == SEEDS[0])
    extraction = core_a_extract.run_extraction(
        manifest_path=PROTOCOL / "CORE_A_FINAL_VIDEO_INDEX.csv",
        config_path=Path(config_row["config_path"]),
        config_sha256=config_row["config_sha256"],
        checkpoint_sha256=core_a_extract.CHECKPOINT_SHA256 if hasattr(core_a_extract, "CHECKPOINT_SHA256") else "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d",
        run_id=f"{PROTOCOL_ID}_CORE_A_{args.attempt}",
        output_dir=extraction_dir,
        mode="final",
        final_access_receipt=PROTOCOL / "FINAL_ACCESS_OPENED.json",
        freeze_receipt=PROTOCOL / "FREEZE_RECEIPT.json",
        device_name=args.device,
        expected_rows=602,
        output_root=RUN_ROOT,
        protocol_root=OUT_ROOT,
    )
    feature_manifests = {
        "A_DYNAMIC": extraction_dir / "feature_manifest_dynamic.csv",
        "A_STATIC": extraction_dir / "feature_manifest_static_repeated_first_frame.csv",
    }
    prediction_root = RUN_ROOT / "predictions/core_a/final" / args.attempt
    records = []
    for row in cells:
        program = row["program"]
        seed = int(row["seed"])
        output = prediction_root / program / f"seed{seed}.npz"
        result = core_a_predict.predict_manifest(
            manifest_path=feature_manifests[program],
            checkpoint_path=Path(row["checkpoint_path"]),
            checkpoint_sha256=row["checkpoint_sha256"],
            run_id=f"{PROTOCOL_ID}_{program}_seed{seed}_{args.attempt}",
            output_path=output,
            device=args.device,
            expected_rows=602,
            output_root=RUN_ROOT,
        )
        records.append({
            "core": "A",
            "program": program,
            "seed": seed,
            "checkpoint_path": row["checkpoint_path"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "prediction_path": str(output.resolve()),
            "prediction_sha256": result["output_sha256"],
            "sidecar_path": str(output.with_suffix(".json").resolve()),
            "sidecar_sha256": sha256_file(output.with_suffix(".json")),
            "rows": 602,
            "shape": result["logits_shape"],
            "device": args.device,
        })
    receipt = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "ALL_SIX_CORE_A_PREDICTIONS_COMPLETE_LABELS_UNREAD",
        "attempt": args.attempt,
        "completed_utc": utc_now(),
        "runtime": {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "runtime_model": lock.get("runtime", {}).get("runtime_model", "not_introspectable"),
            "physical_gpu": int(args.expected_physical_gpu) if args.device == "cuda" else None,
            "logical_device": "cuda:0" if args.device == "cuda" else "cpu",
        },
        "study_lock_sha256": sha256_file(PROTOCOL / "FINAL_STUDY_LOCK.json"),
        "freeze_receipt_sha256": sha256_file(PROTOCOL / "FREEZE_RECEIPT.json"),
        "final_access_receipt_sha256_after_first_access_timestamp": sha256_file(PROTOCOL / "FINAL_ACCESS_OPENED.json"),
        "extraction_receipt_path": str((extraction_dir / "EXTRACTION_RECEIPT.json").resolve()),
        "extraction_receipt_sha256": sha256_file(extraction_dir / "EXTRACTION_RECEIPT.json"),
        "extraction": extraction,
        "feature_extractor_forwards": 1204,
        "classifier_model_cells": 6,
        "classifier_batch_forward_calls": 78,
        "classifier_episode_predictions": 3612,
        "predictions": records,
        "labels_or_targets_read": False,
        "final_score_exposed": False,
    }
    atomic_json(prediction_root / "CORE_A_FINAL_RUN_RECEIPT.json", receipt)
    print(json.dumps({"status": receipt["status"], "cells": len(records), "rows_per_cell": 602, "first_final_payload_access_utc": extraction.get("first_final_payload_access_utc")}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
