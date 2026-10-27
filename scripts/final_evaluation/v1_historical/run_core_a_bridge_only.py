"""Run the fixed 64-row Core A bridge and six predictor checks through watcher."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT_ROOT = PROJECT / "reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
RUN_ROOT = PROJECT / os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z')
CODE = OUT_ROOT / "code"
JOB_ROOT = RUN_ROOT / "features/core_a/bridge_attempt_01"
ATTEMPT = JOB_ROOT / "artifacts"
CONFIG = PROJECT / "repo/ee6008-humanoid-vjepa/reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830/configs/standard_dynamic.yaml"
CONFIG_SHA = "e60ac789add8e7f529aef5c677760d8d61f6ee54305e8dcf601a8d9c4e2f2d93"
ENCODER_SHA = "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.part")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def main() -> int:
    if os.environ.get("GPU_WATCHER_PHYSICAL_GPU") != "3":
        raise RuntimeError("Core A bridge requires watcher-selected physical GPU3")
    if ATTEMPT.exists():
        raise FileExistsError(ATTEMPT)
    ATTEMPT.mkdir(parents=True)
    input_manifest = RUN_ROOT / "replay/core_a/core_a_bridge_manifest.csv"
    status_path = ATTEMPT / "BRIDGE_STATUS.json"
    receipt_path = ATTEMPT / "BRIDGE_RECEIPT.json"
    record: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": "EE6008_FINAL_TEST_V1_20261019T041642Z",
        "status": "RUNNING",
        "started_utc": utc_now(),
        "physical_gpu": 3,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
        "logical_device": "cuda:0",
        "split_support": {"train": 32, "dev": 32},
        "new_training_runs": 0,
        "final_payload_read": False,
        "stages": [],
    }
    atomic_json(status_path, record)
    try:
        extraction_command = [
            sys.executable, str(CODE / "core_a_extract.py"),
            "--manifest", str(input_manifest), "--config", str(CONFIG),
            "--config-sha256", CONFIG_SHA, "--checkpoint-sha256", ENCODER_SHA,
            "--run-id", "EE6008_FINAL_TEST_V1_20261019T041642Z_CORE_A_BRIDGE64",
            "--output-dir", str(ATTEMPT / "features"), "--output-root", str(RUN_ROOT),
            "--mode", "dev", "--device", "cuda", "--expected-rows", "64",
        ]
        stage = {"name": "fresh_dynamic_static_features", "command": extraction_command, "started_utc": utc_now(), "status": "RUNNING"}
        record["stages"].append(stage)
        atomic_json(status_path, record)
        start = time.perf_counter()
        subprocess.run(extraction_command, cwd=PROJECT, check=True, env=os.environ.copy())
        receipt = ATTEMPT / "features/EXTRACTION_RECEIPT.json"
        stage.update({"status": "PASS", "completed_utc": utc_now(), "elapsed_seconds": time.perf_counter() - start, "output_sha256": sha256_file(receipt)})
        atomic_json(status_path, record)

        with (OUT_ROOT / "protocol/FINAL_MODEL_MATRIX.csv").open("r", encoding="utf-8-sig", newline="") as stream:
            matrix = list(csv.DictReader(stream))
        core_a = sorted([row for row in matrix if row["core"] == "A" and row["program"] in {"A_DYNAMIC", "A_STATIC"}], key=lambda row: (row["program"], int(row["seed"])))
        if len(core_a) != 6:
            raise ValueError("Core A bridge requires exactly six model cells")
        prediction_records = []
        for row in core_a:
            program = row["program"]
            seed = int(row["seed"])
            condition = "dynamic" if program == "A_DYNAMIC" else "static_repeated_first_frame"
            command = [
                sys.executable, str(CODE / "core_a_predict.py"),
                "--manifest", str(ATTEMPT / "features" / f"feature_manifest_{condition}.csv"),
                "--checkpoint", row["checkpoint_path"], "--checkpoint-sha256", row["checkpoint_sha256"],
                "--run-id", f"EE6008_FINAL_TEST_V1_20261019T041642Z_{program}_{seed}_BRIDGE64",
                "--output", str(ATTEMPT / "predictions" / program / f"seed{seed}.npz"),
                "--output-root", str(RUN_ROOT), "--device", "cuda", "--expected-rows", "64",
            ]
            stage = {"name": f"predict_{program}_{seed}", "command": command, "started_utc": utc_now(), "status": "RUNNING"}
            record["stages"].append(stage)
            atomic_json(status_path, record)
            start = time.perf_counter()
            subprocess.run(command, cwd=PROJECT, check=True, env=os.environ.copy())
            prediction_path = Path(command[command.index("--output") + 1])
            sidecar = prediction_path.with_suffix(".json")
            receipt = json.loads(sidecar.read_text(encoding="utf-8"))
            stage.update({"status": "PASS", "completed_utc": utc_now(), "elapsed_seconds": time.perf_counter() - start, "output_sha256": receipt["output_sha256"], "sidecar_sha256": sha256_file(sidecar)})
            prediction_records.append({"program": program, "seed": seed, "path": str(prediction_path), "sha256": receipt["output_sha256"], "sidecar_sha256": sha256_file(sidecar), "rows": 64, "checkpoint_sha256": row["checkpoint_sha256"], "feature_manifest_sha256": receipt["feature_manifest_sha256"], "episode_index_sha256": receipt["episode_index_sha256"], "logits_sha256": receipt["logits_sha256"]})
            atomic_json(status_path, record)
        record["status"] = "PASS_PENDING_CPU_BRIDGE_COMPARISON"
        record["completed_utc"] = utc_now()
        record["feature_encoder_forward_rows"] = 64 * 2
        record["classifier_forward_rows"] = 6 * 64
        record["prediction_records"] = prediction_records
        atomic_json(receipt_path, record)
        atomic_json(status_path, record)
        print(json.dumps({"status": record["status"], "feature_rows": 64, "model_cells": 6, "physical_gpu": 3}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        record["status"] = "FAILED"
        record["failed_utc"] = utc_now()
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)
        atomic_json(status_path, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
