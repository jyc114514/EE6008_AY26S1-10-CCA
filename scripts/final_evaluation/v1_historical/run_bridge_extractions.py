"""Run the fixed train/dev feature smoke or 64-row bridges inside the watcher."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT_ROOT = PROJECT / "reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
RUN_ROOT = PROJECT / os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z')
CODE = OUT_ROOT / "code"
CORE_A_FIELDS = ("episode_index", "video_path")
CORE_B_FIELDS = ("dataset_role", "canonical_id", "source_key", "literal_source_key", "episode_id", "task_id", "anchor_id", "anchor_frame", "cache_index", "video_path", "parquet_path")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def select_rows(source: Path, destination: Path, fields: tuple[str, ...], stage0: bool) -> Path:
    if not stage0:
        return source
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError(f"bridge manifest header mismatch at {source}")
        first = next(reader, None)
    if first is None:
        raise ValueError(f"empty bridge manifest: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerow(first)
        stream.flush()
        os.fsync(stream.fileno())
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0", action="store_true", help="extract only the first preselected bridge row per Core")
    args = parser.parse_args()
    physical_gpu = os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "")
    if physical_gpu != "3":
        raise RuntimeError(f"this task was authorized for physical GPU3; watcher selected {physical_gpu!r}")
    attempt = "stage0" if args.stage0 else "bridge_attempt_01"
    status_path = RUN_ROOT / "replay" / ("BRIDGE_STAGE0_STATUS.json" if args.stage0 else "BRIDGE_EXTRACTION_STATUS.json")
    receipt_path = RUN_ROOT / "replay" / ("BRIDGE_STAGE0_RECEIPT.json" if args.stage0 else "BRIDGE_EXTRACTION_RECEIPT.json")
    core_a_source = RUN_ROOT / "replay/core_a/core_a_bridge_manifest.csv"
    core_b_source = RUN_ROOT / "replay/core_b/core_b_bridge_manifest.csv"
    core_a_manifest = select_rows(core_a_source, RUN_ROOT / "replay/core_a/core_a_bridge_stage0_manifest.csv", CORE_A_FIELDS, args.stage0)
    core_b_manifest = select_rows(core_b_source, RUN_ROOT / "replay/core_b/core_b_bridge_stage0_manifest.csv", CORE_B_FIELDS, args.stage0)
    core_a_config = PROJECT / "repo/ee6008-humanoid-vjepa/reports/benchmarks/CORE_A_BATCH48_ABLATION_MATRIX_V3_20260830/configs/standard_dynamic.yaml"
    feature_attempt_root = RUN_ROOT / "features" / attempt
    core_a_out = feature_attempt_root / "core_a"
    core_b_out = feature_attempt_root / "core_b"
    core_a_run_id = "EE6008_FINAL_TEST_V1_20261019T041642Z_CORE_A_" + ("STAGE0" if args.stage0 else "BRIDGE")
    core_a_command = [
        sys.executable, str(CODE / "core_a_extract.py"),
        "--manifest", str(core_a_manifest), "--config", str(core_a_config),
        "--config-sha256", "e60ac789add8e7f529aef5c677760d8d61f6ee54305e8dcf601a8d9c4e2f2d93",
        "--checkpoint-sha256", "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d",
        "--run-id", core_a_run_id, "--output-dir", str(core_a_out), "--output-root", str(RUN_ROOT),
        "--mode", "dev", "--device", "cuda", "--expected-rows", "1" if args.stage0 else "64",
    ]
    core_b_command = [
        sys.executable, str(CODE / "core_b_extract_grid.py"), "--mode", "bridge",
        "--input-manifest", str(core_b_manifest),
        "--output-array", str(core_b_out / "grid.npy"),
        "--output-manifest", str(core_b_out / "grid_manifest.json"),
        "--device", "cuda",
    ]
    if args.stage0:
        core_b_command.append("--stage0")
    names = ("core_a_dynamic_static", "core_b_d3_grid")
    commands = (core_a_command, core_b_command)
    record: dict[str, object] = {
        "schema_version": 1, "protocol_id": "EE6008_FINAL_TEST_V1_20261019T041642Z",
        "stage0_smoke": bool(args.stage0), "attempt": attempt, "status": "RUNNING",
        "started_utc": utc_now(), "physical_gpu": 3,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"),
        "logical_device": "cuda:0", "new_training_runs": 0, "stages": [],
        "bridge_manifests": {"core_a": str(core_a_manifest), "core_b": str(core_b_manifest)},
    }
    atomic_json(status_path, record)
    try:
        for name, command in zip(names, commands, strict=True):
            source = CODE / ("core_a_extract.py" if name.startswith("core_a") else "core_b_extract_grid.py")
            stage = {"name": name, "command": command, "code_sha256": sha256_file(source), "started_utc": utc_now(), "status": "RUNNING"}
            record["stages"].append(stage)  # type: ignore[union-attr]
            atomic_json(status_path, record)
            started = time.perf_counter()
            subprocess.run(command, cwd=PROJECT, check=True, env=os.environ.copy())
            stage["completed_utc"] = utc_now()
            stage["elapsed_seconds"] = time.perf_counter() - started
            stage["status"] = "PASS"
            atomic_json(status_path, record)
        record["status"] = "PASS"
        record["completed_utc"] = utc_now()
        record["artifacts"] = {
            "core_a_extraction_receipt": str(core_a_out / "EXTRACTION_RECEIPT.json"),
            "core_b_grid_manifest": str(core_b_out / "grid_manifest.json"),
        }
        atomic_json(receipt_path, record)
        atomic_json(status_path, record)
        print(json.dumps({"status": "PASS", "attempt": attempt, "stages": names, "physical_gpu": 3}, sort_keys=True), flush=True)
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
