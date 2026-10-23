"""CPU acceptance for the full 36-cell D3-EQ formal training matrix."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
BENCH_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
MATRIX_PATH = EQ_ROOT / "protocol/D3_EQ_MATRIX_36.csv"
LOCK_PATH = EQ_ROOT / "protocol/D3_EQ_PROTOCOL_LOCKED.json"
TARGET_CACHE = EQ_ROOT / "d3/train_targets_trainonly.npz"
TARGET_CACHE_MANIFEST = EQ_ROOT / "d3/TRAIN_TARGET_CACHE_MANIFEST.json"
EXPECTED_CONFIGS = ("TS00-D", "TS10-D", "TS01-D", "TS11-D", "TS11-S", "STATE-CTRL")
EXPECTED_SEEDS = (20261001, 20261002, 20261003)
EXPECTED_HORIZONS = (30, 60)
RUN_STATUS_COMPLETION_FIELDS = (
    "config",
    "seed",
    "horizon",
    "epochs",
    "optimizer_steps",
    "expected_optimizer_steps",
    "protocol_hash",
    "checkpoint_last_sha256",
    "init_hash",
    "encoder_updated",
    "final_test_access",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    part.replace(path)


def run_status_completion_matches(
    run_status: dict[str, Any],
    run_complete: dict[str, Any],
    locked_protocol_hash: str,
) -> bool:
    """Validate terminal status, accounting for the trainer's dict-merge output.

    The trainer writes {"status": "complete", ..., **complete}; since
    RUN_COMPLETE.status is "PASS", the serialized RUN_STATUS.status becomes
    "PASS". Accept that encoding only when both records are PASS and every
    locked identity/provenance field agrees.
    """
    if run_complete.get("status") != "PASS":
        return False
    if run_status.get("status") not in {"complete", "PASS"}:
        return False
    if run_complete.get("protocol_hash") != locked_protocol_hash:
        return False
    return all(
        field in run_status
        and field in run_complete
        and run_status[field] == run_complete[field]
        for field in RUN_STATUS_COMPLETION_FIELDS
    )


def main() -> int:
    torch.set_num_threads(1)
    lock_hash = sha256_file(LOCK_PATH)
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED" or lock.get("final_test_access") != 0:
        raise AssertionError("D3-EQ lock status/final gate is invalid")
    target_manifest = json.loads(TARGET_CACHE_MANIFEST.read_text(encoding="utf-8"))
    if sha256_file(TARGET_CACHE) != target_manifest.get("target_cache_sha256"):
        raise AssertionError("train-only target cache hash changed after training")
    seeds = json.loads((EQ_ROOT / "protocol/D3_EQ_SEED_REGISTRY.json").read_text(encoding="utf-8"))["seeds"]
    expected = {(config, seed, horizon) for config in EXPECTED_CONFIGS for seed in EXPECTED_SEEDS for horizon in EXPECTED_HORIZONS}
    matrix = list(csv.DictReader(MATRIX_PATH.open(newline="", encoding="utf-8")))
    if len(matrix) != 36 or {(row["config"], int(row["seed"]), int(row["horizon_steps"])) for row in matrix} != expected:
        raise AssertionError("locked D3-EQ matrix does not cover the exact 36 cells")

    accepted = []
    total_formal_gpu_hours = 0.0
    ledger_path = EQ_ROOT / "JOB_LEDGER.jsonl"
    if ledger_path.is_file():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("phase") == "FORMAL":
                total_formal_gpu_hours += float(record.get("lease_gpu_hours", 0.0))

    for row in matrix:
        config, seed, horizon = row["config"], int(row["seed"]), int(row["horizon_steps"])
        run_id = row["run_id"]
        root = EQ_ROOT / "d3/runs" / run_id
        complete_path = root / "RUN_COMPLETE.json"
        status_path = root / "RUN_STATUS.json"
        environment_path = root / "RUN_ENVIRONMENT.json"
        checkpoint = root / "checkpoint_last.pt"
        if not all(path.is_file() for path in (complete_path, status_path, environment_path, checkpoint)):
            raise FileNotFoundError(f"formal run artifact missing for {run_id}")
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        status = json.loads(status_path.read_text(encoding="utf-8"))
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        if not run_status_completion_matches(status, complete, lock_hash):
            raise AssertionError(f"formal run terminal status/provenance mismatch: {run_id}")
        if (complete.get("config"), int(complete.get("seed", -1)), int(complete.get("horizon", -1))) != (config, seed, horizon):
            raise AssertionError(f"formal run identity mismatch: {run_id}")
        if int(complete.get("epochs", -1)) != 40 or int(complete.get("optimizer_steps", -1)) != 11120 or int(complete.get("expected_optimizer_steps", -1)) != 11120:
            raise AssertionError(f"formal epoch/step count mismatch: {run_id}")
        if complete.get("final_test_access") != 0 or complete.get("encoder_updated") is not False:
            raise AssertionError(f"formal final-test/encoder gate failed: {run_id}")
        if complete.get("protocol_hash") != lock_hash or environment.get("protocol_hash") != lock_hash:
            raise AssertionError(f"run lock hash mismatch: {run_id}")
        if sha256_file(checkpoint) != complete.get("checkpoint_last_sha256"):
            raise AssertionError(f"epoch40 checkpoint hash mismatch: {run_id}")
        expected_init = seeds[str(seed)]["init_state_hash"]
        if complete.get("init_hash") != expected_init or environment.get("init_state_hash") != expected_init:
            raise AssertionError(f"matched initial state mismatch: {run_id}")
        if environment.get("target_cache_sha256") != target_manifest["target_cache_sha256"]:
            raise AssertionError(f"train-only target cache provenance mismatch: {run_id}")
        if environment.get("source_target_scope") != "primary_train_parquet_only; no dev target array loaded":
            raise AssertionError(f"train/dev target isolation marker mismatch: {run_id}")
        with np.load(TARGET_CACHE, allow_pickle=False) as targets:
            if len(targets["anchor_id"]) != int(environment["train_windows"]) or targets["targets_native_60"].shape != (17757, 60, 28):
                raise AssertionError(f"train-only target cache population mismatch: {run_id}")
        epoch_rows = list(csv.DictReader((root / "EPOCH_LOG.csv").open(newline="", encoding="utf-8")))
        seen_epochs = {int(item["epoch"]) for item in epoch_rows}
        if seen_epochs != set(range(1, 41)) or any(not np.isfinite(float(item["weighted_loss_sum_over_seen_rows"])) for item in epoch_rows):
            raise AssertionError(f"epoch log missing/nonfinite entries: {run_id}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(payload.get("epoch", -1)) != 40 or int(payload.get("optimizer_steps", -1)) != 11120 or payload.get("protocol_hash") != lock_hash:
            raise AssertionError(f"last checkpoint epoch/step/lock mismatch: {run_id}")
        if any(not torch.isfinite(value).all() for value in payload["model"].values() if torch.is_floating_point(value)):
            raise AssertionError(f"last checkpoint contains non-finite parameter: {run_id}")
        accepted.append({
            "run_id": run_id,
            "config": config,
            "seed": seed,
            "horizon_steps": horizon,
            "epochs": 40,
            "optimizer_steps": 11120,
            "init_hash": expected_init,
            "checkpoint_sha256": complete["checkpoint_last_sha256"],
            "run_status_terminal_status": status["status"],
            "checkpoint_reloaded_cpu": True,
            "finite_weights": True,
            "final_test_access": 0,
        })
    if len(accepted) != 36:
        raise AssertionError(f"accepted formal training runs={len(accepted)}/36")

    artifact_bytes = sum(path.stat().st_size for path in EQ_ROOT.rglob("*") if path.is_file())
    artifact_gib = artifact_bytes / (1024**3)
    if artifact_gib > 50.0:
        raise AssertionError(f"D3-EQ output root exceeds 50 GiB: {artifact_gib:.3f}")
    if total_formal_gpu_hours + float(json.loads((EQ_ROOT / "qa/D3_EQ_STAGE0_ACCEPTANCE.json").read_text(encoding="utf-8"))["stage0_gpu_hours_actual"]) > 48.0:
        raise AssertionError("actual Stage0 + formal GPU hours exceed 48-hour cap")

    acceptance = {
        "status": "PASS",
        "protocol_id": lock["protocol_id"],
        "training_cells": len(accepted),
        "expected_cells": 36,
        "all_epochs": 40,
        "optimizer_steps_per_cell": 11120,
        "formal_gpu_hours_actual": total_formal_gpu_hours,
        "stage0_gpu_hours_actual": json.loads((EQ_ROOT / "qa/D3_EQ_STAGE0_ACCEPTANCE.json").read_text(encoding="utf-8"))["stage0_gpu_hours_actual"],
        "total_gpu_hours_actual": total_formal_gpu_hours + json.loads((EQ_ROOT / "qa/D3_EQ_STAGE0_ACCEPTANCE.json").read_text(encoding="utf-8"))["stage0_gpu_hours_actual"],
        "output_root_bytes": artifact_bytes,
        "output_root_gib": artifact_gib,
        "train_target_cache_only": True,
        "run_status_terminal_encoding": {
            "observed_statuses": sorted({cell["run_status_terminal_status"] for cell in accepted}),
            "completion_fields_cross_checked": list(RUN_STATUS_COMPLETION_FIELDS),
            "serialization_note": "d3_train_eq.py writes {'status':'complete', **complete}; complete.status='PASS' overwrites the intended label. PASS is accepted only when RUN_COMPLETE is PASS and all listed fields match.",
        },
        "final_test_access": 0,
        "old_root_write": False,
        "cells": accepted,
    }
    atomic_json(EQ_ROOT / "qa/D3_EQ_TRAINING_ACCEPTANCE.json", acceptance)
    print(json.dumps({k: acceptance[k] for k in ("status", "training_cells", "formal_gpu_hours_actual", "total_gpu_hours_actual", "output_root_gib", "final_test_access")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
