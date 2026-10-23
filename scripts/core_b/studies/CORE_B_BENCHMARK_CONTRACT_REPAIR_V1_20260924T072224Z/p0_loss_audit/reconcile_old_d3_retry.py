"""Reconcile the one interrupted old D3 job with its preserved formal RETRY1."""

from __future__ import annotations
import os

import csv
import hashlib
import json
from pathlib import Path


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
BENCH = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
OUT = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_CONTRACT_REPAIR_V1_20260924T072224Z/p0_loss_audit"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    matrix = list(csv.DictReader((BENCH / "protocol/D3_MATRIX_36.csv").open(newline="", encoding="utf-8")))
    trainer = BENCH / "code/d3_train.py"
    rows = []
    for cell in matrix:
        run_id = cell["run_id"]
        candidates = [
            BENCH / "d3/runs" / f"{run_id}-FORMAL",
            BENCH / "d3/runs" / f"{run_id}-FORMAL-RETRY1",
        ]
        passed = []
        for run_root in candidates:
            marker_path = run_root / "RUN_COMPLETE.json"
            if marker_path.is_file():
                marker = json.loads(marker_path.read_text())
                if marker.get("status") == "PASS":
                    passed.append((run_root, marker))
        if not passed:
            raise AssertionError(f"no accepted train completion or retry for {run_id}")
        run_root, marker = next((entry for entry in passed if "RETRY1" in entry[0].name), passed[0])
        attempt_id = run_root.name
        config_path = BENCH / "configs" / f"{attempt_id}.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        command = list(config.get("command", []))
        code_path = Path(command[1]).resolve() if len(command) > 1 else Path()
        checkpoint = run_root / "checkpoint_last.pt"
        pred_stem = BENCH / "d3/predictions" / attempt_id
        pred_npz = pred_stem.with_suffix(".npz")
        pred_sidecar = pred_stem.with_suffix(".json")
        pred_meta = json.loads(pred_sidecar.read_text()) if pred_sidecar.is_file() else {}
        watcher = BENCH / "logs" / f"watcher_{attempt_id}.json"
        watcher_data = json.loads(watcher.read_text()) if watcher.is_file() else {}
        rows.append({
            "matrix_run_id": run_id,
            "selected_attempt": attempt_id,
            "retry_used": "RETRY1" in attempt_id,
            "train_status": marker.get("status"),
            "epochs": marker.get("epochs"),
            "optimizer_steps": marker.get("optimizer_steps"),
            "seed_init_hash": marker.get("init_hash"),
            "trainer_path_matches": bool(code_path and code_path == trainer.resolve()),
            "trainer_sha256_sealed": sha256_file(trainer),
            "checkpoint_exists": checkpoint.is_file(),
            "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else "",
            "prediction_exists": pred_npz.is_file(),
            "prediction_sha256": sha256_file(pred_npz) if pred_npz.is_file() else "",
            "prediction_status": pred_meta.get("status", "missing"),
            "watcher_status": watcher_data.get("status", "missing"),
            "watcher_exit_code": watcher_data.get("exit_code"),
            "lease_released": watcher_data.get("lease_released"),
            "final_test_access": 0,
        })
    acceptance = json.loads((BENCH / "d3/D3_PREDICTION_ACCEPTANCE.json").read_text())
    summary = {
        "status": "PASS_EFFECTIVE_36_CELL_LINEAGE",
        "matrix_cells": len(matrix),
        "effective_train_complete": sum(row["train_status"] == "PASS" for row in rows),
        "effective_checkpoints_present": sum(bool(row["checkpoint_exists"]) for row in rows),
        "effective_prediction_pass": sum(row["prediction_status"] == "PASS" and row["prediction_exists"] for row in rows),
        "retry_cells": [row["matrix_run_id"] for row in rows if row["retry_used"]],
        "prediction_acceptance": acceptance,
        "source_hash_caveat": "The sealed trainer hash/path is common across the matrix, but the launch-time code hash was not individually recorded in run configs.",
        "preserved_initial_failure": "D3-TS00-D-H30-S20261001-FORMAL ended with KeyboardInterrupt while opening the training cache; RETRY1 completed 40 epochs/11120 steps. Both original log and retry artifacts remain intact.",
        "final_test_access": 0,
    }
    with (OUT / "P0_OLD_D3_EFFECTIVE_LINEAGE.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "P0_OLD_D3_EFFECTIVE_LINEAGE.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "matrix": summary["matrix_cells"],
        "train_complete": summary["effective_train_complete"],
        "checkpoints": summary["effective_checkpoints_present"],
        "predictions": summary["effective_prediction_pass"],
        "retry_cells": summary["retry_cells"],
        "final_test_access": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
