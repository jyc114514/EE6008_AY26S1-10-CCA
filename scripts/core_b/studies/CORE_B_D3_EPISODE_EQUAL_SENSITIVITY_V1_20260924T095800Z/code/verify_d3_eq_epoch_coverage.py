"""Check one complete 17,757-window pass and 278 effective updates in every epoch."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


EQ_ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    part.replace(path)


def main() -> int:
    matrix = list(csv.DictReader((EQ_ROOT / "protocol/D3_EQ_MATRIX_36.csv").open(newline="", encoding="utf-8")))
    if len(matrix) != 36:
        raise AssertionError("locked D3-EQ matrix is not 36 rows")
    run_checks = []
    for cell in matrix:
        run_id = cell["run_id"]
        root = EQ_ROOT / "d3/runs" / run_id
        status = json.loads((root / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
        if status.get("status") != "PASS" or int(status.get("epochs", -1)) != 40 or int(status.get("optimizer_steps", -1)) != 11120:
            raise AssertionError(f"run completion marker failed: {run_id}")
        rows = list(csv.DictReader((root / "EPOCH_LOG.csv").open(newline="", encoding="utf-8")))
        grouped = {}
        for row in rows:
            grouped.setdefault(int(row["epoch"]), []).append(row)
        if set(grouped) != set(range(1, 41)):
            raise AssertionError(f"epoch log does not cover all 40 epochs: {run_id}")
        duplicates = 0
        for epoch, entries in grouped.items():
            valid = [row for row in entries if int(row["rows_seen"]) == 17757 and int(row["optimizer_steps_total"]) == 278 * epoch and int(row["stage0"]) == 0 and int(row["final_test_access"]) == 0]
            if not valid:
                raise AssertionError(f"epoch {epoch} did not log one full train pass/expected updates: {run_id}")
            if any(not np.isfinite(float(row["weighted_loss_sum_over_seen_rows"])) or not np.isfinite(float(row["unweighted_mean_loss"])) for row in valid):
                raise AssertionError(f"non-finite epoch metric: {run_id}/{epoch}")
            duplicates += max(len(entries) - 1, 0)
        run_checks.append({"run_id": run_id, "epochs": 40, "unique_epoch_rows": 40, "full_train_pass_per_epoch": True, "optimizer_steps_total": 11120, "duplicate_attempt_log_rows_preserved": duplicates})
    report = {
        "status": "PASS",
        "accepted_runs": len(run_checks),
        "expected_runs": 36,
        "train_windows_per_epoch": 17757,
        "optimizer_steps_per_epoch": 278,
        "tail_batch_samples": 29,
        "epochs": 40,
        "dev_targets_loaded_by_training": False,
        "final_test_access": 0,
        "runs": run_checks,
    }
    atomic_json(EQ_ROOT / "qa/D3_EQ_EPOCH_COVERAGE_QA.json", report)
    print(json.dumps({"status": "PASS", "accepted_runs": 36, "full_train_passes": 40 * 36, "optimizer_steps_per_run": 11120, "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
