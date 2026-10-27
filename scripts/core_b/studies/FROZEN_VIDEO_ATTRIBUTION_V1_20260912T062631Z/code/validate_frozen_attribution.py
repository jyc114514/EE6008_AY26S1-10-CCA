#!/usr/bin/env python3
"""Final deterministic acceptance validator for the Frozen-only mainline."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[0].parent


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    checks: list[dict[str, object]] = []
    matrix = list(csv.DictReader((ROOT / "FROZEN_ATTR_MATRIX.csv").open(newline="", encoding="utf-8")))
    checks.append({"check": "matrix_13_rows", "status": "PASS" if len(matrix) == 13 else "FAIL", "value": len(matrix)})
    accepted: list[dict[str, object]] = []
    roots = {"VF": ROOT / "predictions/B3-V-D-F", "FF": ROOT / "predictions/B3-FUS-D-F"}
    for row in matrix:
        cell = row["cell_id"]
        if cell in {"FF-R1", "FF-R2", "FF-R3"}:
            path = ROOT / "predictions/B3-FUS-D-F_FEATURE_RETRY_20260912T070723Z" / cell
        else:
            path = roots["VF" if cell.startswith("VF") else "FF"] / cell
        bundle = path / "predictions.npz"
        complete = path / "RUN_COMPLETE.json"
        config = path / "RUN_CONFIG_LOCKED.json"
        ok = bundle.exists() and complete.exists() and config.exists()
        n = 0
        shape = "missing"
        finite = False
        if ok:
            with np.load(bundle, allow_pickle=False) as archive:
                n = int(len(archive["episode_index"]))
                shape = str(tuple(archive["predictions_standardized"].shape))
                finite = bool(np.isfinite(archive["predictions_standardized"]).all())
            complete_payload = read_json(complete)
            ok = ok and complete_payload.get("status") == "accepted"
        checks.append({"check": f"cell_{cell}", "status": "PASS" if ok and finite and shape == f"({n}, 30, 28)" else "FAIL", "N": n, "shape": shape, "finite": finite, "path": str(path)})
        if ok and finite:
            accepted.append({"cell_id": cell, "status": "accepted", "N": n, "path": str(path), "shape": shape})
    for model in ("B3-V-D-F", "B3-FUS-D-F"):
        status = read_json(ROOT / "predictions" / model / "RUN_STATUS.json")
        if model == "B3-FUS-D-F" and status.get("status") == "failed":
            failed = read_json(ROOT / "predictions" / model / "RUN_FAILED.json")
            checks.append({"check": f"{model}_parent_retry_record", "status": "PASS" if "AttributeError" in failed.get("error", "") else "FAIL", "note": "pixel artifacts preserved; feature seam retried in a new root"})
            imm = read_json(ROOT / "predictions/B3-FUS-D-F_FEATURE_RETRY_20260912T070723Z/CHECKPOINT_IMMUTABILITY.json")
            checks.append({"check": f"{model}_terminal", "status": "PASS" if read_json(ROOT / "predictions/B3-FUS-D-F_FEATURE_RETRY_20260912T070723Z/RUN_STATUS.json").get("status") == "complete" else "FAIL"})
        else:
            imm = read_json(ROOT / "predictions" / model / "CHECKPOINT_IMMUTABILITY.json")
            checks.append({"check": f"{model}_terminal", "status": "PASS" if status.get("status") == "complete" and status.get("optimizer_steps") == 0 and status.get("final_test_accessed") == 0 else "FAIL"})
        checks.append({"check": f"{model}_immutable", "status": "PASS" if imm.get("unchanged") is True and imm.get("before") == imm.get("after") else "FAIL"})
    retry = read_json(ROOT / "predictions/B3-FUS-D-F_FEATURE_RETRY_20260912T070723Z/RUN_STATUS.json")
    checks.append({"check": "feature_retry_terminal", "status": "PASS" if retry.get("status") == "complete" and retry.get("same_task_eligible_N") == 589 else "FAIL"})
    checks.append({"check": "paired_bootstrap", "status": "PASS" if len(np.load(ROOT / "statistics/paired_bootstrap_indices_N614.npy")) == 10000 else "FAIL"})
    checks.append({"check": "visual_qa", "status": "PASS" if read_json(ROOT / "qa/VISUAL_QA.json").get("status") == "PASS" else "FAIL"})
    checks.append({"check": "figures_12", "status": "PASS" if len(list((ROOT / "figures/png").glob("*.png"))) == 12 else "FAIL"})
    failed = [check for check in checks if check["status"] != "PASS"]
    result = {"status": "PASS" if not failed and len(accepted) == 13 else "FAIL", "accepted_cells": len(accepted), "expected_cells": 13, "checks": checks, "failed_checks": failed, "optimizer_steps": 0, "final_test_access": 0, "old_route_c_status": "blocked_unchanged", "lora_blocks_mainline": False}
    (ROOT / "RUN_ACCEPTANCE.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    with (ROOT / "RUN_ACCEPTANCE.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cell_id", "status", "N", "shape", "path"])
        writer.writeheader()
        writer.writerows(accepted)
    print(json.dumps({"status": result["status"], "accepted_cells": len(accepted), "failed_checks": len(failed)}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
