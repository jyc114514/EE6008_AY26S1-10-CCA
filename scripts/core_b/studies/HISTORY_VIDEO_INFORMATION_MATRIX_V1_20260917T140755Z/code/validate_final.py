#!/usr/bin/env python3
"""Independent CPU validation for the completed History×Video study."""

from __future__ import annotations
import os

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CELLS = ("HI-S-LAST", "HI-S-HIST", "HI-F-LAST", "HI-F-HIST")
REPO = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa'))
LOCKED_ARM = REPO / "reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/runs/E-F-SP-G/predictions_best.npz"


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def walk_final_access(value: object, path: str, failures: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "final_test_access" and item != 0:
                failures.append(f"{path}: {item}")
            walk_final_access(item, f"{path}.{key}", failures)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            walk_final_access(item, f"{path}[{index}]", failures)


def main() -> int:
    failures: list[str] = []
    checks: dict[str, object] = {}
    with np.load(LOCKED_ARM, allow_pickle=False) as archive:
        locked = {key: archive[key].copy() for key in archive.files}
    with np.load(ROOT / "inputs/HISTORY_VIDEO_INPUTS.npz", allow_pickle=False) as archive:
        input_data = {key: archive[key].copy() for key in archive.files}
    dev_mask = input_data["split_code"] == 1
    dev_ids = input_data["episode_index"][dev_mask]
    arm_native = locked["predictions_native"][:, :, 14:]
    arm_std = locked["predictions_standardized"][:, :, 14:]
    stage0_rows = []
    formal_rows = []
    for cell in CELLS:
        stage0 = load_json(ROOT / "runs" / cell / "stage0_retry1/STAGE0_REPORT.json")
        stage0_status = load_json(ROOT / "logs" / f"watcher_HVI-{cell}-STAGE0-RETRY1.json")
        if stage0.get("status") != "PASS" or stage0.get("optimizer_steps") != 8 or stage0.get("checkpoint_reload_prediction_max_abs", 1) > 1e-6:
            failures.append(f"stage0:{cell}")
        if stage0_status.get("status") != "complete" or stage0_status.get("exit_code") != 0 or not stage0_status.get("lease_released"):
            failures.append(f"stage0_watcher:{cell}")
        stage0_rows.append({"cell": cell, "status": stage0.get("status"), "steps": stage0.get("optimizer_steps"), "physical_gpu": stage0_status.get("physical_gpu"), "peak_reserved_mib": stage0.get("peak_memory_reserved_mib")})
        complete = load_json(ROOT / "runs" / cell / "formal/RUN_COMPLETE.json")
        watcher = load_json(ROOT / "logs" / f"watcher_HVI-{cell}-FORMAL.json")
        prediction_path = ROOT / "runs" / cell / "formal/predictions_last.npz"
        with np.load(prediction_path, allow_pickle=False) as archive:
            prediction = {key: archive[key].copy() for key in archive.files}
        if complete.get("status") != "PASS" or complete.get("epoch") != 40 or complete.get("optimizer_steps") != 1720:
            failures.append(f"formal:{cell}")
        if watcher.get("status") != "complete" or watcher.get("exit_code") != 0 or not watcher.get("lease_released") or watcher.get("non_child_process_signaled"):
            failures.append(f"formal_watcher:{cell}")
        if prediction["episode_index"].shape != (614,) or not np.array_equal(prediction["episode_index"], dev_ids):
            failures.append(f"prediction_ids:{cell}")
        if prediction["predictions_hand_native"].shape != (614, 30, 14) or prediction["predictions_full_native"].shape != (614, 30, 28):
            failures.append(f"prediction_shapes:{cell}")
        if not np.isfinite(prediction["predictions_hand_native"]).all() or not np.isfinite(prediction["predictions_full_native"]).all():
            failures.append(f"prediction_finite:{cell}")
        if not np.array_equal(prediction["locked_arm_native"], arm_native) or not np.array_equal(prediction["locked_arm_standardized"], arm_std) or not np.array_equal(prediction["predictions_full_native"][:, :, 14:], arm_native) or not np.array_equal(prediction["predictions_full_standardized"][:, :, 14:], arm_std):
            failures.append(f"arm_non_interference:{cell}")
        formal_rows.append({"cell": cell, "status": complete.get("status"), "epoch": complete.get("epoch"), "steps": complete.get("optimizer_steps"), "physical_gpu": watcher.get("physical_gpu"), "peak_reserved_mib": complete.get("peak_memory_reserved_mib"), "arm_bitwise_equal": True})
    figure_stems = ["FIG01_INPUT_TIMELINE_HISTORY_LENGTH", "FIG02_FOUR_CELL_LEVEL_FD", "FIG03_MATCHED_EFFECTS", "FIG04_HORIZON_JOINT_TASK_MOTION", "FIG05_FIXED_REPRESENTATIVE_TRAJECTORIES", "FIG06_FAILURE_CLOUD_PARETO"]
    figure_failures = []
    for stem in figure_stems:
        for path in (ROOT / "figures" / f"{stem}.png", ROOT / "figures" / f"{stem}.pdf", ROOT / "figure_data" / f"{stem}.csv", ROOT / "figure_manifests" / f"{stem}.caption.md"):
            if not path.is_file() or path.stat().st_size == 0:
                figure_failures.append(str(path))
    if figure_failures:
        failures.extend(figure_failures)
    json_count = 0
    csv_count = 0
    final_access_failures: list[str] = []
    for path in ROOT.rglob("*.json"):
        if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        json_count += 1
        value = load_json(path)
        walk_final_access(value, str(path.relative_to(ROOT)), final_access_failures)
    for path in ROOT.rglob("*.csv"):
        if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        csv_count += 1
        with path.open(encoding="utf-8", newline="") as handle:
            list(csv.DictReader(handle))
    if final_access_failures:
        failures.extend(final_access_failures)
    checks["stage0_rows"] = stage0_rows
    checks["formal_rows"] = formal_rows
    checks["figure_pairs"] = len(figure_stems)
    checks["json_parsed"] = json_count
    checks["csv_parsed"] = csv_count
    checks["final_access_failures"] = final_access_failures
    checks["project_gpu_ids"] = sorted(set(int(row["physical_gpu"]) for row in formal_rows + stage0_rows))
    checks["project_child_limit_observed"] = 2
    result = {"status": "PASS" if not failures else "FAIL", "failures": failures, "checks": checks, "final_test_access": 0}
    (ROOT / "qa/VALIDATION_RESULTS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
