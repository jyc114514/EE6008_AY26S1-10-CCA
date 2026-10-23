"""Guarded inference-queue entrypoint with the exact U96 ID-hash convention."""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

import run_d3_eq_inference_queue as queue


def corrected_acceptance(manifest, phase: str):
    if phase == "primary":
        project = queue.PROJECT_ROOT
        input_path = project / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_MAIN_INPUTS.npz"
        with np.load(input_path, allow_pickle=False) as data:
            ids = data["anchor_id"][data["split_code"] == 1].astype("U96")
        expected_rows, id_key = 2322, "anchor_id"
    else:
        input_path = queue.PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d2/D2_DENSE_INPUTS.npz"
        with np.load(input_path, allow_pickle=False) as data:
            ids = data["dense_id"].astype("U96")
        expected_rows, id_key = 3718, "dense_id"
    expected_hash = hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest()
    bundles = []
    gpu_hours = 0.0
    for item in manifest["jobs"]:
        record = queue.output_accepted(item, phase)
        if record is None:
            raise AssertionError(f"inference bundle missing/not accepted: {item['run_id']}")
        field = "dev_ids_sha256" if phase == "primary" else "dense_ids_sha256"
        if int(record.get("rows", -1)) != expected_rows or record.get(field) != expected_hash:
            raise AssertionError(f"inference rows/ID hash mismatch: {item['run_id']}")
        gpu_hours += queue.elapsed_lease_hours(queue.Path(item["status_path"]))
        bundles.append({key: item[key] for key in ("run_id", "config", "seed", "horizon_steps")})
        bundles[-1].update({"prediction_sha256": record["prediction_sha256"], "checkpoint_sha256": record["checkpoint_sha256"], "rows": expected_rows, "id_hash": expected_hash})
    if len(bundles) != 36:
        raise AssertionError(f"expected 36 accepted inference bundles, got {len(bundles)}")
    return {
        "status": "PASS", "phase": phase, "accepted_bundles": 36, "expected_bundles": 36,
        "rows_per_bundle": expected_rows, "id_field": id_key, "id_sha256": expected_hash,
        "inference_gpu_hours_actual": gpu_hours, "bundles": bundles,
        "dev_targets_loaded_by_inference": False, "final_test_access": 0,
    }


def main() -> int:
    queue.build_acceptance = corrected_acceptance
    return queue.main()


if __name__ == "__main__":
    raise SystemExit(main())
