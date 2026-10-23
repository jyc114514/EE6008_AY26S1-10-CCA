#!/usr/bin/env python3
"""Validate, compare and seal the 24 new GPU predictions without opening targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SPLITS = {"train": 17757, "dev": 2322}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(values) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temp = path.with_name("." + path.name + ".part")
    temp.write_text(payload, encoding="utf-8")
    temp.replace(path)


def load_npz(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        missing = set(keys) - set(z.files)
        if missing:
            raise AssertionError(f"{path} missing arrays {sorted(missing)}")
        return {k: z[k].copy() for k in keys}


def csv_write(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--v2-root", type=Path, required=True)
    ap.add_argument("--audit-root", type=Path, required=True)
    args = ap.parse_args()
    project, v2, audit = [p.resolve() for p in (args.project_root, args.v2_root, args.audit_root)]
    contract_path, profile_path = audit / "REPLAY_CONTRACT.json", audit / "PROFILE_LOCK.json"
    contract, profile = [json.loads(p.read_text(encoding="utf-8")) for p in (contract_path, profile_path)]
    if contract.get("locked_utc") is None:
        raise RuntimeError("REPLAY_CONTRACT.json was not locked before GPU inference")
    if profile.get("lock_status") != "locked_before_first_GPU_prediction":
        raise RuntimeError("PROFILE_LOCK.json is not locked before GPU inference")
    locked_uuid = profile.get("new_current_profile", {}).get("gpu_uuid")
    if not locked_uuid:
        raise RuntimeError("profile lock is missing the live GPU UUID")

    input_path = v2 / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_MAIN_INPUTS.npz"
    normalizer_path = v2 / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_NORMALIZERS.json"
    input_sha = digest(input_path)
    refs = {(r["cell"], r["split"]): r for r in contract["references"]}
    with np.load(input_path, allow_pickle=False) as z:
        # These are conditioning and indexing fields only; targets_native is not opened here.
        input_fields = {
            "split_code": z["split_code"].copy(),
            "anchor_id": z["anchor_id"].astype("U96", copy=True),
            "episode_index": z["episode_index"].astype(np.int64, copy=True),
            "original_task_index": z["original_task_index"].astype(np.int64, copy=True),
            "anchor_frame": z["anchor_frame"].astype(np.int64, copy=True),
            "anchor_timestamp": z["anchor_timestamp"].astype(np.float64, copy=True),
            "history_standardized": z["history_standardized"].astype(np.float32, copy=True),
            "history_mask": z["history_mask"].astype(np.float32, copy=True),
            "history_relative_delta_standardized": z["history_relative_delta_standardized"].astype(np.float32, copy=True),
            "anchor_command_native": z["anchor_command_native"].astype(np.float32, copy=True)
        }
    normalizers = json.loads(normalizer_path.read_text(encoding="utf-8"))
    sigma = np.asarray(normalizers["target_sigma"], dtype=np.float32)
    if sigma.shape != (28,) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise AssertionError("train-only target sigma is invalid")
    ids, episodes, split_codes = input_fields["anchor_id"], input_fields["episode_index"], input_fields["split_code"]
    task_index, frames, timestamps = input_fields["original_task_index"], input_fields["anchor_frame"], input_fields["anchor_timestamp"]
    anchor = input_fields["anchor_command_native"]
    rows_by_split = {s: np.flatnonzero(split_codes == (0 if s == "train" else 1)) for s in SPLITS}
    if any(len(rows_by_split[s]) != SPLITS[s] for s in SPLITS):
        raise AssertionError("canonical train/dev counts mismatch")
    expected_rows = list(csv.DictReader((audit / "EXPECTED_CELLS.csv").open(newline="", encoding="utf-8")))
    v2_archive = json.loads((v2 / "ARCHIVE.json").read_text(encoding="utf-8"))
    v2_cells = {c["name"]: c for c in v2_archive["cells"]}
    replay_rows = []
    detail_rows = []
    seals = []
    validation_errors = []

    for expected in expected_rows:
        cell = expected["cell"]
        cell_archive = v2_cells[cell]
        old_run = v2 / cell_archive["run"]
        cell_out = audit / "replay/gpu" / cell
        receipt_path = cell_out / "REPLAY_RECEIPT.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else {}
        child_uuid = receipt.get("runtime", {}).get("physical_gpu_uuid_receipt", {}).get("watcher_uuid")
        if receipt.get("status") != "PASS_GPU_FROZEN_REPLAY":
            validation_errors.append(f"{cell}:GPU_CHILD_STATUS")
        if child_uuid != locked_uuid:
            validation_errors.append(f"{cell}:GPU_UUID_MISMATCH")
        child_device = receipt.get("runtime", {}).get("physical_gpu_uuid_receipt", {})
        if (child_device.get("cuda_visible_devices") != str(child_device.get("physical_gpu"))
                or child_device.get("logical_device") != "cuda:0"
                or receipt.get("runtime", {}).get("device") != "cuda:0"):
            validation_errors.append(f"{cell}:GPU_CVD_LOGICAL_MAPPING")
        if receipt.get("target_arrays_loaded") is not False or receipt.get("final_test_access") != 0:
            validation_errors.append(f"{cell}:SCOPE_RECEIPT")
        if receipt.get("parameters_and_buffers_unchanged") is not True:
            validation_errors.append(f"{cell}:MODEL_STATE_CHANGED")
        out_records = {r["split"]: r for r in receipt.get("outputs", [])}
        for split, n in SPLITS.items():
            old_path = old_run / f"predictions_{split}.npz"
            new_path = cell_out / f"predictions_{split}.npz"
            ref = refs[(cell, split)]
            status = "executed"
            errors = []
            if not old_path.is_file() or not new_path.is_file():
                status = "missing_output"
                errors.append("prediction bundle missing")
                replay_rows.append({"cell": cell, "split": split, "status": status, "executed": 0,
                                    "numeric_accepted": 0, "block_reason": "|".join(errors)})
                validation_errors.append(f"{cell}/{split}:MISSING")
                continue
            old_sha, new_sha = digest(old_path), digest(new_path)
            if old_sha != ref.get("sha256"):
                errors.append("reference prediction hash differs from REPLAY_CONTRACT")
            old = load_npz(old_path, ("anchor_id", "episode_index", "predictions_native"))
            new = load_npz(new_path, ("anchor_id", "episode_index", "original_task_index", "anchor_frame",
                                      "anchor_timestamp", "predictions_native", "horizon_steps", "config", "seed", "split"))
            row_indices = rows_by_split[split]
            pred = new["predictions_native"]
            ref_pred = old["predictions_native"]
            shape_ok = pred.shape == ref_pred.shape == (n, int(expected["horizon"]), 28)
            dtype_ok = pred.dtype == ref_pred.dtype == np.float32
            finite_ok = bool(np.isfinite(pred).all() and np.isfinite(ref_pred).all())
            ids_ok = bool(np.array_equal(new["anchor_id"].astype("U96"), ids[row_indices])
                          and np.array_equal(old["anchor_id"].astype("U96"), ids[row_indices]))
            episodes_ok = bool(np.array_equal(new["episode_index"].astype(np.int64), episodes[row_indices])
                               and np.array_equal(old["episode_index"].astype(np.int64), episodes[row_indices]))
            task_ok = np.array_equal(new["original_task_index"].astype(np.int64), task_index[row_indices])
            frame_ok = np.array_equal(new["anchor_frame"].astype(np.int64), frames[row_indices])
            timestamp_ok = np.array_equal(new["anchor_timestamp"].astype(np.float64), timestamps[row_indices])
            metadata_ok = (int(new["horizon_steps"].item()) == int(expected["horizon"])
                           and str(new["config"].item()) == expected["mode"]
                           and int(new["seed"].item()) == int(expected["seed"])
                           and str(new["split"].item()) == split)
            side = out_records.get(split, {})
            input_sha_record = side.get("conditioning_input_hashes", {})
            input_checks = {
                "ids_sha256": array_sha(ids[row_indices].astype("U96")),
                "episode_ids_sha256": array_sha(episodes[row_indices]),
                "row_indices_sha256": array_sha(row_indices.astype(np.int64)),
                "state_sha256": array_sha(input_fields["history_standardized"][row_indices]),
                "history_mask_sha256": array_sha(input_fields["history_mask"][row_indices]),
                "relative_delta_sha256": array_sha(input_fields["history_relative_delta_standardized"][row_indices]),
                "anchor_command_sha256": array_sha(anchor[row_indices]),
                "original_task_index_sha256": array_sha(task_index[row_indices]),
                "anchor_frame_sha256": array_sha(frames[row_indices]),
                "anchor_timestamp_sha256": array_sha(timestamps[row_indices])
            }
            input_hashes_ok = input_sha_record == input_checks
            side_ok = side.get("sha256") == new_sha and side.get("shape") == list(pred.shape) and side.get("rows") == n
            if not all((shape_ok, dtype_ok, finite_ok, ids_ok, episodes_ok, task_ok, frame_ok, timestamp_ok, metadata_ok, input_hashes_ok, side_ok)):
                errors.append("ID/shape/dtype/finite/mask/input hash gate failed")
            if pred.shape != ref_pred.shape:
                validation_errors.append(f"{cell}/{split}:SHAPE")
                replay_rows.append({"cell": cell, "split": split, "status": "structure_failed", "executed": 1,
                                    "numeric_accepted": 0, "block_reason": "|".join(errors)})
                continue
            diff = np.abs(pred.astype(np.float64) - ref_pred.astype(np.float64))
            tolerance = 1e-6 + 1e-6 * np.abs(ref_pred.astype(np.float64))
            fail = diff > tolerance
            fail_count = int(np.count_nonzero(fail))
            total = int(diff.size)
            bitwise = bool(np.array_equal(pred, ref_pred))
            numeric_ok = fail_count == 0
            structure_ok = all((shape_ok, dtype_ok, finite_ok, ids_ok, episodes_ok, task_ok, frame_ok,
                                timestamp_ok, metadata_ok, input_hashes_ok, side_ok)) and not errors
            status = "accepted_predeclared_V2_engineering_gate" if numeric_ok and structure_ok else (
                "numeric_failed" if structure_ok else "structure_failed")
            if not structure_ok:
                validation_errors.append(f"{cell}/{split}:STRUCTURE")
            if not numeric_ok:
                validation_errors.append(f"{cell}/{split}:ENGINEERING_GATE")
            max_index = np.unravel_index(int(np.argmax(diff)), diff.shape)
            row_i, step_i, action_i = [int(x) for x in max_index]
            canon_i = int(row_indices[row_i])
            old_value, new_value = float(ref_pred[max_index]), float(pred[max_index])
            anchor_value, scale = float(anchor[canon_i, action_i]), float(sigma[action_i])
            old_std, new_std = (old_value - anchor_value) / scale, (new_value - anchor_value) / scale
            details = {
                "cell": cell, "split": split, "status": status,
                "anchor_id": str(ids[canon_i]), "episode_index": int(episodes[canon_i]),
                "original_task_index": int(task_index[canon_i]), "anchor_frame": int(frames[canon_i]),
                "anchor_timestamp": float(timestamps[canon_i]), "future_step_1based": step_i + 1,
                "raw_action_index_0based": action_i, "old_native": old_value, "new_native": new_value,
                "abs_native_delta": float(diff[max_index]), "old_standardized_residual": old_std,
                "new_standardized_residual": new_std, "abs_standardized_delta": abs(old_std - new_std),
                "target_sigma": scale
            }
            detail_rows.append(details)
            replay_rows.append({
                "cell": cell, "mode": expected["mode"], "horizon": expected["horizon"], "seed": expected["seed"],
                "split": split, "status": status, "executed": 1, "numeric_accepted": int(numeric_ok and structure_ok),
                "device": receipt.get("runtime", {}).get("device"),
                "physical_gpu": receipt.get("runtime", {}).get("physical_gpu_uuid_receipt", {}).get("physical_gpu"),
                "logical_device": "cuda:0", "gpu_uuid": child_uuid, "rows": n,
                "expected_shape": f"({n},{expected['horizon']},28)", "dtype": str(pred.dtype),
                "prediction_sha256": new_sha, "reference_sha256": old_sha,
                "max_abs": float(diff.max(initial=0)), "mean_abs": float(diff.mean()),
                "p95_abs": float(np.quantile(diff, 0.95)), "p99_abs": float(np.quantile(diff, 0.99)),
                "bitwise_equal": bitwise, "engineering_gate_fail_count": fail_count,
                "engineering_gate_fail_fraction": fail_count / total,
                "ids_shape_finite_exact": structure_ok, "prediction_metadata_exact": metadata_ok,
                "historical_cpu_zero_tolerance_status": "historical_fail_preserved",
                "block_reason": "|".join(errors)
            })
            seals.append({"cell": cell, "split": split, "path": str(new_path), "bytes": new_path.stat().st_size,
                          "sha256": new_sha, "reference_sha256": old_sha, "rows": n, "shape": list(pred.shape),
                          "dtype": str(pred.dtype), "all_finite": finite_ok, "ids_order_exact": ids_ok and episodes_ok,
                          "conditioning_inputs_exact_hashes": input_hashes_ok, "target_arrays_loaded": False})
    fields = [
        "cell", "mode", "horizon", "seed", "split", "status", "executed", "numeric_accepted", "device",
        "physical_gpu", "logical_device", "gpu_uuid", "rows", "expected_shape", "dtype", "prediction_sha256",
        "reference_sha256", "max_abs", "mean_abs", "p95_abs", "p99_abs", "bitwise_equal",
        "engineering_gate_fail_count", "engineering_gate_fail_fraction", "ids_shape_finite_exact",
        "historical_cpu_zero_tolerance_status", "block_reason"
    ]
    csv_write(audit / "CHECKPOINT_REPLAY.csv", fields, replay_rows)
    detail_fields = list(detail_rows[0]) if detail_rows else ["cell", "split", "status"]
    csv_write(audit / "PREDICTION_DIFF_DETAILS.csv", detail_fields, detail_rows)

    if len(seals) != 24:
        validation_errors.append(f"only {len(seals)}/24 prediction bundles sealed")
    if not validation_errors and len(seals) == 24:
        seal = {
            "status": "PREDICTIONS_SEALED_BEFORE_TARGET_JOIN",
            "sealed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "prediction_count": 24, "cell_count": 12,
            "prediction_hashes": seals,
            "contract_sha256": digest(contract_path), "profile_lock_sha256": digest(profile_path),
            "canonical_input_sha256": input_sha,
            "all_ids_shapes_finite_and_conditioning_input_hashes_exact": True,
            "targets_loaded": False, "final_test_access": 0
        }
        write_json(audit / "PREDICTION_SEAL.json", seal)
    summary = {
        "status": "PASS_PREDICTION_STRUCTURE_GATE" if not validation_errors else "PREDICTION_VALIDATION_WITH_FINDINGS",
        "cells": len(expected_rows), "executed_rows": sum(int(r.get("executed", 0)) for r in replay_rows),
        "numeric_accepted_rows": sum(int(r.get("numeric_accepted", 0)) for r in replay_rows),
        "train_executed": sum(r.get("split") == "train" and int(r.get("executed", 0)) == 1 for r in replay_rows),
        "dev_executed": sum(r.get("split") == "dev" and int(r.get("executed", 0)) == 1 for r in replay_rows),
        "prediction_seal_written": (audit / "PREDICTION_SEAL.json").is_file(),
        "targets_loaded": False, "final_test_access": 0,
        "criterion_origin": contract.get("criterion_origin"),
        "errors": validation_errors
    }
    write_json(audit / "PREDICTION_VALIDATION_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if not validation_errors and len(seals) == 24 else 1


if __name__ == "__main__":
    raise SystemExit(main())

