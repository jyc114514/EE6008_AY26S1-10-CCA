#!/usr/bin/env python3
"""Validate the four CPU-only archive integration fixtures without loading targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--v2-root", type=Path, required=True)
    ap.add_argument("--audit-root", type=Path, required=True)
    args = ap.parse_args()
    project, v2, audit = [p.resolve() for p in (args.project_root, args.v2_root, args.audit_root)]
    inputs = v2 / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_MAIN_INPUTS.npz"
    records = []
    expected = {(mode, horizon, 20261001) for mode in ("TS11-D", "STATE-CTRL") for horizon in (30, 60)}
    with np.load(inputs, allow_pickle=False) as source:
        ids = source["anchor_id"].astype("U96", copy=True)
        episodes = source["episode_index"].astype(np.int64, copy=True)
        split_codes = source["split_code"].copy()
        task_index = source["original_task_index"].astype(np.int64, copy=True)
        frames = source["anchor_frame"].astype(np.int64, copy=True)
        timestamps = source["anchor_timestamp"].astype(np.float64, copy=True)
        states = source["history_standardized"].astype(np.float32, copy=True)
        masks = source["history_mask"].astype(np.float32, copy=True)
        relative = source["history_relative_delta_standardized"].astype(np.float32, copy=True)
        anchors = source["anchor_command_native"].astype(np.float32, copy=True)
    train_rows = np.flatnonzero(split_codes == 0)
    selected = np.asarray(sorted(train_rows.tolist(), key=lambda i: hashlib.sha256(str(ids[i]).encode("utf-8")).hexdigest())[:64], dtype=np.int64)
    for mode, horizon, seed in sorted(expected):
        cell = f"{mode}_H{horizon}_S{seed}"
        root = audit / "replay/cpu_archive_fixtures_v2" / cell
        receipt_path = root / "REPLAY_RECEIPT.json"
        bundle_path = root / "predictions_train_fixture.npz"
        errors = []
        if not receipt_path.is_file() or not bundle_path.is_file():
            errors.append("fixture output or receipt missing")
            receipt, shape, checksum = {}, None, ""
        else:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            checksum = digest(bundle_path)
            with np.load(bundle_path, allow_pickle=False) as output:
                shape = list(output["predictions_native"].shape)
                pred = output["predictions_native"].copy()
                out_ids = output["anchor_id"].astype("U96", copy=True)
                out_episodes = output["episode_index"].astype(np.int64, copy=True)
                out_task = output["original_task_index"].copy()
                out_frames = output["anchor_frame"].copy()
                out_times = output["anchor_timestamp"].copy()
                keys = list(output.files)
            if not np.array_equal(out_ids, ids[selected]): errors.append("anchor IDs/order mismatch")
            if not np.array_equal(out_episodes, episodes[selected]): errors.append("episode IDs/order mismatch")
            if not np.array_equal(out_task, task_index[selected]): errors.append("original_task_index mismatch")
            if not np.array_equal(out_frames, frames[selected]): errors.append("anchor_frame mismatch")
            if not np.array_equal(out_times, timestamps[selected]): errors.append("anchor_timestamp mismatch")
            if pred.shape != (64, horizon, 28): errors.append(f"prediction shape mismatch: {pred.shape}")
            if pred.dtype != np.float32 or not np.isfinite(pred).all(): errors.append("prediction dtype/finite mismatch")
            if any("target" in key.lower() for key in keys): errors.append("prediction bundle unexpectedly contains target fields")
            if receipt.get("target_arrays_loaded") is not False or receipt.get("final_test_access") != 0:
                errors.append("scope receipt mismatch")
            if receipt.get("safe_reload_strict") is not True or receipt.get("parameters_and_buffers_unchanged") is not True:
                errors.append("safe reload/state immutability receipt failed")
            if receipt.get("input_sha256") != digest(inputs): errors.append("V3 input hash mismatch")
            output_record=receipt.get("outputs", [{}])[0]
            if output_record.get("sha256") != checksum: errors.append("prediction receipt hash mismatch")
            expected_input_hashes = {
                "ids_sha256": array_sha(ids[selected].astype("U96")),
                "episode_ids_sha256": array_sha(episodes[selected]),
                "row_indices_sha256": array_sha(selected.astype(np.int64)),
                "state_sha256": array_sha(states[selected]),
                "history_mask_sha256": array_sha(masks[selected]),
                "relative_delta_sha256": array_sha(relative[selected]),
                "anchor_command_sha256": array_sha(anchors[selected]),
                "original_task_index_sha256": array_sha(task_index[selected]),
                "anchor_frame_sha256": array_sha(frames[selected]),
                "anchor_timestamp_sha256": array_sha(timestamps[selected])
            }
            if output_record.get("conditioning_input_hashes") != expected_input_hashes:
                errors.append("conditioning input hash mismatch")
            if receipt.get("run_kind") != "cpu-fixture" or receipt.get("runtime", {}).get("device") != "cpu":
                errors.append("fixture did not run on CPU")
            cwd = Path(receipt.get("cwd", ".")).resolve()
            if cwd == project or not cwd.is_relative_to(audit):
                errors.append("fixture did not run from an isolated non-project working directory")
        record = {"cell": cell, "mode": mode, "horizon": horizon, "seed": seed,
                  "status": "PASS" if not errors else "FAIL", "device": "cpu", "rows": 64,
                  "prediction_shape": shape, "prediction_sha256": checksum,
                  "cwd": receipt.get("cwd", ""), "target_arrays_loaded": receipt.get("target_arrays_loaded"),
                  "optimizer_steps": receipt.get("optimizer_steps"), "encoder_extraction": receipt.get("encoder_extraction"),
                  "errors": "|".join(errors)}
        records.append(record)
    fields = list(records[0]) if records else ["cell", "status"]
    path = audit / "ARCHIVE_CPU_FIXTURE_ACCEPTANCE.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    summary = {"status": "PASS" if records and all(r["status"] == "PASS" for r in records) else "FAIL",
               "fixtures_expected": 4, "fixtures_passed": sum(r["status"] == "PASS" for r in records),
               "scope": "fresh-process CPU cached-feature predictor integration; not GPU replay or original-profile match",
               "fixture_ids": "train IDs sorted by SHA256(anchor_id), first 64",
               "new_training": 0, "optimizer_updates": 0, "encoder_extraction": 0,
               "target_arrays_loaded": False, "final_test_access": 0, "current_working_directory": os.getcwd(),
               "records": records}
    write_json(audit / "ARCHIVE_CPU_FIXTURE_SUMMARY.json", summary)
    print(json.dumps({"status": summary["status"], "fixtures_passed": summary["fixtures_passed"], "fixtures_expected": 4}))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

