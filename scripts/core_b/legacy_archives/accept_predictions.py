#!/usr/bin/env python3
"""CPU-only acceptance of the six recovery prediction exports."""

from __future__ import annotations
import os

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
RECOVERY = PROJECT / "research_archive/CORE_B_B_INFERENCE_RECOVERY_20260919T061409Z"
HISTORY_ROOT = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/HISTORY_VIDEO_INFORMATION_MATRIX_V1_20260917T140755Z"
INPUT = HISTORY_ROOT / "inputs/HISTORY_VIDEO_INPUTS.npz"
LOCKED_ARM = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/POSTFREEZE_GATED_RESIDUAL_FACTORIAL_V1_20260913T061201Z/runs/E-F-SP-G/predictions_best.npz"
SOURCE_ROOT = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/ENCODER_ADAPTATION_INDEPENDENT_B_V1_20260918T071854Z"
CELLS = ("FT-D-FRZ", "FT-S-FRZ", "FT-D-LORA4", "FT-S-LORA4", "FT-D-DENSE4", "FT-S-DENSE4")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main() -> int:
    with np.load(INPUT, allow_pickle=False) as archive:
        ids = archive["episode_index"].copy().astype(np.int64)
        split = archive["split_code"].copy().astype(np.int64)
        tasks = archive["original_task_index"].copy().astype(np.int64)
    dev_rows = np.flatnonzero(split == 1)
    train_rows = np.flatnonzero(split == 0)
    expected_ids = ids[dev_rows]
    with np.load(LOCKED_ARM, allow_pickle=False) as arm:
        arm_ids = arm["episode_index"].copy().astype(np.int64)
        arm_native = arm["predictions_native"][:, :, 14:].copy().astype(np.float32)
        arm_std = arm["predictions_standardized"][:, :, 14:].copy().astype(np.float32)
    rows: list[dict[str, object]] = []
    accepted_arrays: dict[str, dict[str, np.ndarray]] = {}
    all_pass = True
    for cell in CELLS:
        pred_dir = RECOVERY / f"predictions/{cell}"
        shard = pred_dir / "shard_000.npz"
        meta = json.loads((pred_dir / "DEV_PREDICTION_ACCEPTANCE.json").read_text(encoding="utf-8"))
        with np.load(shard, allow_pickle=False) as values:
            keys = set(values.files)
            actual_ids = values["episode_index"].copy().astype(np.int64)
            hand_native = values["predictions_hand_native"].copy().astype(np.float32)
            hand_std = values["predictions_hand_standardized"].copy().astype(np.float32)
            full_native = values["predictions_full_native"].copy().astype(np.float32)
            full_std = values["predictions_full_standardized"].copy().astype(np.float32)
            locked_native = values["locked_arm_native"].copy().astype(np.float32)
            locked_std = values["locked_arm_standardized"].copy().astype(np.float32)
        expected_keys = {"episode_index", "original_task_index", "predictions_hand_native", "predictions_hand_standardized", "predictions_full_native", "predictions_full_standardized", "locked_arm_native", "locked_arm_standardized"}
        schema_pass = keys == expected_keys
        id_pass = np.array_equal(actual_ids, expected_ids) and len(np.unique(actual_ids)) == 614
        shape_pass = hand_native.shape == (614, 30, 14) and hand_std.shape == (614, 30, 14) and full_native.shape == (614, 30, 28) and full_std.shape == (614, 30, 28)
        finite_pass = all(np.isfinite(value).all() for value in (hand_native, hand_std, full_native, full_std, locked_native, locked_std))
        arm_pass = np.array_equal(locked_native, arm_native) and np.array_equal(locked_std, arm_std) and np.array_equal(full_native[:, :, 14:], arm_native) and np.array_equal(full_std[:, :, 14:], arm_std)
        meta_pass = meta.get("status") == "PASS" and meta.get("epoch") == 10 and meta.get("optimizer_steps") == 430 and meta.get("seed") == 20260944 and meta.get("final_test_access") == 0 and meta.get("parameter_and_buffer_hash_unchanged") is True
        passed = bool(schema_pass and id_pass and shape_pass and finite_pass and arm_pass and meta_pass)
        all_pass = all_pass and passed
        rows.append({"cell": cell, "status": "PASS" if passed else "FAIL", "schema_pass": schema_pass, "id_pass": id_pass, "N_episode": int(len(actual_ids)), "unique_ids": int(len(np.unique(actual_ids))), "shape_pass": shape_pass, "finite_pass": finite_pass, "arm_bitwise_pass": arm_pass, "metadata_pass": meta_pass, "checkpoint_sha256": meta.get("checkpoint_sha256"), "prediction_sha256": digest(shard), "dev_ids_sha256": array_digest(actual_ids), "physical_gpu": meta.get("physical_gpu"), "final_test_access": 0})
        if passed:
            accepted_arrays[cell] = {"hand_native": hand_native, "hand_std": hand_std, "full_native": full_native, "full_std": full_std}
    # A deliberately mutated train-first ID vector must be rejected by the
    # same identity contract; this is a CPU-only negative fixture.
    train_first = ids[train_rows[:614]]
    negative_fixture = {"train_first_ids_sha256": array_digest(train_first), "matches_canonical_dev": bool(np.array_equal(train_first, expected_ids)), "expected_rejection": True, "rejected": bool(not np.array_equal(train_first, expected_ids)), "final_test_access": 0}
    (RECOVERY / "data/DEV_ID_NEGATIVE_FIXTURE.json").write_text(json.dumps(negative_fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (RECOVERY / "PREDICTION_ACCEPTANCE.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {"status": "PASS" if all_pass and len(accepted_arrays) == 6 and negative_fixture["rejected"] else "FAIL", "accepted": f"{len(accepted_arrays)}/6", "rows": rows, "canonical_dev_rows": 614, "canonical_dev_ids_sha256": array_digest(expected_ids), "train_rows": 2741, "negative_fixture": negative_fixture, "final_test_access": 0}
    (RECOVERY / "PREDICTION_ACCEPTANCE.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "accepted": result["accepted"], "final_test_access": 0}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
