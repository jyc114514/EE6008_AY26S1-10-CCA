#!/usr/bin/env python3
"""Repair B2 prediction archives without retraining.

The first B2 runner snapshot inherited a pre-existing writer bug: it wrote the
normalization scale as ``targets_standardized``.  Training metrics were
computed from the correct in-memory batch target, so this repair re-exports
only the prediction archives with canonical target arrays and preserves the
original files under an explicit evidence name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    part = path.with_name(f".{path.name}.{os.getpid()}.part.npz")
    np.savez_compressed(part, **arrays)
    os.replace(part, path)


def atomic_json(path: Path, value: object) -> None:
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(part, path)


def repair_run(run: Path, canonical_ids: np.ndarray, canonical_targets_std: np.ndarray, canonical_targets_native: np.ndarray) -> dict[str, object]:
    if not (run / "RUN_STATUS.json").is_file() or not (run / "RUN_METRICS.json").is_file():
        raise FileNotFoundError(f"run completion artifacts missing: {run}")
    status = json.loads((run / "RUN_STATUS.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete" or status.get("final_test_accessed") != 0:
        raise ValueError(f"run is not a completed train/dev run: {run}")
    repaired: dict[str, object] = {"run_root": str(run), "status": "pass", "archives": []}
    for suffix in ("best", "last"):
        path = run / f"predictions_{suffix}.npz"
        original = run / f"predictions_{suffix}.writer_bug_original.npz"
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        if arrays["episode_index"].shape != (614,) or not np.array_equal(arrays["episode_index"], canonical_ids):
            raise ValueError(f"episode ID contract failed: {path}")
        if arrays["targets_native"].shape != canonical_targets_native.shape or not np.array_equal(arrays["targets_native"], canonical_targets_native):
            raise ValueError(f"native target contract failed: {path}")
        if arrays["predictions_standardized"].shape != (614, 30, 28) or arrays["predictions_native"].shape != (614, 30, 28):
            raise ValueError(f"prediction shape contract failed: {path}")
        if not original.exists():
            shutil.copy2(path, original)
        arrays["targets_standardized"] = canonical_targets_std.copy()
        arrays["predictions_standardized"] = arrays["predictions_standardized"].astype(np.float32, copy=False)
        atomic_npz(path, arrays)
        repaired["archives"].append({
            "suffix": suffix,
            "original_path": str(original),
            "original_sha256": sha256_file(original),
            "repaired_path": str(path),
            "repaired_sha256": sha256_file(path),
            "target_shape": list(arrays["targets_standardized"].shape),
            "prediction_shape": list(arrays["predictions_standardized"].shape),
        })
    repaired.update({
        "writer_bug": "targets_standardized previously pointed to normalization scale (30,28) instead of batch-aligned targets (614,30,28)",
        "repair": "archive-only writer repair; prediction tensors unchanged, no weights or optimizer state changed",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "final_test_accessed": 0,
    })
    atomic_json(run / "WRITER_REPAIR_STATUS.json", repaired)
    atomic_json(run / "RUN_COMPLETE_WRITERFIX.json", {
        "status": "complete_writer_fixed",
        "base_run_complete": str(run / "RUN_COMPLETE.json"),
        "writer_repair_status": str(run / "WRITER_REPAIR_STATUS.json"),
        "prediction_artifacts": repaired["archives"],
        "final_test_accessed": 0,
    })
    return repaired


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--canonical-inputs", type=Path, required=True)
    args = parser.parse_args()
    exp = args.experiment_root.resolve()
    with np.load(args.canonical_inputs.resolve(), allow_pickle=False) as archive:
        split = archive["split_code"]
        dev = split == 1
        ids = archive["episode_index"][dev].copy()
        targets_std = archive["targets_standardized"][dev].copy()
        targets_native = archive["targets"][dev].copy()
    reports = []
    for condition in ("B2-V-0125", "B2-V-0250", "B2-V-0500", "B2-FUS-0125", "B2-FUS-0250", "B2-FUS-0500"):
        reports.append(repair_run(exp / "runs" / condition, ids, targets_std, targets_native))
    print(json.dumps({"status": "pass", "runs": len(reports), "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
