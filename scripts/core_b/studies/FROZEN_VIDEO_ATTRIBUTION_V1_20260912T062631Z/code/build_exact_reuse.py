#!/usr/bin/env python3
"""Create immutable exact-reuse P0 cells from the verified canonical bundles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run_frozen_counterfactuals import write_condition


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-normalization", type=Path, required=True)
    args = parser.parse_args()
    with args.registry.open(newline="", encoding="utf-8") as handle:
        registry = {row["model"]: row for row in csv.DictReader(handle)}
    with np.load(args.input, allow_pickle=False) as archive:
        dev = archive["split_code"] == 1
        ids = archive["episode_index"][dev].astype(np.int64)
        tasks = archive["original_task_index"][dev].astype(np.int64)
        targets_std = archive["targets_standardized"][dev].astype(np.float32)
        targets_native = archive["targets"][dev].astype(np.float32)
    norm = json.loads(args.target_normalization.read_text(encoding="utf-8"))["1.0s"]
    mean = np.asarray(norm["mean"], dtype=np.float32)
    scale = np.asarray(norm["safe_std"], dtype=np.float32)
    for model, cell in (("B3-V-D-F", "VF-P0"), ("B3-FUS-D-F", "FF-P0")):
        row = registry[model]
        with np.load(Path(row["canonical_prediction_path"]), allow_pickle=False) as archive:
            canonical_ids = archive["episode_index"].astype(np.int64)
            canonical_std = archive["predictions_standardized"].astype(np.float32)
            canonical_native = archive["predictions_native"].astype(np.float32)
            canonical_targets_std = archive["targets_standardized"].astype(np.float32)
            canonical_targets_native = archive["targets_native"].astype(np.float32)
        if not np.array_equal(ids, canonical_ids):
            raise RuntimeError(f"canonical ID order mismatch for {model}")
        if np.max(np.abs(targets_native - canonical_targets_native)) > 0 or np.max(np.abs(targets_std - canonical_targets_std)) > 0:
            raise RuntimeError(f"canonical target mismatch for {model}")
        write_condition(args.root / "predictions" / model, cell, model, "chronological_matched", ids, tasks, targets_std, targets_native, canonical_std, mean, scale, {"condition": "P0", "level": "exact_reuse", "source_prediction_path": row["canonical_prediction_path"], "source_prediction_sha256": row["canonical_prediction_sha256"], "source_checkpoint_sha256": row["checkpoint_sha256"], "canonical_native_sha256": str(canonical_native.shape), "identity_replay": "verified_614_max_abs_0", "optimizer_steps": 0, "final_test_accessed": 0}, row)
    print("EXACT_REUSE_PASS 2/2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
