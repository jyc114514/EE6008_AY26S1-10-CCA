#!/usr/bin/env python3
"""Seal all Core B V2 predictions before any dev/final target scoring."""
from __future__ import annotations
import os

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT = PROJECT / "reports/final_evaluation/EE6008_FINAL_RECOVERY_V2_20261020T053953Z"
RUN = PROJECT / os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_RECOVERY_V2_20261020T053953Z')
V1_OUT = PROJECT / "reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
V1_MATRIX = V1_OUT / "protocol/FINAL_MODEL_MATRIX.csv"
CURRENT_LOCK = OUT / "core_b/CURRENT_PROFILE_CANDIDATE_LOCK.json"
FINAL_FREEZE = OUT / "core_b/CORE_B_V2_FINAL_FREEZE.json"
FINAL_ACCESS = OUT / "protocol/FINAL_ACCESS_OPENED.json"
DEV_PREFIX = RUN / "replay/core_b/DEV_PREFIXES_CURRENT_PROFILE_V2.npz"
FINAL_PREFIX = RUN / "replay/core_b/FINAL_PREFIXES_CURRENT_PROFILE_V2.npz"
SEEDS = (202610131, 202610132, 202610133)
PROGRAMS = ("A0", "AD", "V0", "VD", "S_OPT")
DERIVED = (("MIX0", "S_OPT", "V0"), ("MIXD", "S_OPT", "VD"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "final_test"), required=True)
    parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument("--expected-final-freeze-sha256")
    parser.add_argument("--expected-final-access-sha256")
    args = parser.parse_args()
    if sha(CURRENT_LOCK) != args.expected_lock_sha256:
        raise ValueError("candidate profile lock hash mismatch")
    lock = json.loads(CURRENT_LOCK.read_text(encoding="utf-8"))
    prefix_path = DEV_PREFIX if args.split == "dev" else FINAL_PREFIX
    if args.split == "final_test":
        if not args.expected_final_freeze_sha256 or not args.expected_final_access_sha256:
            raise ValueError("final prediction seal requires both final freeze and access receipt")
        if sha(FINAL_FREEZE) != args.expected_final_freeze_sha256:
            raise ValueError("final freeze hash mismatch")
        from core_b_predict_v2 import verify_final_access_receipt
        verify_final_access_receipt(FINAL_ACCESS, args.expected_final_access_sha256)
        if lock.get("status") != "LOCKED_BEFORE_CURRENT_PROFILE_DEV_FEATURES":
            raise ValueError("CURRENT_PROFILE_V2 lock is not accepted")
    prefix_sha = sha(prefix_path)
    with np.load(prefix_path, allow_pickle=False) as prefix:
        if any(any(term in key.lower() for term in ("target", "targets", "future", "label", "ground_truth")) for key in prefix.files):
            raise ValueError("label-free prefix contains a forbidden data key")
        ids = prefix["canonical_id"].astype(str)
        anchors = prefix["anchor_id"].astype(str)
        episodes = prefix["episode_id"].astype(np.int64)
        tasks = prefix["task_id"].astype(np.int64)
        anchor_native = np.asarray(prefix["anchor_native"], dtype=np.float64)
    n = len(ids)
    expected_n = 185 if args.split == "dev" else 162
    if n != expected_n or len(set(ids.tolist())) != n or len(set(anchors.tolist())) != n:
        raise ValueError(f"prediction seal expected {expected_n} unique rows")
    attempt_dir = RUN / "replay/core_b" / ("current_profile_dev" if args.split == "dev" else "current_profile_final")
    run_receipt_path = attempt_dir / "PREDICTION_RUN_RECEIPT.json"
    receipt = json.loads(run_receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PREDICTIONS_COMPLETE_PENDING_SEAL" or receipt.get("split") != args.split:
        raise ValueError("prediction runner receipt is incomplete or has the wrong split")
    if receipt.get("candidate_profile_lock_sha256") != args.expected_lock_sha256 or receipt.get("prefix_sha256") != prefix_sha:
        raise ValueError("prediction runner does not bind the current profile/prefix")
    if args.split == "final_test":
        if receipt.get("final_freeze_sha256") != args.expected_final_freeze_sha256 or receipt.get("final_access_receipt_sha256") != args.expected_final_access_sha256:
            raise ValueError("final runner receipt does not bind the final freeze/access")

    records = receipt.get("prediction_records", [])
    by_key = {}
    for rec in records:
        key = (str(rec["program"]), "REFERENCE" if rec.get("seed") is None else str(int(rec["seed"])))
        if key in by_key:
            raise ValueError(f"duplicate prediction record: {key}")
        path = Path(rec["path"])
        if sha(path) != rec["sha256"]:
            raise ValueError(f"prediction file hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"canonical_ids", "prediction_native"}:
                raise ValueError(f"prediction file schema mismatch: {path}")
            got_ids = archive["canonical_ids"].astype(str)
            prediction = np.asarray(archive["prediction_native"], dtype=np.float64)
        if not np.array_equal(got_ids, ids):
            raise ValueError(f"prediction row order differs from fixed prefix: {key}")
        if prediction.shape != (n, 30, 28) or not np.isfinite(prediction).all():
            raise ValueError(f"prediction shape/finite check failed: {key}")
        pred_sha = hashlib.sha256(np.asarray(prediction, dtype="<f8", order="C").tobytes()).hexdigest()
        if pred_sha != rec.get("prediction_native_sha256"):
            raise ValueError(f"prediction value hash mismatch: {key}")
        by_key[key] = rec

    expected_keys = {(program, str(seed)) for program in PROGRAMS for seed in SEEDS}
    expected_keys |= {(mix, str(seed)) for mix, _, _ in DERIVED for seed in SEEDS}
    expected_keys.add(("PERSISTENCE", "REFERENCE"))
    if set(by_key) != expected_keys or len(records) != 22:
        raise ValueError(f"prediction matrix mismatch: missing={sorted(expected_keys-set(by_key))}, extra={sorted(set(by_key)-expected_keys)}")

    # Verify derived mixtures and Persistence against their locked definitions.
    for mix, source_program, video_program in DERIVED:
        for seed in SEEDS:
            def load(program):
                with np.load(by_key[(program, str(seed))]["path"], allow_pickle=False) as archive:
                    return np.asarray(archive["prediction_native"], dtype=np.float64)
            expected = 0.5 * load(source_program) + 0.5 * load(video_program)
            with np.load(by_key[(mix, str(seed))]["path"], allow_pickle=False) as archive:
                actual = np.asarray(archive["prediction_native"], dtype=np.float64)
            if not np.array_equal(actual, expected):
                raise ValueError(f"{mix}/{seed} differs from exact float64 recipe")
    with np.load(by_key[("PERSISTENCE", "REFERENCE")]["path"], allow_pickle=False) as archive:
        persistence = np.asarray(archive["prediction_native"], dtype=np.float64)
    expected_persistence = np.repeat(anchor_native[:, None, :], 30, axis=1)
    if not np.array_equal(persistence, expected_persistence):
        raise ValueError("Persistence prediction differs from the fixed observed anchor")

    final_name = "CURRENT_PROFILE_DEV_PREDICTION_SEAL.json" if args.split == "dev" else "PREDICTION_MANIFEST_AND_SEAL.json"
    out = OUT / "core_b" / final_name
    if out.exists():
        raise FileExistsError(f"refusing to overwrite prediction seal: {out}")
    seal = {
        "schema_version": 1,
        "status": "SEALED_BEFORE_DEV_TARGET_SCORING" if args.split == "dev" else "SEALED_BEFORE_FINAL_TARGET_ACCESS",
        "split": args.split, "created_utc": utc(),
        "candidate_profile_lock_path": str(CURRENT_LOCK), "candidate_profile_lock_sha256": args.expected_lock_sha256,
        "final_freeze_sha256": args.expected_final_freeze_sha256,
        "final_access_receipt_sha256": args.expected_final_access_sha256,
        "prefix_path": str(prefix_path), "prefix_sha256": prefix_sha,
        "canonical_id_order_sha256": hashlib.sha256("\n".join(ids.tolist()).encode()).hexdigest(),
        "anchor_id_order_sha256": hashlib.sha256("\n".join(anchors.tolist()).encode()).hexdigest(),
        "episode_support": int(len(set(episodes.tolist()))),
        "task_support": {str(task): int(np.sum(tasks == task)) for task in sorted(set(tasks.tolist()))},
        "prediction_records": records,
        "neural_cells": 15, "derived_mixture_cells": 6, "persistence_references": 1,
        "prediction_bundles": len(records), "new_training_runs": 0,
        "target_values_materialized": 0,
        "final_payload_read": args.split == "final_test",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": seal["status"], "path": str(out), "sha256": sha(out),
                      "prediction_bundles": len(records), "rows": n, "target_values_materialized": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
