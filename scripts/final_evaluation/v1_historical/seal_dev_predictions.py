"""Verify and seal the complete no-target development replay bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT_ROOT = PROJECT / "reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
RUN_ROOT = PROJECT / os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z')
SEEDS_A = (20260825, 20260826, 20260827)
SEEDS_B = (202610131, 202610132, 202610133)
CORE_B_PROGRAMS = ("A0", "AD", "V0", "VD", "S_OPT")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _load_npz(path: Path, expected_keys: set[str]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError(f"prediction archive keys differ at {path}: {archive.files}")
        return {key: archive[key].copy() for key in expected_keys}


def verify_attempt(attempt: str) -> dict[str, Any]:
    attempt_root = RUN_ROOT / "replay" / f"dev_full_{attempt}" / "artifacts"
    run_receipt_path = attempt_root / "DEV_REPLAY_RUN_RECEIPT.json"
    run_receipt = _json(run_receipt_path)
    if run_receipt.get("status") != "PREDICTIONS_COMPLETE_PENDING_SEAL":
        raise ValueError("dev replay run receipt is not complete")
    if run_receipt.get("final_payload_read") is not False or run_receipt.get("new_training_runs") != 0:
        raise ValueError("dev replay has an unexpected final read or training run")
    records = run_receipt.get("prediction_records")
    if not isinstance(records, list) or len(records) != 28:
        raise ValueError("dev prediction inventory must contain 28 complete bundles")
    record_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in records:
        key = (str(row["core"]), str(row["program"]), "REFERENCE" if row.get("seed") is None else str(row["seed"]))
        if key in record_by_key:
            raise ValueError(f"duplicate dev prediction bundle identity: {key}")
        path = Path(row["path"])
        if not path.is_relative_to(attempt_root.resolve()):
            raise ValueError(f"dev prediction path escapes the isolated attempt: {path}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"dev prediction file hash mismatch: {path}")
        record_by_key[key] = row

    expected_a = {("A", program, str(seed)) for program in ("A_DYNAMIC", "A_STATIC") for seed in SEEDS_A}
    expected_b = {("B", program, str(seed)) for program in CORE_B_PROGRAMS for seed in SEEDS_B}
    expected_mix = {("B", program, str(seed)) for program in ("MIX0", "MIXD") for seed in SEEDS_B}
    expected_persistence = {("B", "PERSISTENCE", "REFERENCE")}
    if set(record_by_key) != expected_a | expected_b | expected_mix | expected_persistence:
        raise ValueError("dev prediction bundle identities do not match the frozen 6+15+6+1 matrix")

    core_a_identity = _csv(RUN_ROOT / "replay/core_a/core_a_dev_identity.csv")
    core_a_ids = np.asarray([int(row["episode_id"]) for row in core_a_identity], dtype=np.int64)
    core_a_canonical = [row["canonical_id"] for row in core_a_identity]
    if len(core_a_ids) != 617 or len(set(core_a_ids.tolist())) != 617:
        raise ValueError("Core A dev canonical identity map must contain 617 unique episodes")
    for key in expected_a:
        row = record_by_key[key]
        bundle = _load_npz(Path(row["path"]), {"episode_index", "logits"})
        if not np.array_equal(bundle["episode_index"].astype(np.int64), core_a_ids):
            raise ValueError(f"Core A output ID order differs from canonical dev map: {key}")
        if bundle["logits"].shape != (617, 120) or bundle["logits"].dtype != np.dtype("<f8") or not np.isfinite(bundle["logits"]).all():
            raise ValueError(f"Core A dev logits invalid: {key}")
        row["canonical_ids_sha256"] = hashlib.sha256("\n".join(core_a_canonical).encode()).hexdigest()
        row["payload_sha256"] = hashlib.sha256(np.ascontiguousarray(bundle["logits"], dtype="<f8").tobytes()).hexdigest()

    core_b_prefix_path = RUN_ROOT / "replay/core_b/DEV_PREFIXES_NO_LABELS.npz"
    core_b_prefix_sha = sha256_file(core_b_prefix_path)
    with np.load(core_b_prefix_path, allow_pickle=False) as prefix:
        core_b_ids = prefix["canonical_id"].astype(str)
        anchor = np.asarray(prefix["anchor_native"], dtype=np.float64)
    if len(core_b_ids) != 185 or len(set(core_b_ids.tolist())) != 185 or anchor.shape != (185, 28):
        raise ValueError("Core B canonical dev prefix identity/support is invalid")
    for key in expected_b | expected_mix | expected_persistence:
        row = record_by_key[key]
        bundle = _load_npz(Path(row["path"]), {"canonical_ids", "prediction_native"})
        if not np.array_equal(bundle["canonical_ids"].astype(str), core_b_ids):
            raise ValueError(f"Core B output canonical ID order differs from dev prefix: {key}")
        if bundle["prediction_native"].shape != (185, 30, 28) or bundle["prediction_native"].dtype != np.dtype("<f8") or not np.isfinite(bundle["prediction_native"]).all():
            raise ValueError(f"Core B dev prediction invalid: {key}")
        row["payload_sha256"] = hashlib.sha256(np.ascontiguousarray(bundle["prediction_native"], dtype="<f8").tobytes()).hexdigest()
        row["canonical_ids_sha256"] = hashlib.sha256("\n".join(core_b_ids.tolist()).encode()).hexdigest()
        row["prefix_sha256"] = core_b_prefix_sha

    for seed in SEEDS_B:
        for mix, video in (("MIX0", "V0"), ("MIXD", "VD")):
            with np.load(record_by_key[("B", "S_OPT", str(seed))]["path"], allow_pickle=False) as s, np.load(record_by_key[("B", video, str(seed))]["path"], allow_pickle=False) as v, np.load(record_by_key[("B", mix, str(seed))]["path"], allow_pickle=False) as m:
                expected = 0.5 * np.asarray(s["prediction_native"], dtype=np.float64) + 0.5 * np.asarray(v["prediction_native"], dtype=np.float64)
                if not np.array_equal(expected, m["prediction_native"]):
                    raise ValueError(f"fixed native-float64 mixture differs from sources: {mix}/{seed}")
    with np.load(record_by_key[("B", "PERSISTENCE", "REFERENCE")]["path"], allow_pickle=False) as persistence:
        expected_persistence_native = np.repeat(anchor[:, None, :], 30, axis=1)
        if not np.array_equal(expected_persistence_native, persistence["prediction_native"]):
            raise ValueError("Persistence does not repeat the same observed anchor for 30 rows")
    aliases = _json(attempt_root / "core_b/PERSISTENCE/ALIASES.json")
    if [int(row["seed"]) for row in aliases.get("aliases", [])] != list(SEEDS_B):
        raise ValueError("Persistence seed aliases do not match the fixed three seed slots")

    seal = {
        "schema_version": 1,
        "protocol_id": "EE6008_FINAL_TEST_V1_20261019T041642Z",
        "status": "SEALED_BEFORE_DEV_TARGET_ACCESS",
        "sealed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "split": "dev",
        "prediction_count": len(records),
        "expected_core_a_model_cells": 6,
        "expected_core_b_neural_cells": 15,
        "expected_core_b_derived_mix_cells": 6,
        "expected_persistence_references": 1,
        "prediction_records": records,
        "core_a_identity_map_sha256": sha256_file(RUN_ROOT / "replay/core_a/core_a_dev_identity.csv"),
        "core_b_prefix_sha256": core_b_prefix_sha,
        "core_b_q0_manifest_sha256": "43f0cd52e481cdc88b92b307476e0a6e7e3e6784c8057220501f6ce091e1a2ee",
        "dev_target_archive_opened": False,
        "final_payload_read": False,
    }
    seal_path = attempt_root / "DEV_PREDICTION_SEAL.json"
    seal_path.write_text(json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seal["seal_sha256"] = sha256_file(seal_path)
    return seal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", choices=("attempt_01", "attempt_02", "attempt_03", "attempt_04", "attempt_05", "attempt_06"), required=True)
    args = parser.parse_args()
    seal = verify_attempt(args.attempt)
    print(json.dumps({"status": seal["status"], "prediction_count": seal["prediction_count"], "seal_sha256": seal["seal_sha256"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
