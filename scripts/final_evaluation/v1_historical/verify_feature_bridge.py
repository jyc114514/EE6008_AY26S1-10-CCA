"""Compare fresh train/dev bridge features against accepted caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import core_a_predict as core_a
import core_b_predict as core_b


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
OUT_ROOT = PROJECT / "reports/final_evaluation/EE6008_FINAL_TEST_V1_20261019T041642Z"
RUN_ROOT = PROJECT / os.path.join(os.environ.get("EE6008_OUTPUT_ROOT", os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), "outputs", "ee6008")), 'EE6008_FINAL_TEST_V1_20261019T041642Z')
CORE_A_DYNAMIC_MANIFEST = PROJECT / "features/g1/vjepa2/full-vjepa-b-opencv-20260826/feature_manifest.parquet"
CORE_A_STATIC_MANIFEST = PROJECT / "features/g1/vjepa2/static-first-frame-vjepa-b-v3-tokens-20260828T071500Z/feature_manifest.parquet"
CORE_B_Q0_MANIFEST = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d3/cache/D3_CACHE_MANIFEST.json"
ATOL = 1e-5
RTOL = 1e-5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_paths(path: Path) -> dict[int, Path]:
    # Only episode IDs and feature paths are read; no label-bearing columns.
    frame = pd.read_parquet(path, columns=["episode_index", "feature_path"])
    if frame["episode_index"].duplicated().any():
        raise ValueError(f"duplicate episode IDs in accepted feature manifest: {path}")
    return {int(row.episode_index): Path(str(row.feature_path)) for row in frame.itertuples(index=False)}


def _read_grid_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify(attempt: str, *, stage0: bool) -> dict[str, Any]:
    attempt_root = RUN_ROOT / "features" / attempt
    core_a_root = attempt_root / "core_a"
    core_b_root = attempt_root / "core_b"
    core_a_receipt = json.loads((core_a_root / "EXTRACTION_RECEIPT.json").read_text(encoding="utf-8"))
    if core_a_receipt.get("status") != "FEATURE_EXTRACTION_COMPLETE" or core_a_receipt.get("mode") != "dev":
        raise ValueError("Core A bridge extraction receipt is not a completed dev run")
    source_paths = {
        "dynamic": _feature_paths(CORE_A_DYNAMIC_MANIFEST),
        "static_repeated_first_frame": _feature_paths(CORE_A_STATIC_MANIFEST),
    }
    core_a_reports: dict[str, Any] = {}
    for condition, old_manifest in source_paths.items():
        fresh_manifest = core_a_root / f"feature_manifest_{condition}.csv"
        fresh_rows = core_a.read_feature_manifest(fresh_manifest)
        if len(fresh_rows) != (1 if stage0 else 64):
            raise ValueError(f"Core A {condition} bridge row count mismatch")
        errors: list[float] = []
        for row in fresh_rows:
            if row.episode_index not in old_manifest:
                raise ValueError(f"accepted Core A {condition} cache misses bridge ID {row.episode_index}")
            fresh, _ = core_a.load_feature_tokens(row)
            old, _ = core_a.load_feature_tokens(core_a.FeatureRow(row.episode_index, old_manifest[row.episode_index]))
            if tuple(fresh.shape) != tuple(old.shape):
                raise ValueError(f"Core A {condition} bridge shape mismatch at {row.episode_index}")
            if not np.allclose(fresh.numpy(), old.numpy(), atol=ATOL, rtol=RTOL):
                raise ValueError(f"Core A {condition} fresh feature bridge failed allclose at {row.episode_index}")
            errors.append(float(np.max(np.abs(fresh.float().numpy() - old.float().numpy()))))
        core_a_reports[condition] = {
            "rows": len(fresh_rows),
            "allclose": True,
            "atol": ATOL,
            "rtol": RTOL,
            "max_abs_difference": max(errors, default=0.0),
            "fresh_manifest_sha256": sha256_file(fresh_manifest),
            "fresh_feature_record_sha256": [core_a.sha256_file(row.feature_path) for row in fresh_rows],
        }

    core_b_manifest_path = core_b_root / "grid_manifest.json"
    core_b_manifest = json.loads(core_b_manifest_path.read_text(encoding="utf-8"))
    if core_b_manifest.get("status") != "PASS" or core_b_manifest.get("final_test_access") != 0:
        raise ValueError("Core B bridge feature manifest is not a train/dev-only PASS")
    expected_rows = 1 if stage0 else 64
    if core_b_manifest.get("shape") != [expected_rows, 2, 16, 4, 768]:
        raise ValueError(f"Core B bridge feature shape mismatch: {core_b_manifest.get('shape')}")
    fresh_grid = np.load(Path(core_b_manifest["array_path"]), mmap_mode="r", allow_pickle=False)
    rows = _read_grid_rows(Path(core_b_manifest["row_metadata_path"]))
    if len(rows) != expected_rows or len(fresh_grid) != expected_rows:
        raise ValueError("Core B bridge row metadata and feature array counts differ")
    accepted_manifest = json.loads(CORE_B_Q0_MANIFEST.read_text(encoding="utf-8"))
    sources = core_b.load_accepted_sources()
    q0 = sources.q0_grid_class(accepted_manifest)
    core_b_errors: list[float] = []
    for index, row in enumerate(rows):
        if row["dataset_role"] not in {"train", "dev"}:
            raise ValueError("Core B bridge reader received a non-train/dev row")
        accepted = q0.row(int(row["cache_index"]), row["anchor_id"], row["episode_id"])
        fresh = np.asarray(fresh_grid[index], dtype=np.float32)
        if fresh.shape != accepted.shape or not np.allclose(fresh, accepted, atol=ATOL, rtol=RTOL):
            raise ValueError(f"Core B fresh grid bridge failed allclose at {row['canonical_id']}")
        core_b_errors.append(float(np.max(np.abs(fresh - accepted))))
    core_b_report = {
        "rows": expected_rows,
        "allclose": True,
        "atol": ATOL,
        "rtol": RTOL,
        "max_abs_difference": max(core_b_errors, default=0.0),
        "fresh_array_sha256": sha256_file(Path(core_b_manifest["array_path"])),
        "fresh_manifest_sha256": sha256_file(core_b_manifest_path),
        "accepted_manifest_sha256": sha256_file(CORE_B_Q0_MANIFEST),
        "final_test_access": 0,
    }
    result = {
        "schema_version": 1,
        "status": "PASS",
        "attempt": attempt,
        "stage0_smoke": bool(stage0),
        "verified_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "core_a": core_a_reports,
        "core_b": core_b_report,
        "final_payload_read": False,
    }
    output = RUN_ROOT / "replay" / ("BRIDGE_STAGE0_ACCEPTANCE.json" if stage0 else "FEATURE_BRIDGE_ACCEPTANCE.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.part")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", choices=("stage0", "bridge_attempt_01"), required=True)
    args = parser.parse_args()
    result = verify(args.attempt, stage0=args.attempt == "stage0")
    print(json.dumps({"status": result["status"], "attempt": args.attempt, "core_a": result["core_a"], "core_b": result["core_b"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
