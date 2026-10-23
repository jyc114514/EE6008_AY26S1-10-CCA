"""CPU-only acceptance gates for D3 dense Stage0, cache, and 36 predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pathlibs import BENCH_ROOT, NEXT_ROOT

EXT_ROOT = NEXT_ROOT / "d3_extension"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_ids() -> np.ndarray:
    with np.load(BENCH_ROOT / "d2/D2_DENSE_INPUTS.npz", allow_pickle=False) as archive:
        ids = archive["dense_id"].astype("U96")
        if len(ids) != 3718 or len(set(ids.tolist())) != 3718:
            raise ValueError("fixed D3 dense panel must have 3718 unique IDs")
        return ids


def validate_cache(stage0: bool) -> dict:
    root = EXT_ROOT / ("stage0" if stage0 else "cache")
    manifest_path = root / ("D3_DENSE_STAGE0_MANIFEST.json" if stage0 else "D3_DENSE_EXTRACTION_RUN_MANIFEST.json")
    manifest = json.loads(manifest_path.read_text())
    target = 16 if stage0 else 3718
    if manifest.get("status") != "PASS" or int(manifest.get("rows_processed", -1)) != target:
        raise ValueError(f"D3 dense extraction manifest incomplete: {manifest_path}")
    expected_ids = set(fixed_ids().tolist())
    covered_ids = []
    covered_rows = 0
    for shard in manifest["shards"]:
        arr_path, meta_path = Path(shard["array"]), Path(shard["meta"])
        if not arr_path.is_relative_to(EXT_ROOT) or not meta_path.is_relative_to(EXT_ROOT):
            raise ValueError("D3 dense cache shard path escaped NEXT_ROOT")
        meta = json.loads(meta_path.read_text())
        values = np.load(arr_path, mmap_mode="r")
        expected_rows = int(shard["end_row"]) - int(shard["start_row"])
        if meta.get("status") != "PASS" or values.shape != (expected_rows, 2, 16, 4, 768) or values.dtype != np.float32:
            raise ValueError(f"D3 dense shard schema failed: {arr_path}")
        if not np.isfinite(values).all() or sha256(arr_path) != shard["sha256"]:
            raise ValueError(f"D3 dense shard finiteness/hash failed: {arr_path}")
        ids = [str(value) for value in meta["dense_ids"]]
        if len(ids) != expected_rows or len(set(ids)) != expected_rows:
            raise ValueError(f"D3 dense shard ID cardinality failed: {arr_path}")
        if any(value not in expected_ids for value in ids):
            raise ValueError(f"unknown dense ID in cache shard: {arr_path}")
        covered_ids.extend(ids)
        covered_rows += expected_rows
    if covered_rows != target or len(set(covered_ids)) != target:
        raise ValueError(f"D3 dense cache coverage={covered_rows}, unique IDs={len(set(covered_ids))}, expected={target}")
    if not stage0 and set(covered_ids) != expected_ids:
        raise ValueError("D3 dense cache does not match the complete fixed 3718-ID panel")
    return {"status": "PASS", "stage0": stage0, "rows": covered_rows, "unique_ids": len(set(covered_ids)), "peak_allocated_mib": manifest.get("measured_peak_allocated_mib"), "peak_reserved_mib": manifest.get("measured_peak_reserved_mib"), "manifest_sha256": sha256(manifest_path), "final_test_access": 0}


def validate_predictions() -> dict:
    expected_ids = set(fixed_ids().tolist())
    rows = {}
    for sidecar in sorted((EXT_ROOT / "predictions").glob("D3-*.json")):
        record = json.loads(sidecar.read_text())
        if record.get("status") != "PASS":
            raise ValueError(f"D3 dense sidecar not PASS: {sidecar}")
        bundle_path = Path(record["prediction_path"])
        if not bundle_path.is_relative_to(EXT_ROOT) or not bundle_path.is_file():
            raise ValueError(f"D3 dense prediction path invalid: {bundle_path}")
        if sha256(bundle_path) != record["prediction_sha256"]:
            raise ValueError(f"D3 dense prediction hash mismatch: {bundle_path}")
        with np.load(bundle_path, allow_pickle=False) as archive:
            ids = archive["dense_id"].astype("U96")
            predictions = archive["predictions_native"]
            horizon = int(archive["horizon_steps"])
            config, seed = str(archive["config"].item()), int(archive["seed"].item())
        if set(ids.tolist()) != expected_ids or predictions.shape != (3718, horizon, 28) or not np.isfinite(predictions).all():
            raise ValueError(f"D3 dense prediction ID/shape/finiteness failed: {bundle_path}")
        key = (config, seed, horizon)
        if key in rows:
            raise ValueError(f"duplicate D3 dense cell: {key}")
        rows[key] = record
    if len(rows) != 36:
        raise ValueError(f"D3 dense accepted predictions={len(rows)}, expected 36")
    return {"status": "PASS", "accepted_bundles": len(rows), "expected_bundles": 36, "dense_ids_per_bundle": 3718, "final_test_access": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0", action="store_true")
    parser.add_argument("--full-cache", action="store_true")
    parser.add_argument("--predictions", action="store_true")
    args = parser.parse_args()
    if sum((args.stage0, args.full_cache, args.predictions)) != 1:
        raise SystemExit("choose exactly one of --stage0, --full-cache, --predictions")
    result = validate_cache(True) if args.stage0 else validate_cache(False) if args.full_cache else validate_predictions()
    path = EXT_ROOT / "qa" / ("D3_DENSE_STAGE0_ACCEPTANCE.json" if args.stage0 else "D3_DENSE_CACHE_ACCEPTANCE.json" if args.full_cache else "D3_DENSE_PREDICTION_ACCEPTANCE.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
