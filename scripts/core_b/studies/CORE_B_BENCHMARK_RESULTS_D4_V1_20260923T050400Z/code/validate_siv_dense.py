"""CPU-only acceptance gate for the optional 54-cell V3_STATE dense extension."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pathlibs import NEXT_ROOT

ROOT = NEXT_ROOT / "v3_state_dense"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    dense_input = NEXT_ROOT.parent / "CORE_B_BENCHMARK_LAB_V1_20260920T065538Z/d2/D2_DENSE_INPUTS.npz"
    with np.load(dense_input, allow_pickle=False) as archive:
        expected_ids = set(archive["dense_id"].astype("U96").tolist())
    rows = {}
    for sidecar in sorted((ROOT / "predictions").glob("*/PREDICTION_EXPORT_COMPLETE.json")):
        record = json.loads(sidecar.read_text())
        if record.get("status") != "PASS" or record.get("state_unchanged") is not True:
            raise ValueError(f"V3_STATE dense sidecar acceptance failed: {sidecar}")
        bundle_path = Path(record["prediction_path"])
        if not bundle_path.is_relative_to(ROOT) or sha256(bundle_path) != record["prediction_sha256"]:
            raise ValueError(f"V3_STATE dense prediction path/hash failed: {bundle_path}")
        with np.load(bundle_path, allow_pickle=False) as bundle:
            ids = bundle["dense_id"].astype("U96")
            prediction = bundle["predictions_native"]
            horizon = int(bundle["horizon_steps"])
            cell = str(bundle["cell"].item())
            seed = int(bundle["seed"].item())
        if set(ids.tolist()) != expected_ids or prediction.shape != (3718, horizon, 28) or not np.isfinite(prediction).all():
            raise ValueError(f"V3_STATE dense ID/shape/finiteness failure: {bundle_path}")
        key = (cell, seed, horizon)
        if key in rows:
            raise ValueError(f"duplicate V3_STATE dense bundle: {key}")
        rows[key] = record
    if len(rows) != 54:
        raise ValueError(f"V3_STATE dense accepted predictions={len(rows)}, expected 54")
    summary = {"status": "PASS", "accepted_bundles": len(rows), "expected_bundles": 54, "dense_ids_per_bundle": 3718, "final_test_access": 0}
    (ROOT / "V3_STATE_DENSE_PREDICTION_ACCEPTANCE.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
