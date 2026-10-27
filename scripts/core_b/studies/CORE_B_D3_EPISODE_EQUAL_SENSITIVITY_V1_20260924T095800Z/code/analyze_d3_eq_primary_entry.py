"""Common-scoring entrypoint with sealed-checksum fallback for historical D3 bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import analyze_d3_eq_primary as analysis


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
BENCH_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
OLD_PRED_ROOT = BENCH_ROOT / "d3/predictions"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_with_legacy_seal(prediction_root: Path, expected_ids: np.ndarray, *, is_new: bool):
    if prediction_root.resolve() != OLD_PRED_ROOT.resolve():
        return analysis.load_predictions_original(prediction_root, expected_ids, is_new=is_new)
    checksums_path = BENCH_ROOT / "CHECKSUMS.sha256"
    if not checksums_path.is_file():
        raise FileNotFoundError("sealed historical D3 checksum manifest is missing")
    sealed = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        pieces = line.split(maxsplit=1)
        if len(pieces) == 2:
            rel = pieces[1].lstrip(" *")
            sealed[rel] = pieces[0]
    selected: dict[tuple[str, int, int], np.ndarray] = {}
    for sidecar in sorted(prediction_root.glob("D3-*.json")):
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        if record.get("status") != "PASS":
            continue
        npz_path = sidecar.with_suffix(".npz")
        rel = npz_path.resolve().relative_to(BENCH_ROOT.resolve()).as_posix()
        expected_hash = record.get("prediction_sha256") or sealed.get(rel)
        if not expected_hash or sha256_file(npz_path) != expected_hash:
            raise AssertionError(f"historical prediction is not bound by its sidecar/sealed checksum: {rel}")
        with np.load(npz_path, allow_pickle=False) as archive:
            ids = archive["anchor_id"].astype("U64")
            pred = archive["predictions_native"].astype(np.float32, copy=True)
            config = str(archive["config"].item())
            seed = int(archive["seed"].item())
            horizon = int(archive["horizon_steps"])
        key = (config, seed, horizon)
        if len(set(ids.tolist())) != len(ids) or set(ids.tolist()) != set(expected_ids.tolist()):
            raise AssertionError(f"historical prediction ID mismatch: {rel}")
        if pred.shape != (len(expected_ids), horizon, 28) or not np.isfinite(pred).all():
            raise AssertionError(f"historical prediction shape/finite mismatch: {rel}")
        id_index = {str(value): i for i, value in enumerate(ids)}
        pred = pred[np.asarray([id_index[str(value)] for value in expected_ids], dtype=np.int64)]
        if key in selected:
            raise AssertionError(f"duplicate historical prediction slot: {key}")
        selected[key] = pred
    if len(selected) != 36:
        raise AssertionError(f"expected 36 sealed historical D3 bundles, found {len(selected)}")
    return selected


def main() -> int:
    analysis.load_predictions_original = analysis.load_predictions
    analysis.load_predictions = load_with_legacy_seal
    return analysis.main()


if __name__ == "__main__":
    raise SystemExit(main())
