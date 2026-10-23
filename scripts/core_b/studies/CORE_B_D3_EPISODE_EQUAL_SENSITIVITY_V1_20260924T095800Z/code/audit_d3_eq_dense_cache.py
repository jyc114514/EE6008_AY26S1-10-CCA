"""Read-only identity/shape audit for reuse of the locked D3 dense spatial cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
BENCH_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
RESULT_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z"
EXT_ROOT = RESULT_ROOT / "d3_extension"
DENSE_INPUTS = BENCH_ROOT / "d2/D2_DENSE_INPUTS.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest_path = EXT_ROOT / "cache/D3_DENSE_EXTRACTION_RUN_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    acceptance = json.loads((EXT_ROOT / "qa/D3_DENSE_CACHE_ACCEPTANCE.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or int(manifest.get("rows_processed", -1)) != 3718:
        raise AssertionError("dense cache manifest is not complete")
    if acceptance.get("status") != "PASS" or acceptance.get("rows") != 3718 or acceptance.get("unique_ids") != 3718:
        raise AssertionError("dense cache acceptance is not 3,718 unique rows")
    manifest_sha = sha256_file(manifest_path)
    if acceptance.get("manifest_sha256") != manifest_sha:
        raise AssertionError("dense cache acceptance manifest hash mismatch")

    dense = np.load(DENSE_INPUTS, allow_pickle=False)
    dense_ids = dense["dense_id"].astype("U96", copy=True)
    episodes = dense["episode_index"].astype(np.int64, copy=True)
    tasks = dense["original_task_index"].astype(np.int64, copy=True)
    if len(dense_ids) != 3718 or len(set(dense_ids.tolist())) != 3718:
        raise AssertionError("fixed dense panel IDs are not 3,718 unique IDs")
    if len(np.unique(episodes)) != 118 or len(np.unique(tasks)) != 118:
        raise AssertionError("fixed dense panel source-group/task counts differ from 118")

    cursor = 0
    cache_root = (EXT_ROOT / "cache").resolve(strict=True)
    backbone: set[str] = set()
    preprocess: set[str] = set()
    for shard in manifest["shards"]:
        start, end = int(shard["start_row"]), int(shard["end_row"])
        if start != cursor or end <= start:
            raise AssertionError("dense shards are not contiguous and non-overlapping")
        array_path = Path(shard["array"]).resolve(strict=True)
        meta_path = Path(shard["meta"]).resolve(strict=True)
        if not array_path.is_relative_to(cache_root) or not meta_path.is_relative_to(cache_root):
            raise AssertionError("dense shard path escaped accepted cache root")
        if array_path.stat().st_size != int(shard["bytes"]):
            raise AssertionError("dense shard byte count differs from manifest")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "PASS" or meta.get("array_sha256") != shard.get("sha256"):
            raise AssertionError("dense shard status/hash metadata mismatch")
        if int(meta["start_row"]) != start or int(meta["end_row"]) != end or int(meta["row_count"]) != end - start:
            raise AssertionError("dense shard row range mismatch")
        if meta["dense_ids"] != dense_ids[start:end].tolist():
            raise AssertionError("dense feature row IDs are not aligned to locked query IDs")
        if meta["episode_index"] != episodes[start:end].tolist():
            raise AssertionError("dense feature episode IDs are not aligned to locked query IDs")
        values = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if values.shape != (end - start, 2, 16, 4, 768) or values.dtype != np.float32:
            raise AssertionError("dense feature array shape/dtype mismatch")
        if float(meta["bridge_max_abs"]) > 1e-4:
            raise AssertionError("dense spatial-to-temporal bridge error exceeds 1e-4")
        backbone.add(str(meta["source_backbone_sha256"]))
        preprocess.add(str(meta["preprocess_hash"]))
        cursor = end
    if cursor != 3718 or len(manifest["shards"]) != 59 or len(backbone) != 1 or len(preprocess) != 1:
        raise AssertionError("dense cache coverage/shard/backbone/preprocess identity mismatch")

    report = {
        "status": "PASS",
        "queries": cursor,
        "unique_dense_ids": int(len(set(dense_ids.tolist()))),
        "source_episode_groups": int(len(np.unique(episodes))),
        "task_categories": int(len(np.unique(tasks))),
        "shards_checked": int(len(manifest["shards"])),
        "cache_shape_per_row": [2, 16, 4, 768],
        "dtype": "float32",
        "bridge_max_abs": float(manifest["bridge_max_abs"]),
        "cache_manifest_sha256": manifest_sha,
        "dense_inputs_sha256": sha256_file(DENSE_INPUTS),
        "dense_cache_acceptance_sha256": sha256_file(EXT_ROOT / "qa/D3_DENSE_CACHE_ACCEPTANCE.json"),
        "dense_targets_loaded": False,
        "cache_extraction_repeated": False,
        "final_test_access": 0,
        "old_root_written": False,
    }
    out = EQ_ROOT / "qa/D3_EQ_DENSE_CACHE_REUSE_AUDIT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
