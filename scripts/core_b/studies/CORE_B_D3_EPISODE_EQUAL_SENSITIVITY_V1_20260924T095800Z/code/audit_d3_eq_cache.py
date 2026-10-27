"""Read-only identity/shape audit for D3-EQ's locked train/dev metadata and cache."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
BENCH_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
V3_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    input_path = V3_ROOT / "data/V3_MAIN_INPUTS.npz"
    archive = np.load(input_path, allow_pickle=False)
    # No targets_native access here; only IDs, split and source metadata.
    keys = ("anchor_id", "episode_index", "original_task_index", "split_code", "anchor_frame", "anchor_timestamp")
    meta = {key: archive[key] for key in keys}
    split = meta["split_code"]
    if set(np.unique(split).tolist()) != {0, 1}:
        raise AssertionError("input archive must contain only train/dev split codes")
    train_idx = np.flatnonzero(split == 0)
    dev_idx = np.flatnonzero(split == 1)
    train_ids = meta["anchor_id"][train_idx].astype(str)
    if len(set(train_ids)) != len(train_ids):
        raise AssertionError("duplicate primary-train anchor IDs")

    anchor_rows = list(csv.DictReader((V3_ROOT / "ANCHOR_MANIFEST.csv").open(newline="", encoding="utf-8")))
    train_rows = [row for row in anchor_rows if row["split"] == "train" and row["population"] == "primary"]
    manifest_by_id = {row["anchor_id"]: row for row in train_rows}
    if len(train_rows) != len(train_idx) or len(manifest_by_id) != len(train_idx):
        raise AssertionError("primary-train manifest row count/uniqueness mismatch")
    if set(manifest_by_id) != set(train_ids):
        raise AssertionError("primary-train manifest IDs differ from V3 train IDs")
    for index in train_idx:
        row = manifest_by_id[str(meta["anchor_id"][index])]
        if int(row["episode_index"]) != int(meta["episode_index"][index]):
            raise AssertionError("train manifest episode mismatch")
        if int(row["original_task_index"]) != int(meta["original_task_index"][index]):
            raise AssertionError("train manifest task mismatch")
        if int(row["anchor_frame"]) != int(meta["anchor_frame"][index]):
            raise AssertionError("train manifest anchor-frame mismatch")
        if abs(float(row["anchor_timestamp"]) - float(meta["anchor_timestamp"][index])) > 1e-5:
            raise AssertionError("train manifest timestamp mismatch")

    train_episode_ids = meta["episode_index"][train_idx].astype(np.int64)
    unique_episodes, counts = np.unique(train_episode_ids, return_counts=True)
    weights = len(train_idx) / (len(unique_episodes) * counts[np.searchsorted(unique_episodes, train_episode_ids)])
    if not np.isclose(weights.mean(), 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("episode weights do not have unit mean")
    normalizers = json.loads((V3_ROOT / "data/V3_NORMALIZERS.json").read_text(encoding="utf-8"))
    fit_episode_hash = hashlib.sha256(np.unique(train_episode_ids).tobytes()).hexdigest()
    if fit_episode_hash != normalizers["fit_ids_sha256"]:
        raise AssertionError("V3 train-only normalizer episode fit-ID hash mismatch")
    if normalizers["train_windows"] != len(train_idx) or normalizers["train_episodes"] != len(unique_episodes):
        raise AssertionError("V3 normalizer train counts differ from locked train IDs")

    cache_path = BENCH_ROOT / "d3/cache/D3_CACHE_MANIFEST.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("status") != "PASS" or cache.get("errors"):
        raise AssertionError("D3 cache manifest is not accepted")
    if cache["expected_rows"] != len(meta["anchor_id"]) or cache["covered_rows"] != len(meta["anchor_id"]):
        raise AssertionError("D3 cache coverage differs from V3 metadata rows")
    if cache["logical_shape"] != [len(meta["anchor_id"]), 2, 16, 4, 768] or cache["dtype"] != "float32":
        raise AssertionError("D3 cache logical shape/dtype mismatch")
    cursor = 0
    backbone_hashes: set[str] = set()
    layouts: set[str] = set()
    preprocess_hashes: set[str] = set()
    cache_root = (BENCH_ROOT / "d3/cache").resolve(strict=True)
    for shard in cache["shards"]:
        start, end = int(shard["start_row"]), int(shard["end_row"])
        if start != cursor or end <= start:
            raise AssertionError("cache shards are not contiguous and non-overlapping")
        array_path = Path(shard["array"]).resolve(strict=True)
        meta_path = Path(shard["meta"]).resolve(strict=True)
        if not array_path.is_relative_to(cache_root) or not meta_path.is_relative_to(cache_root):
            raise AssertionError("cache shard path escaped the immutable cache root")
        if array_path.stat().st_size != int(shard["bytes"]):
            raise AssertionError("cache shard byte size differs from manifest")
        shard_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if shard_meta.get("status") != "PASS" or int(shard_meta["start_row"]) != start or int(shard_meta["end_row"]) != end:
            raise AssertionError("cache shard metadata status/range mismatch")
        if shard_meta["anchor_ids"] != meta["anchor_id"][start:end].astype(str).tolist():
            raise AssertionError("cache shard anchor IDs are not aligned to V3 rows")
        if shard_meta["episode_index"] != meta["episode_index"][start:end].astype(int).tolist():
            raise AssertionError("cache shard episode IDs are not aligned to V3 rows")
        values = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if values.shape != (end - start, 2, 16, 4, 768) or values.dtype != np.float32:
            raise AssertionError("cache shard shape/dtype mismatch")
        if float(shard_meta["bridge_max_abs"]) > 1e-4:
            raise AssertionError("spatial-to-temporal bridge error exceeds 1e-4")
        backbone_hashes.add(str(shard_meta["source_backbone_hash"]))
        layouts.add(str(shard_meta["grid_layout"]))
        preprocess_hashes.add(str(shard_meta["preprocess_hash"]))
        cursor = end
    if cursor != len(meta["anchor_id"]) or len(cache["shards"]) != int(cache["shard_count"]):
        raise AssertionError("cache shard count/coverage mismatch")
    if len(backbone_hashes) != 1 or len(layouts) != 1 or len(preprocess_hashes) != 1:
        raise AssertionError("cache has inconsistent backbone, grid layout or preprocessing identity")

    video_norm = json.loads((BENCH_ROOT / "d3/cache/D3_VIDEO_NORMALIZER.json").read_text(encoding="utf-8"))
    if video_norm.get("status") != "LOCKED":
        raise AssertionError("D3 video normalizer is not locked")
    if video_norm["fit_windows"] != len(train_idx) or video_norm["fit_episodes"] != len(unique_episodes):
        raise AssertionError("D3 video normalizer fit scope differs from primary train")
    if video_norm["cache_shape"] != cache["logical_shape"]:
        raise AssertionError("D3 video normalizer cache shape mismatch")
    if len(video_norm["mean"]) != 768 or len(video_norm["safe_std"]) != 768:
        raise AssertionError("D3 video normalizer dimension mismatch")
    if not np.isfinite(video_norm["mean"]).all() or not np.isfinite(video_norm["safe_std"]).all():
        raise AssertionError("D3 video normalizer contains non-finite values")

    # The source-group unit is one manifest episode/recording. Task IDs are
    # broader action categories and are deliberately not treated as duplicate files.
    dev_rows = [row for row in anchor_rows if row["split"] == "dev" and row["population"] == "primary"]
    dev_by_episode: dict[int, tuple[str, str]] = {}
    for row in dev_rows:
        episode = int(row["episode_index"])
        pair = (row["parquet_path"], row["video_path"])
        if episode in dev_by_episode and dev_by_episode[episode] != pair:
            raise AssertionError("one dev episode maps to multiple source files")
        dev_by_episode[episode] = pair
    if len(dev_by_episode) != len(np.unique(meta["episode_index"][dev_idx])):
        raise AssertionError("dev source episode count differs from identity arrays")

    report = {
        "status": "PASS",
        "train_rows": int(len(train_idx)),
        "train_episodes": int(len(unique_episodes)),
        "train_episode_windows_min_max": [int(counts.min()), int(counts.max())],
        "train_primary_manifest_rows": len(train_rows),
        "train_ids_unique": len(set(train_ids)),
        "v3_train_normalizer_fit_episode_hash_matches": True,
        "video_normalizer_fit_windows": int(video_norm["fit_windows"]),
        "video_normalizer_fit_episodes": int(video_norm["fit_episodes"]),
        "dev_rows": int(len(dev_idx)),
        "dev_source_episode_groups": int(len(dev_by_episode)),
        "dev_task_category_count_not_used_as_source_group": int(len(np.unique(meta["original_task_index"][dev_idx]))),
        "cache_rows": int(cursor),
        "cache_shards_checked": int(len(cache["shards"])),
        "cache_grid_layout": next(iter(layouts)),
        "cache_bridge_max_abs": float(cache["bridge_max_abs"]),
        "cache_backbone_hash_consistent": True,
        "cache_manifest_sha256": sha256_file(cache_path),
        "video_normalizer_sha256": sha256_file(BENCH_ROOT / "d3/cache/D3_VIDEO_NORMALIZER.json"),
        "input_npz_sha256": sha256_file(input_path),
        "anchor_manifest_sha256": sha256_file(V3_ROOT / "ANCHOR_MANIFEST.csv"),
        "dev_targets_loaded": False,
        "final_test_access": 0,
        "old_benchmark_root_write": False,
    }
    out = EQ_ROOT / "qa/D3_EQ_CACHE_ID_AUDIT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
