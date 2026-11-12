"""Read-only CPU audit of the explicit VideoMAE cache and train/dev index."""

import argparse
import hashlib
import json
from pathlib import Path


from path_config import scratch_root
def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np

    project, cache = args.project_root.resolve(), args.cache_root.resolve()
    require(cache.is_relative_to(scratch_root(project)), "Cache is outside the configured scratch root")
    manifest = json.loads((cache / "CACHE_MANIFEST.json").read_text())
    require(manifest["status"] == "PASS" and manifest["stage0"] is False, "Full cache required")
    require(manifest["logical_shape"] == [20079, 2, 16, 4, 384], "Logical shape mismatch")
    require(manifest["covered_rows"] == manifest["rows"] == 20079, "Coverage mismatch")
    require(manifest["targets_loaded"] is False and manifest["final_test_access"] == 0, "Access policy mismatch")
    require(manifest["frames"] == 32 and manifest["dtype"] == "float32", "Input/dtype mismatch")
    require(manifest["sampling"] == "floor(linspace(anchor-119,anchor,32))"
            and manifest["static_rule"] == "repeat_anchor_frame", "Sampling mismatch")
    inputs = project / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_MAIN_INPUTS.npz"
    expected_input = "61e8c9bbac6bb83b7d323d6ac07d0edaf5a745caf4c58407de97fdedc7d9ce78"
    require(digest(inputs) == manifest["input_sha256"] == expected_input, "Input identity mismatch")
    index = cache / "ANCHOR_INDEX.npz"
    require(digest(index) == manifest["anchor_index_sha256"], "Index hash mismatch")
    keys = ("anchor_id", "episode_index", "anchor_frame", "video_path", "split_code")
    with np.load(inputs, allow_pickle=False) as source, np.load(index, allow_pickle=False) as saved:
        require(np.array_equal(saved["cache_index"], np.arange(20079)), "Cache row ordering mismatch")
        for key in keys:
            require(np.array_equal(source[key], saved[key]), f"Index mismatch: {key}")
        codes, episodes = saved["split_code"], saved["episode_index"]
        require(set(codes.tolist()) == {0, 1}, "Unexpected split")
        require(len(np.unique(saved["anchor_id"])) == 20079, "Duplicate anchors")
        counts = {}
        for code, name, rows, episode_count in ((0, "train", 17757, 2712), (1, "dev", 2322, 603)):
            mask = codes == code
            require(int(mask.sum()) == rows and len(np.unique(episodes[mask])) == episode_count, "Split count mismatch")
            counts[name] = {"anchors": rows, "episodes": episode_count}
        require(not set(episodes[codes == 0].tolist()) & set(episodes[codes == 1].tolist()), "Episode leakage")
    cursor, total_bytes = 0, 0
    for shard in manifest["shards"]:
        path = Path(shard["array"]).resolve()
        require(path.parent == cache / "shards", "Shard outside explicit cache")
        start, end = shard["start_row"], shard["end_row"]
        require(start == cursor and start < end <= 20079, "Shard gap/overlap")
        require(path.stat().st_size == shard["bytes"] and digest(path) == shard["sha256"], "Shard integrity mismatch")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        require(values.shape == (end - start, 2, 16, 4, 384) and values.dtype == np.float32, "Shard schema mismatch")
        require(bool(np.isfinite(values).all()), "Nonfinite features")
        del values
        cursor, total_bytes = end, total_bytes + shard["bytes"]
    require(cursor == 20079, "Incomplete shards")
    print(json.dumps({"status": "PASS_CACHE_AUDIT_CPU", "splits": counts,
                      "anchor_order_matches": True, "episode_overlap": 0,
                      "shards_verified": len(manifest["shards"]), "bytes_verified": total_bytes,
                      "shape": manifest["logical_shape"], "all_features_finite": True,
                      "gpu_used": False, "targets_loaded": False, "final_test_access": 0,
                      "normalization_fitted": False}, indent=2))


if __name__ == "__main__":
    main()
