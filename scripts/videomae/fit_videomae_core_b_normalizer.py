"""Fit a separate train-only VideoMAE normalizer using the D3 weighting rule."""

from collections import defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np

    project, cache, output = args.project_root.resolve(), args.cache_root.resolve(), args.output.resolve()
    scratch = (project / "tmp/s126mdg33_02").resolve()
    require(cache.is_relative_to(scratch) and output.is_relative_to(scratch), "Personal scratch paths required")
    require(not output.exists(), "Refusing to overwrite existing normalizer")
    manifest_path = cache / "CACHE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["status"] == "PASS" and not manifest["stage0"], "Accepted full cache required")
    require(manifest["logical_shape"] == [20079, 2, 16, 4, 384], "Cache shape mismatch")
    require(manifest["targets_loaded"] is False and manifest["final_test_access"] == 0, "Access policy mismatch")
    inputs = project / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_MAIN_INPUTS.npz"
    require(digest(inputs) == manifest["input_sha256"] == "61e8c9bbac6bb83b7d323d6ac07d0edaf5a745caf4c58407de97fdedc7d9ce78", "Input hash mismatch")
    index = cache / "ANCHOR_INDEX.npz"
    require(digest(index) == manifest["anchor_index_sha256"], "Index hash mismatch")
    with np.load(inputs, allow_pickle=False) as source, np.load(index, allow_pickle=False) as saved:
        require(np.array_equal(saved["cache_index"], np.arange(20079)), "Row order mismatch")
        for key in ("anchor_id", "episode_index", "split_code"):
            require(np.array_equal(source[key], saved[key]), f"Index mismatch: {key}")
        split, episodes = source["split_code"].copy(), source["episode_index"].copy()
    require(int((split == 0).sum()) == 17757 and int((split == 1).sum()) == 2322, "Split mismatch")
    require(not set(episodes[split == 0].tolist()) & set(episodes[split == 1].tolist()), "Episode overlap")
    sums = defaultdict(lambda: np.zeros(384, dtype=np.float64))
    squares = defaultdict(lambda: np.zeros(384, dtype=np.float64))
    counts = defaultdict(int)
    cursor = 0
    for shard in manifest["shards"]:
        start, end = int(shard["start_row"]), int(shard["end_row"])
        require(start == cursor and start < end <= 20079, "Shard coverage mismatch")
        path = Path(shard["array"]).resolve()
        require(path.parent == cache / "shards", "Shard outside cache")
        require(digest(path) == shard["sha256"], "Shard hash mismatch")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        require(values.shape == (end - start, 2, 16, 4, 384) and values.dtype == np.float32, "Shard schema mismatch")
        # Only Dynamic train rows contribute; dev and Static never enter the fit.
        for local in np.flatnonzero(split[start:end] == 0):
            grid = values[int(local), 0].astype(np.float64)
            require(bool(np.isfinite(grid).all()), "Nonfinite train feature")
            episode = int(episodes[start + int(local)])
            sums[episode] += grid.mean(axis=(0, 1))
            squares[episode] += np.square(grid).mean(axis=(0, 1))
            counts[episode] += 1
        del values
        cursor = end
    require(cursor == 20079 and len(counts) == 2712 and sum(counts.values()) == 17757, "Fit population mismatch")
    means = np.asarray([sums[key] / counts[key] for key in sorted(counts)])
    seconds = np.asarray([squares[key] / counts[key] for key in sorted(counts)])
    mean = means.mean(axis=0)
    variance = np.maximum(seconds.mean(axis=0) - mean * mean, 0.0)
    std = np.sqrt(variance)
    safe_std = np.where(std <= 1e-6, 1.0, std)
    require(bool(np.isfinite(mean).all() and np.isfinite(safe_std).all() and (safe_std > 0).all()), "Invalid fitted statistics")
    result = {"status": "LOCKED", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "fit_scope": "dynamic D primary train grid tokens; episode equal, window equal, time/bin equal",
              "fit_episodes": len(counts), "fit_windows": sum(counts.values()),
              "mean": mean.astype(np.float32).tolist(), "safe_std": safe_std.astype(np.float32).tolist(),
              "cache_shape": manifest["logical_shape"], "constant_channels": np.flatnonzero(std <= 1e-6).tolist(),
              "constant_rule": "std <= 1e-6 -> safe_std = 1.0", "shared_by": ["Dynamic", "Static", "train", "dev"],
              "cache_manifest_sha256": digest(manifest_path), "input_sha256": manifest["input_sha256"],
              "anchor_index_sha256": manifest["anchor_index_sha256"], "script_sha256": digest(Path(__file__)),
              "gpu_used": False, "targets_loaded": False, "final_test_access": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": "PASS_TRAIN_ONLY_NORMALIZER", "fit_episodes": len(counts),
                      "fit_windows": sum(counts.values()), "feature_dim": 384,
                      "constant_channels": len(result["constant_channels"]), "output": str(output),
                      "normalizer_sha256": digest(output), "gpu_used": False,
                      "targets_loaded": False, "final_test_access": 0}, indent=2))


if __name__ == "__main__":
    main()
