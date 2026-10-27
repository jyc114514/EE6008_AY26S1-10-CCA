"""Fit the shared D3 video normalizer from dynamic train grid shards."""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from common import BENCH_ROOT, REPO_ROOT, atomic_write_json, read_json, utc_now


V3 = REPO_ROOT / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"


def main() -> int:
    manifest_path = BENCH_ROOT / "d3/cache/D3_CACHE_MANIFEST.json"
    if not manifest_path.exists():
        raise SystemExit("D3_CACHE_MANIFEST.json is required before normalizer fitting")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "PASS":
        raise SystemExit(f"cache not accepted: {manifest.get('status')}")
    with np.load(V3 / "data/V3_MAIN_INPUTS.npz", allow_pickle=False) as archive:
        split = archive["split_code"]
        episodes = archive["episode_index"]
    per_episode_sum: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(768, dtype=np.float64))
    per_episode_sq: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(768, dtype=np.float64))
    per_episode_n: dict[int, int] = defaultdict(int)
    for shard in manifest["shards"]:
        start, end = int(shard["start_row"]), int(shard["end_row"])
        values = np.load(shard["array"], mmap_mode="r")[:, 0].astype(np.float64, copy=False)
        train_local = np.flatnonzero(split[start:end] == 0)
        for local in train_local:
            episode = int(episodes[start + int(local)])
            row = values[int(local)].mean(axis=(0, 1))
            row_sq = np.square(values[int(local)]).mean(axis=(0, 1))
            per_episode_sum[episode] += row
            per_episode_sq[episode] += row_sq
            per_episode_n[episode] += 1
    if not per_episode_n:
        raise SystemExit("no train rows in D3 cache")
    means = np.asarray([per_episode_sum[key] / per_episode_n[key] for key in sorted(per_episode_n)], dtype=np.float64)
    seconds = np.asarray([per_episode_sq[key] / per_episode_n[key] for key in sorted(per_episode_n)], dtype=np.float64)
    mean = means.mean(axis=0)
    variance = np.maximum(seconds.mean(axis=0) - mean * mean, 0.0)
    safe_std = np.where(np.sqrt(variance) <= 1e-6, 1.0, np.sqrt(variance))
    result = {"status": "LOCKED", "generated_utc": utc_now(), "fit_scope": "dynamic D primary train grid tokens; episode equal, window equal, time/bin equal", "fit_episodes": len(means), "fit_windows": int(sum(per_episode_n.values())), "mean": mean.astype(np.float32).tolist(), "safe_std": safe_std.astype(np.float32).tolist(), "cache_shape": manifest["logical_shape"], "bridge_max_abs": manifest.get("bridge_max_abs"), "constant_channels": np.flatnonzero(np.sqrt(variance) <= 1e-6).astype(int).tolist(), "final_test_access": 0}
    atomic_write_json(BENCH_ROOT / "d3/cache/D3_VIDEO_NORMALIZER.json", result)
    print(json.dumps({"status": result["status"], "fit_episodes": result["fit_episodes"], "fit_windows": result["fit_windows"], "constant_channels": len(result["constant_channels"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
