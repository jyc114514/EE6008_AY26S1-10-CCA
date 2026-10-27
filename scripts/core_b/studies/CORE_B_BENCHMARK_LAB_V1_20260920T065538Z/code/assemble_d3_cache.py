"""CPU integrity gate for D3 grid shards."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import BENCH_ROOT, REPO_ROOT, atomic_write_json, read_json, utc_now


V3 = REPO_ROOT / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"


def main() -> int:
    with np.load(V3 / "data/V3_MAIN_INPUTS.npz", allow_pickle=False) as archive:
        ids = archive["anchor_id"].astype("U64")
        episodes = archive["episode_index"].astype(np.int64)
    shard_root = BENCH_ROOT / "d3/cache/shards"
    metas = []
    errors = []
    covered = np.zeros(len(ids), dtype=bool)
    redundant = []
    # Prefer a larger full-extraction shard over a nested Stage0 shard with the
    # same start. Stage0 artifacts remain registered as redundant evidence.
    meta_paths = sorted(shard_root.glob("grid_*.json"), key=lambda path: (int(path.stem.split("_")[1]), -int(path.stem.split("_")[2])))
    for meta_path in meta_paths:
        meta = read_json(meta_path)
        start, end = int(meta["start_row"]), int(meta["end_row"])
        array_path = meta_path.with_suffix(".npy")
        if not array_path.exists():
            errors.append(f"missing_array:{array_path.name}")
            continue
        if start < 0 or end > len(ids) or start >= end:
            errors.append(f"range_bounds:{meta_path.name}")
            continue
        if covered[start:end].all():
            redundant.append(meta_path.name)
            continue
        if covered[start:end].any():
            errors.append(f"partial_range_overlap:{meta_path.name}")
            continue
        values = np.load(array_path, mmap_mode="r")
        if tuple(values.shape) != (end - start, 2, 16, 4, 768) or values.dtype != np.float32 or not np.isfinite(values).all():
            errors.append(f"shape_finite_dtype:{array_path.name}:{values.shape}:{values.dtype}")
            continue
        if list(meta.get("anchor_ids", [])) != [str(value) for value in ids[start:end]]:
            errors.append(f"id_order:{meta_path.name}")
        if list(meta.get("episode_index", [])) != [int(value) for value in episodes[start:end]]:
            errors.append(f"episode_order:{meta_path.name}")
        covered[start:end] = True
        metas.append({"meta": str(meta_path), "array": str(array_path), "start_row": start, "end_row": end, "bytes": array_path.stat().st_size, "bridge_max_abs": meta.get("bridge_max_abs")})
    manifest = {"status": "PASS" if not errors and covered.all() else "FAIL", "generated_utc": utc_now(), "expected_rows": len(ids), "covered_rows": int(covered.sum()), "shard_count": len(metas), "redundant_shards": redundant, "errors": errors, "logical_shape": [len(ids), 2, 16, 4, 768], "dtype": "float32", "bridge_max_abs": max((float(item["bridge_max_abs"] or 0.0) for item in metas), default=None), "final_test_access": 0, "shards": metas}
    atomic_write_json(BENCH_ROOT / "d3/cache/D3_CACHE_MANIFEST.json", manifest)
    print(json.dumps({"status": manifest["status"], "expected_rows": len(ids), "covered_rows": int(covered.sum()), "shards": len(metas), "errors": errors[:10]}, ensure_ascii=False))
    return 0 if manifest["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
