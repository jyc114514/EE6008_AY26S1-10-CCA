"""Build a primary-train-only H60 target cache from the locked raw Parquets."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
V3_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
OUT = EQ_ROOT / "d3/train_targets_trainonly.npz"
MANIFEST_OUT = EQ_ROOT / "d3/TRAIN_TARGET_CACHE_MANIFEST.json"
HASHES_OUT = EQ_ROOT / "d3/TRAIN_PARQUET_SHA256.csv"
HORIZON = 60


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_digest(ids: np.ndarray) -> str:
    return hashlib.sha256("\n".join(str(item) for item in ids).encode("utf-8")).hexdigest()


def main() -> int:
    source_registry = json.loads((V3_ROOT / "SOURCE_REGISTRY.json").read_text(encoding="utf-8"))
    raw_root = Path(source_registry["raw_root"]).resolve(strict=True)
    normalizers = json.loads((V3_ROOT / "data/V3_NORMALIZERS.json").read_text(encoding="utf-8"))
    archive = np.load(V3_ROOT / "data/V3_MAIN_INPUTS.npz", allow_pickle=False)
    split = archive["split_code"].astype(np.int8, copy=True)
    all_ids = archive["anchor_id"].astype("U64", copy=True)
    all_episodes = archive["episode_index"].astype(np.int64, copy=True)
    all_tasks = archive["original_task_index"].astype(np.int64, copy=True)
    all_frames = archive["anchor_frame"].astype(np.int64, copy=True)
    all_times = archive["anchor_timestamp"].astype(np.float64, copy=True)
    train_indices = np.flatnonzero(split == 0)
    if set(np.unique(split).tolist()) != {0, 1}:
        raise AssertionError("only locked train/dev split codes are expected")
    train_ids = all_ids[train_indices]
    if len(set(train_ids.tolist())) != len(train_indices):
        raise AssertionError("duplicate primary-train anchor IDs")

    with (V3_ROOT / "ANCHOR_MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
        manifest_rows = [row for row in csv.DictReader(handle) if row["split"] == "train" and row["population"] == "primary"]
    row_by_id = {row["anchor_id"]: row for row in manifest_rows}
    if len(manifest_rows) != len(train_indices) or set(row_by_id) != set(train_ids.tolist()):
        raise AssertionError("train-only anchor manifest does not exactly cover locked train IDs")

    rows_by_episode: dict[int, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for train_position, source_index in enumerate(train_indices):
        anchor_id = str(train_ids[train_position])
        row = row_by_id[anchor_id]
        episode = int(all_episodes[source_index])
        if int(row["episode_index"]) != episode or int(row["original_task_index"]) != int(all_tasks[source_index]):
            raise AssertionError("train manifest episode/task join mismatch")
        if int(row["anchor_frame"]) != int(all_frames[source_index]):
            raise AssertionError("train manifest frame join mismatch")
        if abs(float(row["anchor_timestamp"]) - float(all_times[source_index])) > 1e-5:
            raise AssertionError("train manifest timestamp join mismatch")
        rows_by_episode[episode].append((train_position, row))

    n = len(train_indices)
    anchors = np.empty((n, 28), dtype=np.float32)
    targets = np.empty((n, HORIZON, 28), dtype=np.float32)
    timestamps = np.empty(n, dtype=np.float64)
    parquet_hashes: dict[str, tuple[str, int]] = {}
    max_anchor_delta = 0.0
    action_episode_count = 0
    for episode, indexed_rows in sorted(rows_by_episode.items()):
        relative_paths = {row["parquet_path"] for _, row in indexed_rows}
        if len(relative_paths) != 1:
            raise AssertionError(f"train episode {episode} maps to multiple Parquet sources")
        relative_path = next(iter(relative_paths))
        raw_path = (raw_root / relative_path).resolve(strict=True)
        if not raw_path.is_relative_to(raw_root):
            raise AssertionError("train Parquet path escaped the locked raw root")
        table = pq.read_table(raw_path, columns=["timestamp", "action"])
        raw_times = table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        if action.ndim != 2 or action.shape[1] != 28 or len(raw_times) != len(action):
            raise AssertionError(f"train action schema mismatch for episode {episode}")
        if not np.isfinite(action).all() or not np.isfinite(raw_times).all():
            raise AssertionError(f"non-finite train action/timestamp in episode {episode}")
        dt = np.diff(raw_times)
        if not len(dt) or np.any(dt <= 0) or np.max(np.abs(dt - 1.0 / 30.0)) > 1e-5:
            raise AssertionError(f"train raw timeline is not verified 30 Hz for episode {episode}")
        parquet_hashes[relative_path] = (sha256_file(raw_path), raw_path.stat().st_size)
        for train_position, row in indexed_rows:
            frame = int(row["anchor_frame"])
            if frame < 0 or frame + HORIZON >= len(action):
                raise AssertionError(f"train H60 target out of bounds for {row['anchor_id']}")
            if int(row["target_2s_start_frame"]) != frame + 1:
                raise AssertionError(f"train H60 target start seam mismatch for {row['anchor_id']}")
            if int(row["target_2s_end_frame"]) not in {frame + HORIZON, frame + HORIZON + 1}:
                raise AssertionError(f"train H60 target end seam mismatch for {row['anchor_id']}")
            anchor_time = float(all_times[train_indices[train_position]])
            if abs(raw_times[frame] - anchor_time) > 1e-4:
                raise AssertionError(f"train anchor timestamp mismatch for {row['anchor_id']}")
            anchors[train_position] = action[frame].astype(np.float32)
            targets[train_position] = action[frame + 1 : frame + HORIZON + 1].astype(np.float32)
            timestamps[train_position] = raw_times[frame]
            action_episode_count += 1

    npz_anchor = archive["anchor_command_native"][train_indices].astype(np.float32, copy=True)
    max_anchor_delta = float(np.max(np.abs(npz_anchor - anchors)))
    if max_anchor_delta > 1e-6:
        raise AssertionError(f"raw train anchor differs from locked input anchor by {max_anchor_delta}")
    if normalizers["train_windows"] != n:
        raise AssertionError("normalizer train window count mismatch")
    if normalizers["train_episodes"] != len(rows_by_episode):
        raise AssertionError("normalizer train episode count mismatch")
    if hashlib.sha256(np.unique(all_episodes[train_indices]).tobytes()).hexdigest() != normalizers["fit_ids_sha256"]:
        raise AssertionError("V3 target/state normalizer fit-ID hash differs from train episodes")
    if not np.isfinite(anchors).all() or not np.isfinite(targets).all():
        raise AssertionError("train target cache contains non-finite values")

    if OUT.exists() or MANIFEST_OUT.exists() or HASHES_OUT.exists():
        raise FileExistsError("train-only target-cache artifacts already exist; refusing overwrite")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    part = OUT.with_name(f".{OUT.name}.{os.getpid()}.part")
    with part.open("wb") as handle:
        np.savez(
            handle,
            anchor_id=train_ids,
            episode_index=all_episodes[train_indices],
            original_task_index=all_tasks[train_indices],
            anchor_frame=all_frames[train_indices],
            anchor_timestamp=timestamps.astype(np.float32),
            anchor_command_native=anchors,
            targets_native_60=targets,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(part, OUT)
    with HASHES_OUT.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parquet_path_relative_to_locked_raw_root", "sha256", "bytes", "final_test_access"])
        for relative, (digest, size) in sorted(parquet_hashes.items()):
            writer.writerow([relative, digest, size, 0])
    manifest = {
        "status": "PASS",
        "scope": "primary_train_only",
        "target_horizon_steps": HORIZON,
        "train_windows": n,
        "train_episodes": len(rows_by_episode),
        "train_ids_sha256_ordered_newline": id_digest(train_ids),
        "train_episode_fit_ids_sha256": normalizers["fit_ids_sha256"],
        "anchor_manifest_sha256": sha256_file(V3_ROOT / "ANCHOR_MANIFEST.csv"),
        "input_metadata_npz_sha256": sha256_file(V3_ROOT / "data/V3_MAIN_INPUTS.npz"),
        "normalizer_sha256": sha256_file(V3_ROOT / "data/V3_NORMALIZERS.json"),
        "raw_train_parquet_count": len(parquet_hashes),
        "raw_train_parquet_hashes_csv": str(HASHES_OUT.relative_to(EQ_ROOT)),
        "target_cache_sha256": sha256_file(OUT),
        "target_cache_bytes": OUT.stat().st_size,
        "anchor_join_max_abs_error": max_anchor_delta,
        "all_raw_train_rows_30hz": True,
        "h30_is_prefix_of_h60": True,
        "dev_targets_loaded": False,
        "final_test_access": 0,
        "old_root_written": False,
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("status", "scope", "train_windows", "train_episodes", "raw_train_parquet_count", "target_cache_bytes", "anchor_join_max_abs_error", "dev_targets_loaded", "final_test_access")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
