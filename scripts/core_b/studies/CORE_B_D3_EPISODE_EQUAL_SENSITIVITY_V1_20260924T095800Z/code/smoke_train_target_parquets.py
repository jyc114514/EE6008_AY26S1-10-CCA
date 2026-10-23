"""Read a tiny primary-train-only Parquet subset and validate target seams."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
V3_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"


def main() -> int:
    source_registry = json.loads((V3_ROOT / "SOURCE_REGISTRY.json").read_text(encoding="utf-8"))
    raw_root = Path(source_registry["raw_root"]).resolve(strict=True)
    archive = np.load(V3_ROOT / "data/V3_MAIN_INPUTS.npz", allow_pickle=False)
    split = archive["split_code"]
    ids = archive["anchor_id"].astype(str)
    episodes = archive["episode_index"].astype(np.int64)
    frames = archive["anchor_frame"].astype(np.int64)
    times = archive["anchor_timestamp"].astype(np.float64)
    train_indices = np.flatnonzero(split == 0)
    train_position = {str(ids[source]): index for index, source in enumerate(train_indices)}
    train_rows = [
        row
        for row in csv.DictReader((V3_ROOT / "ANCHOR_MANIFEST.csv").open(newline="", encoding="utf-8"))
        if row["split"] == "train" and row["population"] == "primary"
    ]
    by_episode: dict[int, list[dict[str, str]]] = {}
    for row in train_rows:
        by_episode.setdefault(int(row["episode_index"]), []).append(row)
    chosen = sorted(by_episode)[:2]
    if len(chosen) != 2:
        raise AssertionError("expected at least two primary-train episodes")
    checked_windows = 0
    for episode in chosen:
        rows = by_episode[episode]
        paths = {row["parquet_path"] for row in rows}
        if len(paths) != 1:
            raise AssertionError("one train episode maps to multiple Parquet files")
        raw_path = (raw_root / next(iter(paths))).resolve(strict=True)
        if not raw_path.is_relative_to(raw_root):
            raise AssertionError("train Parquet path escaped locked raw root")
        table = pq.read_table(raw_path, columns=["timestamp", "action"])
        raw_times = table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        dt = np.diff(raw_times)
        if action.ndim != 2 or action.shape[1] != 28 or not np.isfinite(action).all():
            raise AssertionError("train action schema/finite check failed")
        if not len(dt) or np.any(dt <= 0) or np.max(np.abs(dt - 1.0 / 30.0)) > 1e-5:
            raise AssertionError("train raw timeline is not 30 Hz")
        for row in rows:
            source_index = train_indices[train_position[row["anchor_id"]]]
            frame = int(row["anchor_frame"])
            if frame != int(frames[source_index]) or int(row["episode_index"]) != int(episodes[source_index]):
                raise AssertionError("train ID/episode/frame metadata join failed")
            if int(row["target_1s_start_frame"]) != frame + 1 or int(row["target_1s_end_frame"]) != frame + 30:
                raise AssertionError("H30 target frame boundary mismatch")
            if int(row["target_2s_start_frame"]) != frame + 1 or int(row["target_2s_end_frame"]) != frame + 60:
                raise AssertionError("H60 target frame boundary mismatch")
            if frame + 60 >= len(action) or abs(raw_times[frame] - float(times[source_index])) > 1e-4:
                raise AssertionError("train anchor/time/H60 bound mismatch")
            anchor_from_npz = archive["anchor_command_native"][source_index]
            anchor_from_raw = action[frame].astype(np.float32)
            if not np.allclose(anchor_from_npz, anchor_from_raw, rtol=0.0, atol=1e-6):
                raise AssertionError("train raw anchor differs from locked model input")
            target60 = action[frame + 1 : frame + 61].astype(np.float32)
            target30 = action[frame + 1 : frame + 31].astype(np.float32)
            if target60.shape != (60, 28) or target30.shape != (30, 28):
                raise AssertionError("train target horizon shape mismatch")
            if not np.array_equal(target60[:30], target30):
                raise AssertionError("H30 is not exactly the same-anchor H60 prefix")
            checked_windows += 1
    print(json.dumps({
        "status": "PASS",
        "train_episodes": len(chosen),
        "train_windows_checked": checked_windows,
        "horizon_steps": [30, 60],
        "anchor_join_max_abs_error": "<=1e-6",
        "dev_targets_loaded": False,
        "final_test_access": 0,
        "output_written": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
