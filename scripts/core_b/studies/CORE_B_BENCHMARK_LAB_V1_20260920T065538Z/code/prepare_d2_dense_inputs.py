"""Build input-only arrays for the locked D2 dense panel."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from common import BENCH_ROOT, PROJECT_ROOT, REPO_ROOT, read_csv, read_json, atomic_write_json


V3 = REPO_ROOT / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
RAW_ROOT = PROJECT_ROOT / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"


def main() -> int:
    panel = read_csv(BENCH_ROOT / "d2/D2_DENSE_PANEL.csv")
    manifest = {row["episode_index"]: row for row in read_csv(V3 / "ANCHOR_MANIFEST.csv")}
    normalizer = read_json(V3 / "data/V3_NORMALIZERS.json")
    n = len(panel)
    history = np.empty((n, 32, 71), dtype=np.float32)
    history_query = np.empty((n, 32), dtype=np.float32)
    history_selected = np.empty((n, 32), dtype=np.float32)
    anchor_native = np.empty((n, 28), dtype=np.float32)
    episode_index = np.empty(n, dtype=np.int64)
    tasks = np.empty(n, dtype=np.int64)
    frames = np.empty(n, dtype=np.int64)
    timestamps = np.empty(n, dtype=np.float32)
    ids = np.empty(n, dtype="U96")
    video_paths = np.empty(n, dtype="U200")
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for i, row in enumerate(panel):
        episode = int(row["episode_index"])
        anchor = int(row["anchor_frame"])
        if episode not in cache:
            meta = manifest[str(episode)]
            table = pq.read_table(RAW_ROOT / Path(meta["parquet_path"]), columns=["frame_index", "timestamp", "action", "observation.arm_joints", "observation.hand_joints", "observation.leg_joints"])
            ts = table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
            action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
            arm = np.asarray(table["observation.arm_joints"].to_pylist(), dtype=np.float64)
            hand = np.asarray(table["observation.hand_joints"].to_pylist(), dtype=np.float64)
            leg = np.asarray(table["observation.leg_joints"].to_pylist(), dtype=np.float64)
            cache[episode] = (ts, action, arm, hand, leg)
        ts, action, arm, hand, leg = cache[episode]
        t0 = float(ts[anchor])
        queries = t0 + np.arange(-31, 1, dtype=np.float64) * 0.125
        selected = np.searchsorted(ts, queries, side="right") - 1
        if np.any(selected < 0) or np.any(selected > anchor) or np.any(ts[anchor + 1 : anchor + 61] <= t0):
            raise ValueError(f"dense input timeline failure episode={episode} anchor={anchor}")
        native = np.concatenate((arm[selected], hand[selected], leg[selected], action[selected]), axis=1)
        history[i] = ((native - np.asarray(normalizer["state_mean"])[None, :]) / np.asarray(normalizer["state_safe_std"])[None, :]).astype(np.float32)
        relative = queries - t0
        history_query[i] = queries.astype(np.float32)
        history_selected[i] = ts[selected].astype(np.float32)
        anchor_native[i] = action[anchor].astype(np.float32)
        episode_index[i] = episode; tasks[i] = int(row["task"]); frames[i] = anchor; timestamps[i] = t0
        ids[i] = f"dense_e{episode:06d}_a{anchor:06d}"
        video_paths[i] = manifest[str(episode)]["video_path"]
    relative = history_query - history_query[:, -1:,]
    relative_std = ((relative - float(normalizer["relative_delta_mean"])) / float(normalizer["relative_delta_safe_std"])).astype(np.float32)
    out = BENCH_ROOT / "d2/D2_DENSE_INPUTS.npz"
    part = out.with_suffix(".part")
    with part.open("wb") as handle:
        np.savez_compressed(handle, dense_id=ids, episode_index=episode_index, original_task_index=tasks, anchor_frame=frames, anchor_timestamp=timestamps, video_path=video_paths, history_standardized=history, history_mask=np.ones((n, 32), dtype=np.float32), history_relative_delta_standardized=relative_std, anchor_command_native=anchor_native)
    part.replace(out)
    atomic_write_json(BENCH_ROOT / "d2/D2_DENSE_INPUT_METADATA.json", {"status": "PASS", "rows": n, "episodes": len(np.unique(episode_index)), "input_features": ["history_standardized", "history_mask", "history_relative_delta_standardized", "anchor_command_native", "video_path"], "future_target_saved": False, "final_test_access": 0})
    print({"status": "PASS", "rows": n, "episodes": len(np.unique(episode_index))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
