"""Outcome-blind, hash-fixed primary-dev trajectory IDs for later figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
V3_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"


def main() -> int:
    archive = np.load(V3_ROOT / "data/V3_MAIN_INPUTS.npz", allow_pickle=False)
    split = archive["split_code"]
    dev = np.flatnonzero(split == 1)
    ids = archive["anchor_id"][dev].astype("U64")
    episodes = archive["episode_index"][dev].astype(np.int64)
    tasks = archive["original_task_index"][dev].astype(np.int64)
    frames = archive["anchor_frame"][dev].astype(np.int64)
    rows = []
    for local, anchor_id in enumerate(ids):
        key = hashlib.sha256(f"CORE_B_D3_EQ_TRAJECTORY_V1|{anchor_id}".encode("utf-8")).hexdigest()
        rows.append((key, str(anchor_id), int(episodes[local]), int(tasks[local]), int(frames[local])))
    selected = []
    used_tasks = set()
    for key, anchor_id, episode, task, frame in sorted(rows):
        if task in used_tasks:
            continue
        selected.append({"selection_hash": key, "anchor_id": anchor_id, "episode_index": episode, "original_task_index": task, "anchor_frame": frame})
        used_tasks.add(task)
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise AssertionError("could not select three outcome-blind representative dev anchors")
    report = {
        "status": "LOCKED_BEFORE_PREDICTION_OUTCOMES",
        "selection_rule": "lowest SHA256(CORE_B_D3_EQ_TRAJECTORY_V1|anchor_id), greedily require distinct original_task_index",
        "representatives": selected,
        "dev_rows_seen": int(len(dev)),
        "dev_targets_loaded": False,
        "prediction_outcomes_opened": False,
        "final_test_access": 0,
    }
    out = EQ_ROOT / "protocol/D3_EQ_TRAJECTORY_SELECTION.json"
    if out.exists():
        raise FileExistsError("trajectory IDs already locked; refusing reselection")
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "selected_count": len(selected), "dev_rows": len(dev), "dev_targets_loaded": False, "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
