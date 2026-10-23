#!/usr/bin/env python3
"""Fit train-only D temporal-token normalization after cache acceptance."""

from __future__ import annotations
import os

import json
from pathlib import Path

import numpy as np


EXP = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z'))
EPS = 1e-6


def main() -> int:
    with np.load(EXP / "data/V3_MAIN_INPUTS.npz", allow_pickle=False) as inputs:
        split = inputs["split_code"].astype(np.int8); episodes = inputs["episode_index"].astype(np.int64)
    features = np.load(EXP / "cache/V3_TEMPORAL_FEATURES.npy", mmap_mode="r")
    train_rows = np.flatnonzero(split == 0)
    if features.shape[0] < len(split) or not np.isfinite(features[:len(split)]).all(): raise RuntimeError("feature cache primary shape/finite failure")
    features = features[:len(split)]
    means=[]; seconds=[]
    for episode in np.unique(episodes[train_rows]):
        selected = features[train_rows[episodes[train_rows] == episode], 0].astype(np.float64)
        values = selected.reshape(-1, selected.shape[-1])
        means.append(values.mean(axis=0)); seconds.append((values * values).mean(axis=0))
    mean = np.mean(np.asarray(means), axis=0); var = np.maximum(np.mean(np.asarray(seconds), axis=0) - mean * mean, 0.0); safe = np.where(np.sqrt(var) <= EPS, 1.0, np.sqrt(var))
    result = {"status": "LOCKED", "fit_scope": "primary train D temporal tokens only; episode equal and window/time equal", "fit_episodes": int(len(np.unique(episodes[train_rows]))), "fit_windows": int(len(train_rows)), "mean": mean.astype(np.float32), "safe_std": safe.astype(np.float32), "epsilon": EPS, "cache_shape": list(features.shape), "final_test_access": 0}
    (EXP / "data/V3_VIDEO_NORMALIZER.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=lambda x: x.tolist() if isinstance(x,np.ndarray) else (x.item() if isinstance(x,np.generic) else str(x))) + "\n", encoding="utf-8")
    print(json.dumps({"status":"PASS","fit_windows":len(train_rows),"fit_episodes":len(np.unique(episodes[train_rows]))},sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
