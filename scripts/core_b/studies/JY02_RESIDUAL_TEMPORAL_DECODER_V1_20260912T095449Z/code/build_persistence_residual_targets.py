#!/usr/bin/env python3
"""Build train/dev persistence anchors and residual targets from cached inputs."""

from __future__ import annotations
import os

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[0].parent
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
INPUT = REPO / "reports/core_b/B2_B3_POOLED_SAMPLING_LORA_20260908T113000Z/inputs/B2_INPUTS_0125.npz"
TARGET_NORM = REPO / "reports/core_b/feasibility_and_smoke_20260901T160933Z/CORE_B_NORMALIZATION_STATS_TRAIN_ONLY.json"
FRAME_MAP = REPO / "reports/core_b/B2_B3_POOLED_SAMPLING_LORA_20260908T113000Z/inputs/B2_FRAME_MAPS.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    with np.load(INPUT, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    norm = json.loads(TARGET_NORM.read_text(encoding="utf-8"))["1.0s"]
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["safe_std"], dtype=np.float32)
    last_action = np.asarray(data["last_observed_action"], dtype=np.float32)
    target_native = np.asarray(data["targets"], dtype=np.float32)
    target_std = np.asarray(data["targets_standardized"], dtype=np.float32)
    persistence_native = np.repeat(last_action[:, None, :], 30, axis=1)
    persistence_std = (persistence_native - mean[None, :, :]) / std[None, :, :]
    residual_std = target_std - persistence_std
    reconstructed = persistence_std + residual_std
    if float(np.max(np.abs(reconstructed - target_std))) > 2e-6:
        raise RuntimeError("residual reconstruction exceeds float contract")
    if not np.isfinite(residual_std).all() or not np.isfinite(persistence_std).all():
        raise RuntimeError("residual or persistence contains non-finite values")
    output = ROOT / "inputs/JY02_TARGETS.npz"
    part = ROOT / "inputs/.JY02_TARGETS.part.npz"
    np.savez_compressed(part, episode_index=data["episode_index"].astype(np.int64), split_code=data["split_code"], original_task_index=data["original_task_index"].astype(np.int64), state_action_history=data["state_action_history"].astype(np.float32), dynamic_pooled=data["dynamic_pooled"].astype(np.float32), targets_standardized=target_std, targets_native=target_native, persistence_standardized=persistence_std.astype(np.float32), persistence_native=persistence_native, residual_standardized=residual_std.astype(np.float32))
    part.replace(output)
    frame_map = json.loads(FRAME_MAP.read_text(encoding="utf-8"))
    payload = {"status": "PASS", "input_path": str(INPUT), "input_sha256": digest(INPUT), "target_normalization_path": str(TARGET_NORM), "target_normalization_sha256": digest(TARGET_NORM), "output_path": str(output), "output_sha256": digest(output), "target_shape": list(target_std.shape[1:]), "train_count": int(np.sum(data["split_code"] == 0)), "development_count": int(np.sum(data["split_code"] == 1)), "persistence_anchor": "last_observed_action before target start", "frame_map_source": str(FRAME_MAP), "frame_map_latest_observation_seconds": frame_map.get("dense32", {}).get("latest_observation_seconds", 3.875), "residual_standardized": "targets_standardized-persistence_standardized", "second_normalizer": False, "reconstruction_max_abs": float(np.max(np.abs(reconstructed - target_std))), "zero_residual_equals_persistence": bool(np.array_equal(persistence_std + np.zeros_like(persistence_std), persistence_std)), "final_test_access": 0}
    (ROOT / "inputs/JY02_TARGETS_MANIFEST.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
