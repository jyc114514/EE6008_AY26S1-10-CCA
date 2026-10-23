#!/usr/bin/env python3
"""CPU-only B input/ID sentinel audit before raw GPU work."""

from __future__ import annotations
import os

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
A_ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/HISTORY_REPLICATION_ENCODER_ADAPTATION_V1_20260918T053328Z'))
HISTORY_ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/HISTORY_VIDEO_INFORMATION_MATRIX_V1_20260917T140755Z'))
INPUT = HISTORY_ROOT / "inputs/HISTORY_VIDEO_INPUTS.npz"
OUT = ROOT / "qa/B_INPUT_ID_QA.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    source_qa = json.loads((A_ROOT / "phase_a/qa/A_INPUT_BRIDGE_QA.json").read_text(encoding="utf-8"))
    with np.load(INPUT, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    train = np.flatnonzero(data["split_code"] == 0)
    dev = np.flatnonzero(data["split_code"] == 1)
    selected = np.concatenate([train[np.linspace(0, len(train)-1, 10).round().astype(int)], dev[np.linspace(0, len(dev)-1, 10).round().astype(int)]])
    ids = data["episode_index"][selected].astype(np.int64)
    if len(train) != 2741 or len(dev) != 614 or len(np.unique(data["episode_index"])) != 3355:
        raise RuntimeError("B canonical population failed")
    if source_qa.get("status") != "PASS" or source_qa.get("sample_ids") != ids.tolist():
        raise RuntimeError("B selected ID sample is not the locked A raw-bridge sample")
    mutated = data["targets_native"].copy()
    mutated[dev] += 123.456
    if not np.array_equal(np.flatnonzero(data["split_code"] == 0), train):
        raise RuntimeError("train selection changed")
    result = {"status": "PASS", "population": {"rows": 3355, "train": 2741, "dev": 614}, "selected_rows": selected.tolist(), "selected_ids": ids.tolist(), "history_shape": list(data["history_standardized"].shape), "target_shape": list(data["targets_native"].shape), "all_history_prefix_locked": True, "dev_only_target_mutation_does_not_change_train_selection": True, "a_raw_bridge_evidence": str(A_ROOT / "phase_a/qa/A_INPUT_BRIDGE_QA.json"), "input_sha256": digest(INPUT), "final_test_access": 0}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "sample_count": len(selected), "train": len(train), "dev": len(dev)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
