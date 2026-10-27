#!/usr/bin/env python3
"""Lock the 54-cell state-information x video protocol before new scoring."""

from __future__ import annotations
import os

import csv
import hashlib
import json
from pathlib import Path


REPO = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa'))
V3 = REPO / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
EXP = REPO / "reports/core_b/V3_STATE_INFORMATION_VIDEO_V1_20260920T105403Z"
SEEDS = (20260953, 20260954, 20260955)
HORIZONS = (30, 60)
STATE_LEVELS = ("FULL", "OBS", "ANCHOR")
VIDEO_LEVELS = ("N", "D", "S")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    cells = [f"{state}-{video}" for state in STATE_LEVELS for video in VIDEO_LEVELS]
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for horizon in HORIZONS:
            for cell in cells:
                state, video = cell.split("-")
                rows.append({
                    "run_id": f"SIV-{cell}-H{horizon}-S{seed}",
                    "seed": seed,
                    "horizon_steps": horizon,
                    "horizon_seconds": horizon / 30.0,
                    "cell": cell,
                    "state_level": state,
                    "video_level": video,
                    "pilot_seed": seed == SEEDS[0],
                    "status": "LOCKED",
                    "checkpoint_rule": "epoch40_last",
                    "epochs": 40,
                    "expected_optimizer_steps": 11120,
                    "final_test_access": 0,
                    "encoder_updated": False,
                })
    with (EXP / "MATRIX_54.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(EXP / "MATRIX_54.json", {"status": "LOCKED", "rows": rows, "count": len(rows), "written_before_new_dev_scores": True, "final_test_access": 0})
    write_json(EXP / "SEED_REGISTRY.json", {
        "status": "LOCKED",
        "seeds": list(SEEDS),
        "fresh_seed_rule": "selected before any new experiment dev score; excluded V3 20260950/51/52 and prior Core B seeds",
        "excluded_historical": [20260921, 20260922, 20260923, 20260939, 20260940, 20260941, 20260942, 20260944, 20260950, 20260951, 20260952],
        "final_test_access": 0,
    })
    write_json(EXP / "PROTOCOL_LOCKED.json", {
        "status": "LOCKED",
        "protocol_id": "CORE_B_V3_STATE_INFORMATION_VIDEO_V1",
        "source_v3_root": str(V3),
        "state_levels": list(STATE_LEVELS),
        "video_levels": list(VIDEO_LEVELS),
        "cells": cells,
        "horizons": list(HORIZONS),
        "seeds": list(SEEDS),
        "run_count": 54,
        "pilot_run_count": 18,
        "epochs": 40,
        "optimizer_steps": 11120,
        "prediction_target": "Full28",
        "checkpoint_rule": "last epoch only",
        "encoder_updated": False,
        "new_encoder_forward_for_extraction": False,
        "final_test_access": 0,
    })
    write_json(EXP / "STATE_COLUMN_MAP.json", {
        "status": "VERIFIED",
        "source": "V3 prepare_v3_inputs.py plus raw Parquet schema spot-check",
        "history_native_shape": [32, 71],
        "columns": {
            "observation.arm_joints": {"indices": [0, 14], "width": 14},
            "observation.hand_joints": {"indices": [14, 28], "width": 14},
            "observation.leg_joints": {"indices": [28, 43], "width": 15},
            "action": {"indices": [43, 71], "width": 28},
        },
        "observation_indices": list(range(43)),
        "action_indices": list(range(43, 71)),
        "anchor_command_source": "V3 anchor_command_native = action at anchor frame; independent anchor token retained for FULL/OBS/ANCHOR",
        "not_inferred_from_width_subtraction": True,
        "final_test_access": 0,
    })
    write_json(EXP / "INPUT_MASK_CONTRACT.json", {
        "status": "LOCKED",
        "mask_stage": "after normalization and before state_projection",
        "state_levels": {
            "FULL": {"state_values": "all normalized history columns", "zero_indices": []},
            "OBS": {"state_values": "normalized observation history only", "zero_indices": list(range(43, 71))},
            "ANCHOR": {"state_values": "zero history branch; independent anchor token remains", "zero_indices": list(range(71))},
        },
        "video_levels": {
            "N": "normalized video tensor set to zero",
            "D": "V3 dynamic temporal tokens index 0",
            "S": "V3 repeated-last-frame static temporal tokens index 1",
        },
        "same_mask_train_and_inference": True,
        "no_blocked_column_derived_features": True,
        "state_meta_relative_time_kept_and_shared": True,
        "anchor_token_kept_and_shared": True,
        "token_shape": [16, 768],
        "final_test_access": 0,
    })
    source_files = [
        V3 / "PROTOCOL_LOCKED.json",
        V3 / "SOURCE_REGISTRY.json",
        V3 / "data/V3_NORMALIZERS.json",
        V3 / "data/V3_VIDEO_NORMALIZER.json",
        V3 / "ANCHOR_MANIFEST.csv",
        V3 / "data/V3_MAIN_INPUTS.npz",
        V3 / "cache/V3_TEMPORAL_FEATURES.npy",
        V3 / "code/model.py",
        V3 / "code/prepare_v3_inputs.py",
    ]
    write_json(EXP / "SOURCE_REGISTRY.json", {
        "status": "LOCKED",
        "v3_root": str(V3),
        "v3_protocol_sha256": digest(V3 / "PROTOCOL_LOCKED.json"),
        "v3_source_registry_sha256": digest(V3 / "SOURCE_REGISTRY.json"),
        "files": {str(path.relative_to(REPO)): digest(path) for path in source_files},
        "cache_reused_read_only": True,
        "new_encoder_forward_for_extraction": False,
        "final_test_access": 0,
    })
    write_json(EXP / "RESOURCE_ESTIMATE.json", {
        "status": "PENDING_STAGE0",
        "budget_upper_bound_gpu_hours": 16,
        "additional_disk_gib_upper_bound": 20,
        "run_count": 54,
        "stage0_count": 18,
        "dynamic_visible_gpu_pool": True,
        "max_project_gpu_children": 2,
        "one_project_child_per_physical_gpu": True,
        "co_location_allowed": True,
        "utilization_is_sorting_only": True,
        "final_test_access": 0,
    })
    (EXP / "OUTCOME_EXPOSURE.md").write_text(
        "# Outcome exposure\n\n"
        "All 54 rows were written to `MATRIX_54.csv` before any new study development score was read. "
        "No cell, seed, horizon, checkpoint or metric was removed or changed based on a result. "
        "Only engineering validity and the pre-locked resource budget may control execution.\n\n"
        "`FINAL_TEST_ACCESS=0`; `ENCODER_UPDATED=false`; old roots are read-only.\n",
        encoding="utf-8",
    )
    (EXP / "CLAIM_LEDGER.md").write_text(
        "# Claim ledger\n\n"
        "- Video gain after removing state information is conditional on that reduced-information input; it cannot be presented as improvement on the complete-state task.\n"
        "- ANCHOR retains the independent last-command token and is not strict video-only.\n"
        "- FULL/OBS/ANCHOR differences can reflect optimization or regularization, not only physical information redundancy.\n"
        "- All results are development-only and exploratory; no final/generalization claim is authorized.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "matrix_rows": len(rows), "seeds": SEEDS, "cells": cells, "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
