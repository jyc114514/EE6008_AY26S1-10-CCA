#!/usr/bin/env python3
"""Lock the observation-delay protocol before new development scoring."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from delay_contract import DELAYS, EXP, REPO, SIV, V3, load_main


SEEDS = (20260956, 20260957, 20260958)
CELLS = ("CLEAN-N", "CLEAN-D", "CLEAN-S", "ROBUST-N", "ROBUST-D", "ROBUST-S")
HORIZON = 60


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_default) + "\n", encoding="utf-8")


def _default(value: object) -> object:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return str(value)


def main() -> int:
    data, normalizer = load_main()
    dev = np.flatnonzero(data["split_code"] == 1)
    train = np.flatnonzero(data["split_code"] == 0)
    matrix_rows = []
    eval_rows = []
    for seed in SEEDS:
        for cell in CELLS:
            run_id = f"ODR-{cell}-H60-S{seed}"
            robust = cell.startswith("ROBUST")
            video = cell.split("-")[1]
            matrix_rows.append({"run_id": run_id, "seed": seed, "cell": cell, "train_delay_policy": "uniform_hash_four_delays" if robust else "always_zero", "video": video, "horizon_steps": HORIZON, "epochs": 40, "expected_optimizer_steps": 11120, "checkpoint_rule": "epoch40_last", "status": "LOCKED", "pilot_seed": seed == SEEDS[0], "final_test_access": 0, "encoder_updated": False})
            for delay_index, delay in enumerate(DELAYS):
                eval_rows.append({"bundle_id": f"{run_id}-D{delay_index}", "run_id": run_id, "seed": seed, "cell": cell, "delay_index": delay_index, "delay_seconds": float(delay), "horizon_steps": HORIZON, "status": "LOCKED", "final_test_access": 0})
    with (EXP / "MATRIX_18.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix_rows[0])); writer.writeheader(); writer.writerows(matrix_rows)
    with (EXP / "EVAL_MATRIX_72.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(eval_rows[0])); writer.writeheader(); writer.writerows(eval_rows)
    write_json(EXP / "MATRIX_18.json", {"status": "LOCKED", "rows": matrix_rows, "count": 18, "written_before_new_dev_scores": True, "final_test_access": 0})
    write_json(EXP / "EVAL_MATRIX_72.json", {"status": "LOCKED", "rows": eval_rows, "count": 72, "final_test_access": 0})
    write_json(EXP / "SEED_REGISTRY.json", {"status": "LOCKED", "seeds": list(SEEDS), "excluded": [20260921, 20260922, 20260923, 20260939, 20260940, 20260941, 20260942, 20260944, 20260950, 20260951, 20260952, 20260953, 20260954, 20260955], "fresh_seed_rule": "selected before any new dev score", "final_test_access": 0})
    write_json(EXP / "PROTOCOL_LOCKED.json", {"status": "LOCKED", "protocol_id": "CORE_B_OBSERVATION_DELAY_ROBUSTNESS_V1", "task": "simulated_observation_feedback_delay", "horizon_steps": 60, "delay_seconds": [float(x) for x in DELAYS], "cells": list(CELLS), "seeds": list(SEEDS), "training_run_count": 18, "prediction_bundle_count": 72, "train_episodes": int(len(np.unique(data["episode_index"][train]))), "train_windows": int(len(train)), "primary_dev_episodes": int(len(np.unique(data["episode_index"][dev]))), "primary_dev_windows": int(len(dev)), "command_history_delayed": False, "anchor_command_delayed": False, "video_delayed": False, "encoder_updated": False, "new_video_extraction_started": False, "final_test_access": 0})
    write_json(EXP / "DELAY_TRANSFORM_CONTRACT.json", {"status": "LOCKED", "delay_definition": "cutoff=t0-delta; selected observation timestamp <= min(query_timestamp, cutoff); sample-and-hold observation only", "history_steps": 32, "observation_columns": [0, 43], "action_columns": [43, 71], "command_history_delayed": False, "anchor_command_delayed": False, "video_delayed": False, "target_future_unchanged": True, "age_formula": "max(0, query_timestamp-selected_observation_timestamp)", "native_then_normalize_equivalence_required": True, "deltas_seconds": [float(x) for x in DELAYS], "no_cross_episode": True, "final_test_access": 0})
    write_json(EXP / "TRAIN_DELAY_SCHEDULE.json", {"status": "LOCKED", "clean_policy": "delay_index=0", "robust_policy": "sha256(f'{seed}|{epoch}|{global_window_row_id}')[:8] integer modulo 4", "robust_probabilities": {str(float(delay)): 0.25 for delay in DELAYS}, "same_schedule_across_N_D_S": True, "schedule_independent_of_gt_task_motion_loss": True, "final_test_access": 0})
    timeline_rows = []
    for row in dev[: min(20, len(dev))]:
        timeline_rows.append({"anchor_id": str(data["anchor_id"][row]), "episode_index": int(data["episode_index"][row]), "anchor_timestamp": float(data["anchor_timestamp"][row]), "query_start": float(data["history_query_timestamps"][row, 0]), "query_end": float(data["history_query_timestamps"][row, -1]), "source_frame_start": int(data["history_selected_frame_indices"][row, 0]), "source_frame_end": int(data["history_selected_frame_indices"][row, -1]), "target_start_frame": int(data["anchor_frame"][row] + 1), "target_end_frame": int(data["anchor_frame"][row] + HORIZON), "final_test_access": 0})
    with (EXP / "DATA_TIMELINE.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timeline_rows[0])); writer.writeheader(); writer.writerows(timeline_rows)
    source_files = [V3 / "PROTOCOL_LOCKED.json", V3 / "data/V3_NORMALIZERS.json", V3 / "data/V3_VIDEO_NORMALIZER.json", V3 / "ANCHOR_MANIFEST.csv", V3 / "cache/CACHE_MANIFEST.json", SIV / "PROTOCOL_LOCKED.json", SIV / "SOURCE_REGISTRY.json", SIV / "INPUT_MASK_CONTRACT.json", SIV / "STATE_COLUMN_MAP.json", SIV / "qa/CHECKSUMS.sha256"]
    write_json(EXP / "SOURCE_REGISTRY.json", {"status": "LOCKED", "v3_root": str(V3), "siv_root": str(SIV), "files": {str(path.relative_to(REPO)): digest(path) for path in source_files}, "cache_reused_read_only": True, "cache_path": str(V3 / "cache/V3_TEMPORAL_FEATURES.npy"), "final_test_access": 0})
    write_json(EXP / "RESOURCE_ESTIMATE.json", {"status": "PENDING_STAGE0", "budget_upper_bound_gpu_hours": 12, "additional_disk_gib_upper_bound": 20, "training_runs": 18, "prediction_bundles": 72, "dynamic_all_visible": True, "max_project_gpu_children": 2, "co_location_allowed": True, "final_test_access": 0})
    (EXP / "OUTCOME_EXPOSURE.md").write_text("# Outcome exposure\n\nMATRIX_18 and EVAL_MATRIX_72 were written before any new dev score was read. No cell, seed or delay was removed based on results. Stage0 controls only engineering validity and the prelocked resource budget.\n", encoding="utf-8")
    (EXP / "CLAIM_LEDGER.md").write_text("# Claim ledger\n\n- Delay is an artificial, reproducible observation-feedback simulation, not a deployment/network-safety result.\n- Local command history, command anchor, video and future targets are not delayed.\n- A robust model helping at delayed inputs does not automatically improve clean-input/full-task performance.\n- This remains development-only; final/held-out access is forbidden.\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "matrix": 18, "eval_matrix": 72, "seeds": SEEDS, "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
