#!/usr/bin/env python3
"""Freeze P1/P8 train-only memorization panels and the eight-run matrix."""
from __future__ import annotations
import os

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
SCHED = PROJECT / "reports/benchmarks/D3_EQ_CORRECTED_SCHEDULER_V1_20261006T073406Z"
EQ = PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
DIAG = PROJECT / "reports/benchmarks/D3_EQ_REPORT_LEARNING_DIAGNOSIS_V1_20261006T022541Z"
BENCH = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
sys.path.insert(0, str(SCHED / "code"))
from d3_train_corrected import load_train_bundle, sha256_file  # noqa: E402
sys.path.insert(0, str(ROOT / "code"))
from collapse_diagnostic_core import (  # noqa: E402
    choose_nested_high_panel,
    exact_input_signature,
    stable_id_hash,
)
from replay_temporal_variation import score_motion_strata, train_group_map  # noqa: E402

PROTOCOL_ID = "CORE_B_COLLAPSE_LEARNABILITY_DIAGNOSTIC_V1"
SEED = 20261002
HORIZON = 30
INIT_PATH = BENCH / f"d3/runs/INIT_S{SEED}.pt"
OUT_PANEL = ROOT / "protocol/B_PANEL_SELECTION_LOCK.json"
OUT_PROTOCOL = ROOT / "protocol/B_PROTOCOL_LOCKED.json"
OUT_MATRIX = ROOT / "protocol/B_MATRIX_8.csv"
OUT_EXCLUSIONS = ROOT / "protocol/B_PANEL_COLLISION_EXCLUSIONS.csv"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        fields = ["status"]
        rows = [{"status": "no_exact_input_collisions_excluded"}]
    else:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def persistence_metrics(bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    target = bundle["targets60"][:, :30].astype(np.float64)
    anchor = bundle["anchors"].astype(np.float64)
    sigma = bundle["target_sigma"].numpy().astype(np.float64)
    safe_fd = bundle["safe_fd_scale"].numpy().astype(np.float64)
    residual = (target - anchor[:, None, :]) / sigma[None, None, :]
    pos_error = np.abs(residual)
    pos = 0.5 * (pos_error[..., :14].mean(axis=(1, 2)) + pos_error[..., 14:].mean(axis=(1, 2)))
    delta = np.diff(target, axis=1) / safe_fd[None, None, :]
    fd_error = np.abs(delta)
    fd = 0.5 * (fd_error[..., :14].mean(axis=(1, 2)) + fd_error[..., 14:].mean(axis=(1, 2)))
    return pos, fd


def main() -> int:
    if not (ROOT / "A_FORENSIC_REPORT.json").is_file():
        raise FileNotFoundError("Phase A report/gate must complete before panel selection")
    a_report = read_json(ROOT / "A_FORENSIC_REPORT.json")
    if a_report.get("status") != "PHASE_A_COMPLETE_B_GATE_PASS":
        raise AssertionError("Phase A hard identity/formula gate did not pass")
    if OUT_PANEL.exists() or OUT_PROTOCOL.exists() or OUT_MATRIX.exists():
        raise FileExistsError("B protocol/panel/matrix already exist; refusing to overwrite")
    bundle = load_train_bundle()
    per_window, labels, thresholds = score_motion_strata(bundle)
    pos_persistence, fd_persistence = persistence_metrics(bundle)
    groups_by_episode = train_group_map()
    high = np.flatnonzero(np.asarray([labels[groups_by_episode[int(e)]] == "high" for e in bundle["episode_ids"]]))
    eligible = high[(pos_persistence[high] > 0.0) & (fd_persistence[high] > 0.0)]
    if not len(eligible):
        raise AssertionError("no high-motion rows have nonzero Persistence MAE and FD denominators")
    records: dict[str, dict[str, Any]] = {}
    metadata = {}
    with np.load(PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_MAIN_INPUTS.npz", allow_pickle=False) as archive:
        train_indices = bundle["train_indices"]
        for key in ("anchor_id", "episode_index", "original_task_index", "split_code"):
            metadata[key] = np.asarray(archive[key][train_indices]).copy()
        if np.any(metadata["split_code"] != 0):
            raise AssertionError("non-train row entered the tiny-fit candidate pool")
    id_order = sorted(eligible.tolist(), key=lambda i: stable_id_hash(PROTOCOL_ID, str(bundle["train_ids"][i])))
    for position in id_order:
        episode = int(bundle["episode_ids"][position])
        group = groups_by_episode[episode]
        anchor_id = str(bundle["train_ids"][position])
        records[anchor_id] = {
            "anchor_id": anchor_id,
            "train_position": int(position),
            "source_row_index": int(bundle["train_indices"][position]),
            "episode_id": episode,
            "source_group_id": group,
            "original_task_index": int(bundle["task_ids"][position]),
            "stratum": "high",
            "selection_hash": stable_id_hash(PROTOCOL_ID, anchor_id),
            "persistence_position_error_normalized": float(pos_persistence[position]),
            "persistence_fd_error_safe_scale_normalized": float(fd_persistence[position]),
            "target_signature": hashlib.sha256(np.ascontiguousarray(bundle["targets60"][position, :30]).tobytes()).hexdigest(),
        }
    # P1 is the lowest protocol+ID SHA-256 among high-motion rows with positive denominators.
    p1 = records[str(bundle["train_ids"][id_order[0]])]
    video_mean = bundle["video_mean"]
    video_std = bundle["video_std"]
    state_all = bundle["inputs"]["history_standardized"]
    mask_all = bundle["inputs"]["history_mask"]
    rel_all = bundle["inputs"]["history_relative_delta_standardized"]
    anchor_all = bundle["anchors"]
    # Build exact effective-input signatures in bounded batches; never use predictions/errors.
    eligible_by_position = set(eligible.tolist())
    order_by_position = sorted(eligible_by_position)
    for start in range(0, len(order_by_position), 64):
        positions = np.asarray(order_by_position[start:start + 64], dtype=np.int64)
        source_rows = bundle["train_indices"][positions]
        dynamic, _static = bundle["grid_store"].get(source_rows)
        dynamic_norm = ((dynamic - video_mean[None, None, None, :]) / video_std[None, None, None, :]).astype(np.float32)
        for local, position in enumerate(positions.tolist()):
            anchor_id = str(bundle["train_ids"][position])
            core = {
                "state": state_all[position],
                "history_mask": mask_all[position],
                "history_relative_delta": rel_all[position],
                "anchor_command": anchor_all[position],
            }
            rec = records[anchor_id]
            rec["signature_STATE-CTRL"] = exact_input_signature(core)
            rec["signature_TS11-D"] = exact_input_signature({**core, "dynamic_grid_normalized": dynamic_norm[local]})
            rec["normalized_dynamic_sha256"] = hashlib.sha256(np.ascontiguousarray(dynamic_norm[local]).tobytes()).hexdigest()
    candidates = list(records.values())
    p8, rejected = choose_nested_high_panel(p1, candidates, n_total=8, protocol_id=PROTOCOL_ID)
    if len(p8) != 8 or p8[0]["anchor_id"] != p1["anchor_id"]:
        raise AssertionError("P8 selection is not nested around locked P1")
    if len({row["source_group_id"] for row in p8}) != 8:
        raise AssertionError("P8 does not have eight distinct source groups")
    selected_by_id = {row["anchor_id"]: row for row in p8}
    harmful_collisions = []
    excluded_collisions = []
    for rej in rejected:
        if rej["reason"] != "exact_input_collision":
            continue
        candidate = records[str(rej["anchor_id"])]
        matches = [row for row in p8 if any(candidate[f"signature_{model}"] == row[f"signature_{model}"]
                                            for model in ("TS11-D", "STATE-CTRL"))]
        for chosen in matches:
            target_differs = candidate["target_signature"] != chosen["target_signature"]
            item = {"excluded_anchor_id": candidate["anchor_id"], "excluded_source_group_id": candidate["source_group_id"],
                    "collides_with_anchor_id": chosen["anchor_id"], "model_input_collision": True,
                    "future_target_differs": target_differs,
                    "reason": "exact_effective_input_collision_with_different_target" if target_differs else "exact_effective_input_collision_same_target"}
            excluded_collisions.append(item)
            if target_differs:
                harmful_collisions.append(item)
    write_csv(OUT_EXCLUSIONS, excluded_collisions)
    init_registry = read_json(EQ / "protocol/D3_EQ_SEED_REGISTRY.json")["seeds"][str(SEED)]
    source_acceptance = read_json(ROOT / "SOURCE_ACCEPTANCE.json")
    init_sha = sha256_file(INIT_PATH)
    if init_sha != source_acceptance["seed_20261002_init"]["sha256"] or init_sha != init_registry["init_state_file_sha256"]:
        raise AssertionError("original seed-20261002 INIT file identity mismatch")
    panel_rows: dict[str, list[dict[str, Any]]] = {"P1": [p1], "P8": p8}
    panel_records = []
    for panel, rows in panel_rows.items():
        for order, row in enumerate(rows, start=1):
            panel_records.append({
                **row, "panel": panel, "panel_order": order,
                "panel_objective_weight": 1.0,
                "state_input_sha256": row["signature_STATE-CTRL"],
                "TS11-D_effective_input_sha256": row["signature_TS11-D"],
            })
    source_hashes = {
        "phase_a_report": sha256_file(ROOT / "A_FORENSIC_REPORT.json"),
        "phase_a_fixture": sha256_file(ROOT / "TRAIN_FORENSIC_FIXTURE.json"),
        "phase_a_replay": sha256_file(ROOT / "forensics/A_REPLAY_3/A_REPLAY.json"),
        "phase_a_repeatability": sha256_file(ROOT / "forensics/REPLAY_REPEATABILITY.json"),
        "v3_input_archive": sha256_file(PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_MAIN_INPUTS.npz"),
        "train_target_cache": sha256_file(EQ / "d3/train_targets_trainonly.npz"),
        "target_cache_manifest": sha256_file(EQ / "d3/TRAIN_TARGET_CACHE_MANIFEST.json"),
        "feature_cache_manifest": sha256_file(BENCH / "d3/cache/D3_CACHE_MANIFEST.json"),
        "video_normalizer": sha256_file(BENCH / "d3/cache/D3_VIDEO_NORMALIZER.json"),
        "target_normalizer": sha256_file(PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z/data/V3_NORMALIZERS.json"),
        "motion_threshold": sha256_file(SCHED / "MOTION_ACTIVITY_THRESHOLD.json"),
        "init_file": init_sha,
        "locked_model_source": sha256_file(SCHED / "code/d3_model.py"),
        "loss_contract_source": sha256_file(SCHED / "code/d3_eq_contract.py"),
    }
    panel_lock = {
        "schema_version": 1, "status": "LOCKED_BEFORE_STAGE0_OR_TRAINING",
        "protocol_id": PROTOCOL_ID, "locked_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "selection_rule": {
            "P1": "minimum SHA256(protocol_id NUL anchor_id) among primary-train H30 high-motion rows with positive Persistence position and FD denominators; no model errors used",
            "P8": "nested P1 plus seven distinct video source groups; prioritize new original_task_index then SHA256 order; reject exact effective-input collisions for either model; no fit/error-based selection",
            "strata": "frozen H30 train source-group score >= Q75",
            "panel_weights": "one row from each distinct source group; all objective weights equal 1",
        },
        "thresholds": thresholds,
        "candidate_support": {"high_motion_rows_with_positive_persistence_denominators": int(len(eligible)),
                              "distinct_high_motion_source_groups": int(len(set(records[x]["source_group_id"] for x in records))),
                              "exact_input_collision_rows_excluded": len(set(x["excluded_anchor_id"] for x in excluded_collisions)),
                              "collision_rows_with_different_future_target": len(harmful_collisions)},
        "persistence_denominator_minima": {"P1_position": p1["persistence_position_error_normalized"],
                                             "P1_fd": p1["persistence_fd_error_safe_scale_normalized"],
                                             "P8_min_position": min(x["persistence_position_error_normalized"] for x in p8),
                                             "P8_min_fd": min(x["persistence_fd_error_safe_scale_normalized"] for x in p8)},
        "init": {"path": str(INIT_PATH), "file_sha256": init_sha, "state_dict_hash": init_registry["init_state_hash"]},
        "source_hashes": source_hashes,
        "dev_targets_opened": False, "final_or_heldout_access": 0,
        "rows": panel_records,
    }
    write_json(OUT_PANEL, panel_lock)
    panel_hash = sha256_file(OUT_PANEL)
    phase_b = {
        "schema_version": 1, "status": "LOCKED_BEFORE_STAGE0_OR_TRAINING",
        "protocol_id": PROTOCOL_ID, "panel_lock_sha256": panel_hash,
        "matrix_rows": 8, "models": ["TS11-D", "STATE-CTRL"], "horizon_steps": HORIZON,
        "seed": SEED, "init_state_file_sha256": init_sha,
        "panels": {"P1": 1, "P8": 8, "nested": True, "one_row_per_source_group": True},
        "losses": {"FULL_LOSS": "L_pos + 0.1*L_delta", "POS_ONLY": "L_pos"},
        "training_conditions": {
            "updates_per_run": 2000, "eval_interval_updates": 100,
            "optimizer": "AdamW", "learning_rate": 0.001, "betas": [0.9, 0.999], "eps": 1e-8,
            "weight_decay": 0.0, "grad_clip_norm": 1.0, "scheduler": None,
            "dropout": "all nn.Dropout p=0; MultiheadAttention dropout=0; model.eval(); verified",
            "dtype": "float32", "batch": "entire P1 or P8 each real optimizer update",
            "panel_weights": "equal 1 per row; one row per source group",
            "checkpoint": "rotating resumable state every100 updates; final endpoint fixed update2000",
            "initialization": "every run reloads the same accepted original INIT_S20261002; never warm-start",
        },
        "evaluation": {
            "position_metric": "locked episode-equal normalized Full28 L_pos",
            "fd_metric": "locked episode-equal safe-FD-normalized future-internal L_delta; H30 29 differences, no seam",
            "tv_ratio": "mean(abs(diff(pred))/safe_fd_scale) / mean(abs(diff(target))/safe_fd_scale)",
            "fit_label": "strong_training_fit only if every panel row has position error <=0.10*its Persistence, FD error <=0.20*its Persistence, and TV ratio in [0.8,1.2]; report panel means and every row",
            "other_metrics": ["native prediction amplitude", "direction accuracy on active train differences using frozen train thresholds", "within-sample time variance", "cross-sample time-mean offset dispersion"],
            "no_best_step_selection": True,
        },
        "stage0": {"p1_cells": 4, "updates_each": 4, "factorial": "2 models x 2 losses on P1", "fresh_process_save_reload_verification": True},
        "smoke_cost_policy": {"run_four_update_timing_probe_for_all_8_cells": True,
                              "choose_one_common_device_before_formal_runs": True,
                              "cpu_if_projected_total_seconds_le_7200": True,
                              "otherwise": "stop before formal fits and request host approval for physical GPU2; no sandbox GPU probe"},
        "resource_limits": {"max_gpu_hours": 2.0, "max_gpu_memory_gib": 5.0, "max_cpu_wall_hours": 2.0,
                            "project_gpu_child_limit": 2, "own_gpu_concurrency": 1, "disk_free_fraction_min": 0.1},
        "prohibitions": {"dev_model_selection": False, "dev_targets_opened": False,
                         "final_or_heldout_access": 0, "new_benchmark_candidate": False,
                         "new_full_data_benchmark_training": False, "D5_training": False},
        "source_hashes": source_hashes,
    }
    write_json(OUT_PROTOCOL, phase_b)
    protocol_hash = sha256_file(OUT_PROTOCOL)
    matrix_rows = []
    for model in ("TS11-D", "STATE-CTRL"):
        for loss in ("FULL_LOSS", "POS_ONLY"):
            for panel in ("P1", "P8"):
                matrix_rows.append({
                    "run_id": f"TINYFIT-{model}-H30-{loss}-{panel}-S{SEED}",
                    "config": model, "horizon_steps": HORIZON, "loss": loss,
                    "panel": panel, "seed": SEED, "updates": 2000,
                    "protocol_sha256": protocol_hash, "panel_lock_sha256": panel_hash,
                    "init_file_sha256": init_sha, "new_benchmark_candidate": "false",
                })
    with OUT_MATRIX.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(matrix_rows[0]))
        writer.writeheader(); writer.writerows(matrix_rows)
    print(json.dumps({
        "status": "LOCKED_BEFORE_STAGE0_OR_TRAINING", "P1": p1["anchor_id"],
        "P8": [x["anchor_id"] for x in p8], "P8_tasks": [x["original_task_index"] for x in p8],
        "P8_distinct_groups": len(set(x["source_group_id"] for x in p8)),
        "eligible_high_rows": len(eligible), "input_collision_rows_excluded": len(set(x["excluded_anchor_id"] for x in excluded_collisions)),
        "different_target_collisions_excluded": len(harmful_collisions),
        "panel_lock_sha256": panel_hash, "protocol_sha256": protocol_hash,
        "matrix_sha256": sha256_file(OUT_MATRIX),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
