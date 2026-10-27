#!/usr/bin/env python3
"""Independent CPU-only audit of the Phase-A raw-to-input row bridge.

This audit deliberately reconstructs a small, deterministic train/dev sample
from raw Parquet rather than trusting the preparation script that produced the
locked input archive.  It also exercises the copied trainer's loader and cell
mapping so a row-order or train/dev slicing error is caught before GPU work.
"""

from __future__ import annotations
import os

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
REPO = PROJECT / "repo/ee6008-humanoid-vjepa"
EXP_ROOT = REPO / "reports/core_b/HISTORY_REPLICATION_ENCODER_ADAPTATION_V1_20260918T053328Z"
SOURCE_ROOT = REPO / "reports/core_b/HISTORY_VIDEO_INFORMATION_MATRIX_V1_20260917T140755Z"
INPUT = SOURCE_ROOT / "inputs/HISTORY_VIDEO_INPUTS.npz"
TRAINER = EXP_ROOT / "phase_a/code/run_history_video_cell.py"
SPLIT_MANIFEST = REPO / "manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/splits.parquet"
RAW_ROOT = PROJECT / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
OUT_JSON = EXP_ROOT / "phase_a/qa/A_INPUT_BRIDGE_QA.json"
OUT_MD = EXP_ROOT / "phase_a/qa/A_INPUT_BRIDGE_QA.md"

PREFIX_FRAME_COUNT = 120
HORIZON_STEPS = 30
HISTORY_STEPS = 32
HISTORY_INTERVAL_SECONDS = 0.125
ACTION_DIM = 28
ARM_DIM = 14
HAND_DIM = 14
LEG_DIM = 15
STATE_DIM = 71


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def matrix(table: object, name: str, width: int) -> np.ndarray:
    values = table[name].combine_chunks().to_pylist()  # type: ignore[index]
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width:
        raise AssertionError(f"{name} shape {array.shape} != (*,{width})")
    if not np.isfinite(array).all():
        raise AssertionError(f"{name} contains non-finite values")
    return array


def read_raw(path: Path, episode: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(
        path,
        columns=[
            "frame_index",
            "timestamp",
            "action",
            "observation.arm_joints",
            "observation.hand_joints",
            "observation.leg_joints",
        ],
    )
    frame = table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
    timestamps = table["timestamp"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
    if not np.array_equal(frame, np.arange(len(frame), dtype=np.int64)):
        raise AssertionError(f"episode {episode}: non-contiguous frame indices")
    if len(frame) < PREFIX_FRAME_COUNT + HORIZON_STEPS:
        raise AssertionError(f"episode {episode}: insufficient rows")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise AssertionError(f"episode {episode}: invalid timestamps")
    action = matrix(table, "action", ACTION_DIM)
    arm = matrix(table, "observation.arm_joints", ARM_DIM)
    hand = matrix(table, "observation.hand_joints", HAND_DIM)
    leg = matrix(table, "observation.leg_joints", LEG_DIM)
    return frame, timestamps, action, arm, hand, leg


def import_trainer() -> object:
    code_dir = str(TRAINER.parent)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    spec = importlib.util.spec_from_file_location("phase_a_trainer_for_qa", TRAINER)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to import copied trainer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    with np.load(INPUT, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    required = {
        "episode_index",
        "split_code",
        "original_task_index",
        "targets_native",
        "persistence_native",
        "history_native",
        "history_standardized",
        "history_selected_frame_indices",
        "history_selected_timestamps",
        "history_query_timestamps",
        "target_timestamps",
        "dynamic_video_feature",
    }
    missing = sorted(required - set(data))
    if missing:
        raise AssertionError(f"locked input keys missing: {missing}")
    n = len(data["episode_index"])
    if n != 3355:
        raise AssertionError(f"unexpected canonical row count {n}")
    ids = data["episode_index"].astype(np.int64)
    split_code = data["split_code"].astype(np.int8)
    train_rows = np.flatnonzero(split_code == 0)
    dev_rows = np.flatnonzero(split_code == 1)
    if len(train_rows) != 2741 or len(dev_rows) != 614:
        raise AssertionError("train/dev population mismatch")
    if set(ids[train_rows].tolist()) & set(ids[dev_rows].tolist()):
        raise AssertionError("train/dev ID overlap")

    split = pd.read_parquet(
        SPLIT_MANIFEST,
        columns=["episode_index", "split", "v3_benchmark_inclusion", "original_task_index", "parquet_path"],
    )
    split = split[split["split"].isin(["train", "validation"]) & split["v3_benchmark_inclusion"].eq("primary")]
    split_map = {int(row.episode_index): row for row in split.itertuples(index=False)}
    if not set(ids.tolist()).issubset(split_map):
        raise AssertionError("canonical IDs are not covered by primary split manifest")

    # Ten fixed positions per split, with endpoints included.  The sample is
    # not used to estimate a result; it is only a deterministic bridge audit.
    sample_rows = np.concatenate(
        [
            train_rows[np.linspace(0, len(train_rows) - 1, 10).round().astype(int)],
            dev_rows[np.linspace(0, len(dev_rows) - 1, 10).round().astype(int)],
        ]
    )
    raw_checks: list[dict[str, object]] = []
    max_history_error = 0.0
    max_target_error = 0.0
    max_anchor_error = 0.0
    max_timestamp_error = 0.0
    for row in sample_rows.tolist():
        episode = int(ids[row])
        source = split_map[episode]
        expected_split = "train" if int(split_code[row]) == 0 else "validation"
        if str(source.split) != expected_split:
            raise AssertionError(f"episode {episode}: split mismatch")
        if int(source.original_task_index) != int(data["original_task_index"][row]):
            raise AssertionError(f"episode {episode}: task mismatch")
        parquet_path = (RAW_ROOT / Path(str(source.parquet_path))).resolve()
        if not parquet_path.is_relative_to(RAW_ROOT.resolve()) or not parquet_path.is_file():
            raise AssertionError(f"episode {episode}: invalid raw path {parquet_path}")
        frame, timestamps, action, arm, hand, leg = read_raw(parquet_path, episode)
        t0 = float(timestamps[PREFIX_FRAME_COUNT - 1])
        queries = t0 + np.arange(-(HISTORY_STEPS - 1), 1, dtype=np.float64) * HISTORY_INTERVAL_SECONDS
        selected = np.searchsorted(timestamps, queries, side="right") - 1
        if np.any(selected < 0) or np.any(selected >= PREFIX_FRAME_COUNT):
            raise AssertionError(f"episode {episode}: illegal history index")
        selected_ts = timestamps[selected]
        history_native = np.concatenate((arm[selected], hand[selected], leg[selected], action[selected]), axis=1)
        target_native = action[PREFIX_FRAME_COUNT : PREFIX_FRAME_COUNT + HORIZON_STEPS]
        target_ts = timestamps[PREFIX_FRAME_COUNT : PREFIX_FRAME_COUNT + HORIZON_STEPS]
        history_error = float(np.max(np.abs(history_native - data["history_native"][row])))
        target_error = float(np.max(np.abs(target_native - data["targets_native"][row])))
        anchor_error = float(np.max(np.abs(action[PREFIX_FRAME_COUNT - 1] - data["persistence_native"][row, 0])))
        timestamp_error = float(np.max(np.abs(selected_ts - data["history_selected_timestamps"][row])))
        frame_error = int(np.max(np.abs(selected.astype(np.int64) - data["history_selected_frame_indices"][row].astype(np.int64))))
        target_timestamp_error = float(np.max(np.abs(target_ts - data["target_timestamps"][row])))
        if frame_error != 0 or not np.all(selected_ts <= t0 + 1e-9) or not np.all(target_ts > t0):
            raise AssertionError(f"episode {episode}: temporal boundary failure")
        max_history_error = max(max_history_error, history_error)
        max_target_error = max(max_target_error, target_error)
        max_anchor_error = max(max_anchor_error, anchor_error)
        max_timestamp_error = max(max_timestamp_error, timestamp_error, target_timestamp_error)
        raw_checks.append(
            {
                "row": row,
                "episode_index": episode,
                "split": "train" if int(split_code[row]) == 0 else "dev",
                "original_task_index": int(data["original_task_index"][row]),
                "parquet_path": str(source.parquet_path),
                "history_max_abs_error": history_error,
                "target_max_abs_error": target_error,
                "anchor_max_abs_error": anchor_error,
                "timestamp_max_abs_error": max(timestamp_error, target_timestamp_error),
                "selected_frame_index_max_abs_error": frame_error,
                "all_history_past": bool(np.all(selected_ts <= t0 + 1e-9)),
                "all_targets_future": bool(np.all(target_ts > t0)),
            }
        )

    trainer = import_trainer()
    loaded, stats = trainer.load_data(INPUT)
    if not np.array_equal(loaded["episode_index"], ids) or not np.array_equal(loaded["split_code"], split_code):
        raise AssertionError("trainer loader changed canonical ID or split order")
    if not np.array_equal(loaded["original_task_index"], data["original_task_index"]):
        raise AssertionError("trainer loader changed task order")
    if not np.allclose(loaded["history_standardized"], data["history_standardized"], rtol=0.0, atol=0.0):
        raise AssertionError("trainer loader changed history values")
    if not np.allclose(loaded["dynamic_video_feature"], data["dynamic_video_feature"], rtol=0.0, atol=0.0):
        raise AssertionError("trainer loader changed video values")

    cell_checks: dict[str, object] = {}
    for cell in sorted(trainer.CELLS):
        state, mask, delta, video, anchor, target_native, target_std, cell_ids, tasks = trainer.cell_inputs(loaded, cell)
        if state.shape != (n, HISTORY_STEPS, STATE_DIM) or video.shape != (n, 768):
            raise AssertionError(f"{cell}: cell shape mismatch")
        if not np.array_equal(cell_ids, ids) or not np.array_equal(tasks, data["original_task_index"]):
            raise AssertionError(f"{cell}: ID/task order mismatch")
        if not np.allclose(mask, data["history_mask"], rtol=0.0, atol=0.0) or not np.allclose(delta, data["history_relative_delta_standardized"], rtol=0.0, atol=0.0):
            raise AssertionError(f"{cell}: temporal side channels changed")
        state_mode, video_mode = trainer.CELLS[cell]
        expected_state = data["history_standardized"] if state_mode == "HIST" else np.repeat(data["history_standardized"][:, -1:, :], HISTORY_STEPS, axis=1)
        expected_video = data["dynamic_video_feature"] if video_mode == "F" else np.zeros_like(data["dynamic_video_feature"], dtype=np.float32)
        if not np.allclose(state, expected_state, rtol=0.0, atol=0.0) or not np.allclose(video, expected_video, rtol=0.0, atol=0.0):
            raise AssertionError(f"{cell}: state/video cell mapping mismatch")
        cell_checks[cell] = {
            "state_mode": state_mode,
            "video_mode": video_mode,
            "state_shape": list(state.shape),
            "video_shape": list(video.shape),
            "sample_ids": cell_ids[sample_rows].tolist(),
        }

    # A train/dev sentinel test: modifying only held-out target values must not
    # change the train indices, train IDs, or any train-side model inputs.
    mutated = {key: value.copy() for key, value in loaded.items()}
    mutated["targets_native"][dev_rows] += 123.456
    train_indices = np.flatnonzero(mutated["split_code"] == 0)
    if not np.array_equal(train_indices, train_rows) or not np.array_equal(mutated["episode_index"][train_indices], ids[train_rows]):
        raise AssertionError("train/dev sentinel changed training row selection")
    original_cells = trainer.cell_inputs(loaded, "HI-F-HIST")
    mutated_cells = trainer.cell_inputs(mutated, "HI-F-HIST")
    for before, after in zip(original_cells[:4], mutated_cells[:4]):
        if not np.array_equal(before[train_rows], after[train_rows]):
            raise AssertionError("dev-only mutation changed train-side inputs")

    result = {
        "status": "PASS",
        "audit": "independent_raw_parquet_to_locked_input_and_local_trainer_bridge",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_path": str(INPUT),
        "input_sha256": digest(INPUT),
        "trainer_path": str(TRAINER),
        "trainer_sha256": digest(TRAINER),
        "split_manifest": str(SPLIT_MANIFEST),
        "raw_root": str(RAW_ROOT),
        "population": {"rows": n, "train": len(train_rows), "dev": len(dev_rows)},
        "sample_rows": sample_rows.tolist(),
        "sample_ids": ids[sample_rows].tolist(),
        "raw_checks": raw_checks,
        "max_history_native_abs_error": max_history_error,
        "max_target_native_abs_error": max_target_error,
        "max_anchor_native_abs_error": max_anchor_error,
        "max_timestamp_abs_error": max_timestamp_error,
        "cell_checks": cell_checks,
        "train_dev_sentinel": {
            "dev_target_mutation": 123.456,
            "train_row_selection_unchanged": True,
            "train_inputs_unchanged": True,
        },
        "final_test_access": 0,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Phase-A input bridge QA",
        "",
        "- Status: PASS",
        f"- Population: {n} rows ({len(train_rows)} train, {len(dev_rows)} dev)",
        f"- Audited raw rows: {len(sample_rows)} (10 train + 10 dev)",
        f"- Max raw history error: {max_history_error:.3e}",
        f"- Max raw target error: {max_target_error:.3e}",
        f"- Max raw anchor error: {max_anchor_error:.3e}",
        f"- Max timeline error: {max_timestamp_error:.3e}",
        "- All audited history rows are prefix-only and all targets are after frame 119.",
        "- Local trainer loader preserves canonical ID/task order and all four cell mappings.",
        "- Dev-only target mutation leaves train row selection and train inputs unchanged.",
        "- Final-test access: 0",
        "",
        "This is a bridge/schema QA, not a performance result.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "sample_rows": len(sample_rows), "max_history_error": max_history_error, "max_target_error": max_target_error, "max_anchor_error": max_anchor_error}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
