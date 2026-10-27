#!/usr/bin/env python3
"""Fresh-process, target-blind H30 inference for one locked Hadamard cell."""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

TASK_ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), ''))
V3 = PROJECT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
BENCH = PROJECT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
EQ = PROJECT / "reports/benchmarks/CORE_B_D3_EPISODE_EQUAL_SENSITIVITY_V1_20260924T095800Z"
INPUTS = V3 / "data/V3_MAIN_INPUTS.npz"
CACHE_MANIFEST = BENCH / "d3/cache/D3_CACHE_MANIFEST.json"
VIDEO_NORMALIZER = BENCH / "d3/cache/D3_VIDEO_NORMALIZER.json"
TARGET_NORMALIZER = V3 / "data/V3_NORMALIZERS.json"
SPLIT_MANIFEST = PROJECT / "repo/ee6008-humanoid-vjepa/manifests/g1_6b6b599def75e3558b2736062e79bdbeec6cd224/splits/core_a_protocol_v3_duplicate_safe/split_manifest.parquet"

sys.path.insert(0, str(TASK_ROOT / "code"))
from d3_model import D3TimeSpacePredictor  # noqa: E402
from fixed_future_query_model import FixedFutureQueryOverride  # noqa: E402
from train_full_query import runtime_metadata  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def sha_query_table(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float32)
    h = hashlib.sha256(); h.update(array.dtype.str.encode("ascii")); h.update(b"\0")
    h.update(np.asarray(array.shape, dtype=np.int64).tobytes()); h.update(array.tobytes())
    return h.hexdigest()

def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


TARGET_BLIND_KEYS = (
    "anchor_id", "episode_index", "original_task_index", "anchor_frame", "anchor_timestamp",
    "history_standardized", "history_mask", "history_relative_delta_standardized", "anchor_command_native",
)


def select_target_blind_inputs(archive: Any, indices: np.ndarray) -> dict[str, np.ndarray]:
    # Intentionally do not access archive['targets_native'] or any target cache.
    return {key: archive[key][indices].copy() for key in TARGET_BLIND_KEYS}


class GridStore:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.shards = sorted(manifest["shards"], key=lambda item: int(item["start_row"]))
        self.starts = [int(item["start_row"]) for item in self.shards]

    def get(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dynamic = np.empty((len(indices), 16, 4, 768), dtype=np.float32)
        static = np.empty_like(dynamic)
        by_shard: dict[int, list[tuple[int, int]]] = {}
        for out_i, row_i in enumerate(indices.astype(np.int64, copy=False)):
            shard_i = bisect.bisect_right(self.starts, int(row_i)) - 1
            shard = self.shards[shard_i]
            if not int(shard["start_row"]) <= int(row_i) < int(shard["end_row"]):
                raise IndexError(f"feature cache misses input row {row_i}")
            by_shard.setdefault(shard_i, []).append((out_i, int(row_i) - int(shard["start_row"])))
        for shard_i, pairs in by_shard.items():
            values = np.load(self.shards[shard_i]["array"], mmap_mode="r", allow_pickle=False)
            for out_i, local_i in pairs:
                dynamic[out_i] = values[local_i, 0]
                static[out_i] = values[local_i, 1]
        return dynamic, static


def read_source_groups() -> dict[int, tuple[str, str]]:
    mapping: dict[int, tuple[str, str]] = {}
    with (TASK_ROOT / "inventory/SOURCE_GROUP_MAP.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            episode = int(row["episode_index"])
            value = (row["split"], row["source_group_id"])
            if episode in mapping and mapping[episode] != value:
                raise AssertionError(f"conflicting episode/source group mapping {episode}")
            mapping[episode] = value
    return mapping


def query_table_and_hash() -> tuple[np.ndarray, str]:
    locked = json.loads((TASK_ROOT / "protocol/QUERY_TABLES_AND_HASHES.json").read_text(encoding="utf-8"))
    path = TASK_ROOT / "protocol/HADAMARD_QUERY_TABLES.npz"
    if sha_file(path) != locked["archive_sha256"]:
        raise AssertionError("HADAMARD query archive SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        table = np.asarray(archive["candidate_h30"], dtype=np.float32).copy()
    expected = locked["candidate_table_sha256_h30"]
    if table.shape != (30, 128) or not np.isfinite(table).all() or sha_query_table(table) != expected:
        raise AssertionError("candidate H30 query table shape/finite/hash mismatch")
    return table, expected


def load_split_data(split: str, group_map: dict[int, tuple[str, str]], archive: Any) -> dict[str, Any]:
    codes = archive["split_code"].astype(np.int8, copy=True)
    if len(codes) != 20079 or set(np.unique(codes).tolist()) != {0, 1}:
        raise AssertionError("accepted train/dev only input population mismatch")
    code = 0 if split == "train" else 1
    indices = np.flatnonzero(codes == code).astype(np.int64)
    expected = 17757 if split == "train" else 2322
    if len(indices) != expected:
        raise AssertionError(f"{split} input row count {len(indices)} != {expected}")
    data = select_target_blind_inputs(archive, indices)
    ids = data["anchor_id"].astype("U96")
    episodes = data["episode_index"].astype(np.int64)
    groups = []
    for episode in episodes:
        mapped_split, group = group_map[int(episode)]
        if mapped_split != split:
            raise AssertionError(f"episode {episode} maps to {mapped_split}, expected {split}")
        groups.append(group)
    group_array = np.asarray(groups, dtype="U128")
    if len(np.unique(ids)) != expected or len(np.unique(episodes)) != (2712 if split == "train" else 603):
        raise AssertionError(f"{split} IDs/episode population mismatch")
    return {**data, "archive_indices": indices, "source_group_id": group_array, "split": split}


def predict_rows(model: torch.nn.Module, data: dict[str, Any], rows: np.ndarray, grid: GridStore,
                 video_mean: np.ndarray, video_std: np.ndarray, sigma: torch.Tensor,
                 device: torch.device, batch_size: int, output: np.ndarray | None = None) -> np.ndarray:
    if output is None:
        output = np.empty((len(rows), 30, 28), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            local = rows[start:end]
            source_indices = data["archive_indices"][local]
            dynamic, static = grid.get(source_indices)
            dynamic = ((dynamic - video_mean[None, None, None, :]) / video_std[None, None, None, :]).astype(np.float32)
            static = ((static - video_mean[None, None, None, :]) / video_std[None, None, None, :]).astype(np.float32)
            state = torch.as_tensor(data["history_standardized"][local], device=device, dtype=torch.float32)
            mask = torch.as_tensor(data["history_mask"][local], device=device, dtype=torch.float32)
            relative = torch.as_tensor(data["history_relative_delta_standardized"][local], device=device, dtype=torch.float32)
            anchor = torch.as_tensor(data["anchor_command_native"][local], device=device, dtype=torch.float32)
            residual = model(state, mask, relative, torch.from_numpy(dynamic).to(device),
                             torch.from_numpy(static).to(device), anchor)
            output[start:end] = (anchor[:, None, :] + residual * sigma).detach().cpu().numpy().astype(np.float32)
            if end == len(rows) or end % 2048 == 0:
                print(json.dumps({"status": "inference_progress", "split": data["split"], "rows_done": end,
                                  "rows_total": len(rows)}, sort_keys=True), flush=True)
    if not np.isfinite(output).all():
        raise AssertionError(f"non-finite {data['split']} native predictions")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", choices=("train", "dev", "both"), default="both")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size != 64 or args.threads < 1:
        raise ValueError("locked inference batch=64; threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.device == "cuda":
        physical_gpu = int(os.environ.get("GPU_WATCHER_PHYSICAL_GPU", "-1"))
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if physical_gpu < 0 or visible != str(physical_gpu):
            raise RuntimeError("GPU watcher physical/logical mapping is absent or inconsistent")
        if not torch.cuda.is_available():
            raise RuntimeError("watcher-assigned CUDA device is unavailable in this process")
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        physical_gpu = None
        device = torch.device("cpu")

    lock_path = TASK_ROOT / "protocol/PROTOCOL_LOCKED.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_sha = sha_file(lock_path)
    code_hashes = lock.get("task_code_hashes", {})
    if code_hashes.get("code/infer_full_query.py") != sha_file(Path(__file__).resolve()):
        raise AssertionError("inference source hash differs from the locked protocol")
    matrix_path = TASK_ROOT / "protocol/MATRIX12.csv"
    if sha_file(matrix_path) != lock["matrix12_sha256"]:
        raise AssertionError("MATRIX12 hash mismatch")
    with matrix_path.open(newline="", encoding="utf-8") as handle:
        matrix = list(csv.DictReader(handle))
    selected = [row for row in matrix if row["action"] == "train_had" and row["run_id"] == args.run_id]
    if len(matrix) != 12 or len(selected) != 1:
        raise AssertionError("requested new Hadamard run is absent from locked MATRIX12")
    row = selected[0]
    if row["config"] not in {"TS11-D", "STATE-CTRL"} or int(row["horizon_steps"]) != 30:
        raise AssertionError("inference matrix row is outside the locked H30 study")
    query_table, query_sha = query_table_and_hash()
    run_root = TASK_ROOT / "runs/formal" / args.run_id
    complete = json.loads((run_root / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
    checkpoint = run_root / "checkpoint_last.pt"
    checkpoint_sha = sha_file(checkpoint)
    if (complete.get("status") != "PASS" or int(complete.get("epochs", -1)) != 40
            or int(complete.get("optimizer_steps", -1)) != 11120
            or complete.get("study_protocol_hash") != lock_sha
            or complete.get("query_table_sha256") != query_sha
            or complete.get("checkpoint_last_sha256") != checkpoint_sha):
        raise AssertionError("formal checkpoint/protocol/query acceptance failed before inference")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base = D3TimeSpacePredictor(horizon=30, seed=int(row["seed"]), video_mode=row["config"])
    model = FixedFutureQueryOverride(base, torch.from_numpy(query_table))
    model.load_state_dict(payload["model"], strict=True)
    if sha_query_table(model.future_query_table.detach().cpu().numpy()) != query_sha:
        raise AssertionError("checkpoint reload restored or changed the future-query table")
    model.to(device)
    model.eval()

    # Same fixed 128 train-only monitor rows, order and batch size as the training run.
    with (TASK_ROOT / "protocol/TRAIN_MONITOR_PANEL.csv").open(newline="", encoding="utf-8") as handle:
        monitor_rows = list(csv.DictReader(handle))
    monitor_ids = np.asarray([item["anchor_id"] for item in monitor_rows], dtype="U96")
    if len(monitor_ids) != 128 or len(set(monitor_ids.tolist())) != 128:
        raise AssertionError("locked monitor panel schema/count mismatch")
    archive = np.load(INPUTS, allow_pickle=False)
    group_map = read_source_groups()
    split_names = ("train", "dev") if args.split == "both" else (args.split,)
    split_data = {split: load_split_data(split, group_map, archive) for split in split_names}
    if "train" not in split_data:
        train_data = load_split_data("train", group_map, archive)
    else:
        train_data = split_data["train"]
    train_rows_by_id = {value: i for i, value in enumerate(train_data["anchor_id"].astype("U96").tolist())}
    if any(value not in train_rows_by_id for value in monitor_ids.tolist()):
        raise AssertionError("train monitor panel IDs are not in primary train")
    monitor_positions = np.asarray([train_rows_by_id[value] for value in monitor_ids.tolist()], dtype=np.int64)

    cache_manifest = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))
    video_norm = json.loads(VIDEO_NORMALIZER.read_text(encoding="utf-8"))
    target_norm = json.loads(TARGET_NORMALIZER.read_text(encoding="utf-8"))
    grid = GridStore(cache_manifest)
    video_mean = np.asarray(video_norm["mean"], dtype=np.float32)
    video_std = np.asarray(video_norm["safe_std"], dtype=np.float32)
    sigma = torch.as_tensor(np.asarray(target_norm["target_sigma"], dtype=np.float32), device=device).view(1, 1, 28)

    replay_path = run_root / "replay_reference.npz"
    if sha_file(replay_path) != complete.get("replay_reference_sha256"):
        raise AssertionError("train replay reference hash mismatch")
    with np.load(replay_path, allow_pickle=False) as replay:
        reference_ids = replay["anchor_id"].astype("U96")
        reference = replay["predictions_native"].astype(np.float32, copy=True)
        if str(replay["query_table_sha256"].item()) != query_sha:
            raise AssertionError("replay reference query hash mismatch")
    if not np.array_equal(reference_ids, monitor_ids) or reference.shape != (128, 30, 28):
        raise AssertionError("replay reference IDs/shape mismatch")
    replay_pred = predict_rows(model, train_data, monitor_positions, grid, video_mean, video_std,
                               sigma, device, args.batch_size)
    max_abs = float(np.max(np.abs(replay_pred - reference)))
    max_rel = float(np.max(np.abs(replay_pred - reference) / np.maximum(np.abs(reference), 1e-6)))
    replay_ok = bool(np.allclose(replay_pred, reference, atol=1e-5, rtol=1e-5))
    replay_acceptance = {
        "status": "PASS" if replay_ok else "FAIL",
        "run_id": args.run_id,
        "fresh_process": True,
        "checkpoint_sha256": checkpoint_sha,
        "query_table_sha256": query_sha,
        "split": "primary_train_monitor_only",
        "rows": 128,
        "batch_size": args.batch_size,
        "dtype": "float32",
        "device": str(device),
        "physical_gpu": physical_gpu,
        "logical_device": "cuda:0" if physical_gpu is not None else "cpu",
        "runtime": runtime_metadata(device) if physical_gpu is not None else {"torch":str(torch.__version__),"dtype":"float32","device":"cpu"},
        "atol": 1e-5,
        "rtol": 1e-5,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "prediction_allclose": replay_ok,
        "target_arrays_loaded": False,
        "final_test_access": 0,
    }
    atomic_json(run_root / "RUN_RELOAD_ACCEPTANCE.json", replay_acceptance)
    if not replay_ok:
        raise AssertionError(f"fresh-process replay failed: max_abs={max_abs}, max_rel={max_rel}")

    if args.split == "both":
        requested = ("train", "dev")
    else:
        requested = (args.split,)
    all_ok = True
    for split in requested:
        data = split_data[split]
        expected_rows = 17757 if split == "train" else 2322
        ids = data["anchor_id"].astype("U96")
        ids_hash = sha_array(ids)
        pred_root = TASK_ROOT / "predictions" / split / args.run_id
        pred_root.mkdir(parents=True, exist_ok=True)
        accepted_existing = False
        if args.resume:
            for side_path in sorted(pred_root.glob("attempt_*.json")):
                side = json.loads(side_path.read_text(encoding="utf-8"))
                candidate = pred_root / side.get("prediction_file", "")
                if (side.get("status") == "PASS" and side.get("run_id") == args.run_id
                        and side.get("split") == split and side.get("ids_sha256") == ids_hash
                        and side.get("checkpoint_sha256") == checkpoint_sha and side.get("query_table_sha256") == query_sha
                        and candidate.is_file() and sha_file(candidate) == side.get("prediction_sha256")):
                    print(json.dumps({"status": "PASS_REUSED", "run_id": args.run_id, "split": split,
                                      "prediction_file": str(candidate)}, sort_keys=True), flush=True)
                    accepted_existing = True
                    break
        if accepted_existing:
            continue
        rows = np.arange(expected_rows, dtype=np.int64)
        start = time.monotonic()
        predictions = predict_rows(model, data, rows, grid, video_mean, video_std, sigma, device, args.batch_size)
        target_steps = np.arange(1, 31, dtype=np.int64)
        target_frames = data["anchor_frame"][:, None].astype(np.int64) + target_steps[None, :]
        target_times = data["anchor_timestamp"][:, None].astype(np.float64) + target_steps[None, :].astype(np.float64) / 30.0
        anchors = data["anchor_command_native"].astype(np.float32, copy=False)
        attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        prediction_file = f"attempt_{attempt_id}.npz"
        prediction_path = pred_root / prediction_file
        tmp = prediction_path.with_name(f".{prediction_path.name}.{os.getpid()}.part")
        with tmp.open("wb") as handle:
            np.savez_compressed(
                handle,
                anchor_id=ids,
                episode_index=data["episode_index"].astype(np.int64, copy=False),
                source_group_id=data["source_group_id"],
                original_task_index=data["original_task_index"].astype(np.int64, copy=False),
                anchor_frame=data["anchor_frame"].astype(np.int64, copy=False),
                anchor_timestamp=data["anchor_timestamp"].astype(np.float64, copy=False),
                anchor_command_native=anchors,
                target_frame_indices=target_frames,
                target_timestamp_seconds=target_times,
                predictions_native=predictions,
                horizon_steps=np.asarray(30, dtype=np.int64),
                config=np.asarray(row["config"]),
                seed=np.asarray(int(row["seed"]), dtype=np.int64),
                split=np.asarray(split),
                query_table_sha256=np.asarray(query_sha),
                checkpoint_sha256=np.asarray(checkpoint_sha),
            )
            handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, prediction_path)
        prediction_sha = sha_file(prediction_path)
        side = {
            "schema_version": "1.0",
            "status": "PASS" if len(ids) == expected_rows and predictions.shape == (expected_rows, 30, 28) else "FAIL",
            "run_id": args.run_id,
            "config": row["config"],
            "seed": int(row["seed"]),
            "horizon_steps": 30,
            "split": split,
            "rows": int(len(ids)),
            "expected_rows": expected_rows,
            "prediction_shape": [int(x) for x in predictions.shape],
            "ids_sha256": ids_hash,
            "source_group_map_sha256": sha_file(TASK_ROOT / "inventory/SOURCE_GROUP_MAP.csv"),
            "checkpoint_sha256": checkpoint_sha,
            "query_table_sha256": query_sha,
            "prediction_sha256": prediction_sha,
            "prediction_file": prediction_file,
            "prediction_path": str(prediction_path),
            "feature_cache_manifest_sha256": sha_file(CACHE_MANIFEST),
            "video_normalizer_sha256": sha_file(VIDEO_NORMALIZER),
            "target_normalizer_sha256": sha_file(TARGET_NORMALIZER),
            "study_protocol_sha256": lock_sha,
            "target_timestamp_formula": "anchor_timestamp_seconds + target_step/30; target_frame=anchor_frame+target_step; target_step=1..30",
            "target_arrays_loaded": False,
            "dev_targets_opened_by_inference": False,
            "optimizer_created": False,
            "backward_called": False,
            "optimizer_steps": 0,
            "device": str(device),
            "physical_gpu": physical_gpu,
            "logical_device": "cuda:0" if physical_gpu is not None else "cpu",
            "dtype": "float32",
            "runtime": runtime_metadata(device) if physical_gpu is not None else {"torch":str(torch.__version__),"dtype":"float32","device":"cpu"},
            "batch_size": args.batch_size,
            "fresh_process_reload_acceptance": replay_ok,
            "fresh_process_replay_max_abs_diff": max_abs,
            "fresh_process_replay_max_rel_diff": max_rel,
            "elapsed_seconds": round(time.monotonic() - start, 6),
            "created_utc": utc_now(),
            "final_test_access": 0,
            "heldout_access": 0,
        }
        side_path = pred_root / f"attempt_{attempt_id}.json"
        atomic_json(side_path, side)
        if side["status"] != "PASS":
            all_ok = False
            raise AssertionError(f"prediction bundle shape/row acceptance failed for {args.run_id}/{split}")
        print(json.dumps({"status": side["status"], "run_id": args.run_id, "split": split,
                          "rows": len(ids), "prediction_sha256": prediction_sha,
                          "path": str(prediction_path), "elapsed_seconds": side["elapsed_seconds"]}, sort_keys=True), flush=True)
    if not all_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
