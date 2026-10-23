"""Fresh-process D3-EQ inference on the accepted fixed 3,718-query dense panel."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from d3_model import D3TimeSpacePredictor


EQ_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EQ_ROOT.parents[2]
BENCH_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_LAB_V1_20260920T065538Z"
RESULT_ROOT = PROJECT_ROOT / "reports/benchmarks/CORE_B_BENCHMARK_RESULTS_D4_V1_20260923T050400Z"
EXT_ROOT = RESULT_ROOT / "d3_extension"
DENSE_INPUTS = BENCH_ROOT / "d2/D2_DENSE_INPUTS.npz"
DENSE_CACHE_MANIFEST = EXT_ROOT / "cache/D3_DENSE_EXTRACTION_RUN_MANIFEST.json"
PRED_ROOT = EQ_ROOT / "d3/dense_predictions"
PROTOCOL = EQ_ROOT / "protocol/D3_EQ_PROTOCOL_LOCKED.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(part, path)


class GridStore:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self.shards = sorted(manifest["shards"], key=lambda row: int(row["start_row"]))
        self.starts = [int(row["start_row"]) for row in self.shards]

    def get(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dynamic = np.empty((len(indices), 16, 4, 768), dtype=np.float32)
        static = np.empty_like(dynamic)
        grouped: dict[int, list[tuple[int, int]]] = {}
        for out_index, source_index in enumerate(indices.astype(int)):
            shard_index = bisect.bisect_right(self.starts, int(source_index)) - 1
            shard = self.shards[shard_index]
            if not int(shard["start_row"]) <= int(source_index) < int(shard["end_row"]):
                raise IndexError(f"dense D3 feature cache does not cover row {source_index}")
            grouped.setdefault(shard_index, []).append((out_index, int(source_index) - int(shard["start_row"])))
        for shard_index, pairs in grouped.items():
            values = np.load(self.shards[shard_index]["array"], mmap_mode="r", allow_pickle=False)
            for out_index, local in pairs:
                dynamic[out_index] = values[local, 0]
                static[out_index] = values[local, 1]
        return dynamic, static


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size != 64:
        raise ValueError("dense inference batch size is locked to 64")
    if not torch.cuda.is_available():
        raise RuntimeError("D3-EQ dense inference requires a separately approved watcher child")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    import csv
    with (EQ_ROOT / "protocol/D3_EQ_MATRIX_36.csv").open(newline="", encoding="utf-8") as handle:
        matrix = list(csv.DictReader(handle))
    matches = [row for row in matrix if row["run_id"] == args.run_id]
    if len(matches) != 1:
        raise ValueError(f"run ID is not unique in D3-EQ matrix: {args.run_id}")
    config, horizon, seed = str(matches[0]["config"]), int(matches[0]["horizon_steps"]), int(matches[0]["seed"])
    run_root = EQ_ROOT / "d3/runs" / args.run_id
    complete = json.loads((run_root / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
    checkpoint = run_root / "checkpoint_last.pt"
    if complete.get("status") != "PASS" or int(complete.get("epochs", -1)) != 40 or int(complete.get("optimizer_steps", -1)) != 11120:
        raise AssertionError("D3-EQ checkpoint is not accepted at epoch40/11,120 steps")
    if not checkpoint.is_file() or sha256_file(checkpoint) != complete.get("checkpoint_last_sha256"):
        raise AssertionError("D3-EQ epoch40 checkpoint hash mismatch")
    lock = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if complete.get("protocol_hash") != sha256_file(PROTOCOL):
        raise AssertionError("run completion protocol hash differs from the locked protocol")

    archive = np.load(DENSE_INPUTS, allow_pickle=False)
    keys = (
        "dense_id", "episode_index", "original_task_index", "anchor_frame", "anchor_timestamp",
        "history_standardized", "history_mask", "history_relative_delta_standardized", "anchor_command_native",
    )
    data = {key: archive[key].copy() for key in keys}
    ids = data["dense_id"].astype("U96")
    if len(ids) != 3718 or len(set(ids.tolist())) != 3718 or len(np.unique(data["episode_index"])) != 118:
        raise AssertionError("locked dense panel must be 3,718 unique queries in 118 source episodes")
    cache_manifest = json.loads(DENSE_CACHE_MANIFEST.read_text(encoding="utf-8"))
    cache_audit = json.loads((EQ_ROOT / "qa/D3_EQ_DENSE_CACHE_REUSE_AUDIT.json").read_text(encoding="utf-8"))
    if cache_manifest.get("status") != "PASS" or cache_manifest.get("rows_processed") != 3718 or cache_audit.get("status") != "PASS":
        raise AssertionError("fixed dense feature cache is not accepted for reuse")
    out = PRED_ROOT / f"{args.run_id}.npz"
    sidecar_path = PRED_ROOT / f"{args.run_id}.json"
    ids_hash = sha256_array(ids)
    if args.resume and out.is_file() and sidecar_path.is_file():
        previous = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if previous.get("status") == "PASS" and previous.get("dense_ids_sha256") == ids_hash and previous.get("checkpoint_sha256") == sha256_file(checkpoint):
            print(json.dumps({"status": "PASS_REUSED", "run_id": args.run_id, "rows": 3718}, sort_keys=True))
            return 0
        raise ValueError("existing dense prediction bundle does not match checkpoint/IDs")
    if out.exists() or sidecar_path.exists():
        raise FileExistsError("prior/partial dense prediction bundle exists; preserve it and use --resume after inspection")

    video_norm_path = BENCH_ROOT / "d3/cache/D3_VIDEO_NORMALIZER.json"
    target_norm_path = V3_ROOT / "data/V3_NORMALIZERS.json"
    video_norm = json.loads(video_norm_path.read_text(encoding="utf-8"))
    target_norm = json.loads(target_norm_path.read_text(encoding="utf-8"))
    mean = np.asarray(video_norm["mean"], dtype=np.float32)
    std = np.asarray(video_norm["safe_std"], dtype=np.float32)
    sigma = torch.tensor(np.asarray(target_norm["target_sigma"], dtype=np.float32), device=device).view(1, 1, 28)
    state = torch.tensor(data["history_standardized"], device=device)
    mask = torch.tensor(data["history_mask"], device=device)
    relative = torch.tensor(data["history_relative_delta_standardized"], device=device)
    anchor = torch.tensor(data["anchor_command_native"], device=device)
    store = GridStore(cache_manifest)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = D3TimeSpacePredictor(horizon=horizon, seed=seed, video_mode=config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    predictions = np.empty((3718, horizon, 28), dtype=np.float32)
    for start in range(0, 3718, args.batch_size):
        end = min(3718, start + args.batch_size)
        dynamic, static = store.get(np.arange(start, end, dtype=np.int64))
        dynamic = ((dynamic - mean[None, None, None, :]) / std[None, None, None, :]).astype(np.float32)
        static = ((static - mean[None, None, None, :]) / std[None, None, None, :]).astype(np.float32)
        with torch.inference_mode():
            residual = model(state[start:end], mask[start:end], relative[start:end], torch.tensor(dynamic, device=device), torch.tensor(static, device=device), anchor[start:end])
            predictions[start:end] = (anchor[start:end, None, :] + residual * sigma).cpu().numpy().astype(np.float32)
    if predictions.shape != (3718, horizon, 28) or not np.isfinite(predictions).all():
        raise AssertionError("dense D3-EQ predictions are non-finite or wrong shape")

    PRED_ROOT.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(f".{out.name}.{os.getpid()}.part")
    with temp.open("wb") as handle:
        np.savez_compressed(
            handle,
            dense_id=ids,
            episode_index=data["episode_index"],
            original_task_index=data["original_task_index"],
            anchor_frame=data["anchor_frame"],
            anchor_timestamp=data["anchor_timestamp"],
            predictions_native=predictions,
            horizon_steps=np.asarray(horizon),
            config=np.asarray(config),
            seed=np.asarray(seed),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, out)
    sidecar = {
        "status": "PASS",
        "protocol_id": lock["protocol_id"],
        "run_id": args.run_id,
        "config": config,
        "horizon_steps": horizon,
        "seed": seed,
        "rows": len(ids),
        "source_episode_groups": int(len(np.unique(data["episode_index"]))),
        "dense_ids_sha256": ids_hash,
        "checkpoint_sha256": sha256_file(checkpoint),
        "prediction_sha256": sha256_file(out),
        "dense_cache_manifest_sha256": sha256_file(DENSE_CACHE_MANIFEST),
        "dense_inputs_sha256": sha256_file(DENSE_INPUTS),
        "video_normalizer_sha256": sha256_file(video_norm_path),
        "target_normalizer_sha256": sha256_file(target_norm_path),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "dev_targets_loaded": False,
        "final_test_access": 0,
    }
    atomic_json(sidecar_path, sidecar)
    print(json.dumps({"status": "PASS", "run_id": args.run_id, "rows": len(ids), "groups": sidecar["source_episode_groups"], "peak_reserved_bytes": sidecar["peak_reserved_bytes"], "dev_targets_loaded": False, "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
