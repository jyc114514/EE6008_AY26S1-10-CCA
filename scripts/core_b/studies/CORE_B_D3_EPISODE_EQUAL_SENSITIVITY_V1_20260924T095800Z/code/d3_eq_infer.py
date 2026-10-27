"""Fresh-process D3-EQ inference on the locked 2,322-row primary dev panel."""

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
V3_ROOT = PROJECT_ROOT / "repo/ee6008-humanoid-vjepa/reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"
PRED_ROOT = EQ_ROOT / "d3/predictions"
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
        self.shards = sorted(manifest["shards"], key=lambda item: int(item["start_row"]))
        self.starts = [int(item["start_row"]) for item in self.shards]

    def get(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dynamic = np.empty((len(indices), 16, 4, 768), dtype=np.float32)
        static = np.empty_like(dynamic)
        grouped: dict[int, list[tuple[int, int]]] = {}
        for out_index, source_index in enumerate(indices.astype(int)):
            shard_index = bisect.bisect_right(self.starts, int(source_index)) - 1
            shard = self.shards[shard_index]
            if not int(shard["start_row"]) <= int(source_index) < int(shard["end_row"]):
                raise IndexError(f"D3 feature cache does not cover row {source_index}")
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
        raise ValueError("primary inference batch size is locked to 64")
    if not torch.cuda.is_available():
        raise RuntimeError("D3-EQ inference requires a separately approved watcher child")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    matrix = json.loads((EQ_ROOT / "protocol/D3_EQ_MATRIX_36.csv").read_text()) if False else None
    import csv
    with (EQ_ROOT / "protocol/D3_EQ_MATRIX_36.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["run_id"] == args.run_id]
    if len(selected) != 1:
        raise ValueError(f"run ID is not unique in the locked matrix: {args.run_id}")
    config = str(selected[0]["config"])
    horizon = int(selected[0]["horizon_steps"])
    seed = int(selected[0]["seed"])
    run_root = EQ_ROOT / "d3/runs" / args.run_id
    complete = json.loads((run_root / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
    checkpoint = run_root / "checkpoint_last.pt"
    if complete.get("status") != "PASS" or int(complete.get("epochs", -1)) != 40 or int(complete.get("optimizer_steps", -1)) != 11120:
        raise AssertionError("D3-EQ checkpoint is not an accepted epoch-40/11,120-step run")
    if not checkpoint.is_file() or sha256_file(checkpoint) != complete.get("checkpoint_last_sha256"):
        raise AssertionError("epoch-40 checkpoint hash mismatch")
    lock = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if complete.get("protocol_hash") != sha256_file(PROTOCOL):
        raise AssertionError("run completion protocol hash differs from the locked protocol")

    input_archive = np.load(V3_ROOT / "data/V3_MAIN_INPUTS.npz", allow_pickle=False)
    split = input_archive["split_code"].astype(np.int8, copy=True)
    dev_indices = np.flatnonzero(split == 1).astype(np.int64)
    # Target arrays are intentionally not accessed in the inference process.
    keys = (
        "anchor_id", "episode_index", "original_task_index", "anchor_frame", "anchor_timestamp",
        "history_standardized", "history_mask", "history_relative_delta_standardized", "anchor_command_native",
    )
    data = {key: input_archive[key][dev_indices].copy() for key in keys}
    ids = data["anchor_id"].astype("U64")
    if len(ids) != 2322 or len(set(ids.tolist())) != 2322:
        raise AssertionError("locked primary-dev IDs must be exactly 2,322 unique anchors")
    output_path = PRED_ROOT / f"{args.run_id}.npz"
    sidecar_path = PRED_ROOT / f"{args.run_id}.json"
    ids_hash = sha256_array(ids.astype("U96"))
    if args.resume and output_path.is_file() and sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("status") == "PASS" and sidecar.get("dev_ids_sha256") == ids_hash and sidecar.get("checkpoint_sha256") == sha256_file(checkpoint):
            print(json.dumps({"status": "PASS_REUSED", "run_id": args.run_id, "rows": 2322}, sort_keys=True))
            return 0
        raise ValueError("existing prediction bundle does not match the current checkpoint/IDs")
    if output_path.exists() or sidecar_path.exists():
        raise FileExistsError("prior or partial primary prediction bundle exists; use --resume after validating it")

    cache_manifest_path = BENCH_ROOT / "d3/cache/D3_CACHE_MANIFEST.json"
    video_norm_path = BENCH_ROOT / "d3/cache/D3_VIDEO_NORMALIZER.json"
    target_norm_path = V3_ROOT / "data/V3_NORMALIZERS.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    video_norm = json.loads(video_norm_path.read_text(encoding="utf-8"))
    target_norm = json.loads(target_norm_path.read_text(encoding="utf-8"))
    store = GridStore(cache_manifest)
    mean = np.asarray(video_norm["mean"], dtype=np.float32)
    std = np.asarray(video_norm["safe_std"], dtype=np.float32)
    sigma = torch.tensor(np.asarray(target_norm["target_sigma"], dtype=np.float32), device=device).view(1, 1, 28)
    state = torch.tensor(data["history_standardized"], device=device)
    mask = torch.tensor(data["history_mask"], device=device)
    relative = torch.tensor(data["history_relative_delta_standardized"], device=device)
    anchor = torch.tensor(data["anchor_command_native"], device=device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = D3TimeSpacePredictor(horizon=horizon, seed=seed, video_mode=config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    predictions = np.empty((len(ids), horizon, 28), dtype=np.float32)
    for start in range(0, len(ids), args.batch_size):
        end = min(len(ids), start + args.batch_size)
        dynamic, static = store.get(dev_indices[start:end])
        dynamic = ((dynamic - mean[None, None, None, :]) / std[None, None, None, :]).astype(np.float32)
        static = ((static - mean[None, None, None, :]) / std[None, None, None, :]).astype(np.float32)
        with torch.inference_mode():
            residual = model(
                state[start:end], mask[start:end], relative[start:end],
                torch.tensor(dynamic, device=device), torch.tensor(static, device=device), anchor[start:end],
            )
            predictions[start:end] = (anchor[start:end, None, :] + residual * sigma).cpu().numpy().astype(np.float32)
    if predictions.shape != (2322, horizon, 28) or not np.isfinite(predictions).all():
        raise AssertionError("primary D3-EQ prediction array shape/finite check failed")

    PRED_ROOT.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.part")
    with temp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            anchor_id=ids,
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
    os.replace(temp_path, output_path)
    sidecar = {
        "status": "PASS",
        "protocol_id": lock["protocol_id"],
        "run_id": args.run_id,
        "config": config,
        "horizon_steps": horizon,
        "seed": seed,
        "rows": len(ids),
        "dev_source_episode_groups": int(len(np.unique(data["episode_index"]))),
        "dev_ids_sha256": ids_hash,
        "checkpoint_sha256": sha256_file(checkpoint),
        "prediction_sha256": sha256_file(output_path),
        "cache_manifest_sha256": sha256_file(cache_manifest_path),
        "video_normalizer_sha256": sha256_file(video_norm_path),
        "target_normalizer_sha256": sha256_file(target_norm_path),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "inference_process": "fresh watcher child; checkpoint reloaded in this process",
        "dev_targets_loaded": False,
        "final_test_access": 0,
    }
    atomic_json(sidecar_path, sidecar)
    print(json.dumps({"status": "PASS", "run_id": args.run_id, "rows": len(ids), "peak_allocated_bytes": sidecar["peak_allocated_bytes"], "peak_reserved_bytes": sidecar["peak_reserved_bytes"], "dev_targets_loaded": False, "final_test_access": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
