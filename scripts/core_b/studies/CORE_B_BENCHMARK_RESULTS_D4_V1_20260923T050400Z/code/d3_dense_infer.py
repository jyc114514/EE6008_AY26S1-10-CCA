"""Inference-only D3 prediction export over the fixed 3,718-query dense panel."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from pathlibs import BENCH_ROOT, NEXT_ROOT, V3_ROOT

import sys

sys.path.insert(0, str(BENCH_ROOT / "code"))
from d3_model import D3TimeSpacePredictor  # noqa: E402

DENSE_INPUTS = BENCH_ROOT / "d2/D2_DENSE_INPUTS.npz"
EXT_ROOT = NEXT_ROOT / "d3_extension"
PRED_ROOT = EXT_ROOT / "predictions"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(part, path)


def accepted_formals() -> list[dict]:
    selected = {}
    for sidecar in sorted((BENCH_ROOT / "d3/predictions").glob("D3-*.json")):
        marker = json.loads(sidecar.read_text())
        if marker.get("status") != "PASS":
            continue
        key = (str(marker["config"]), int(marker["horizon_steps"]), int(marker["seed"]))
        previous = selected.get(key)
        if previous is None or ("RETRY" in str(marker["run_id"]) and "RETRY" not in str(previous["run_id"])):
            selected[key] = marker
    if len(selected) != 36:
        raise ValueError(f"accepted D3 model slots={len(selected)}, expected 36")
    return sorted(selected.values(), key=lambda r: (int(r["horizon_steps"]), int(r["seed"]), str(r["config"])))


class GridStore:
    def __init__(self, manifest: dict):
        self.shards = sorted(manifest["shards"], key=lambda row: int(row["start_row"]))
        self.starts = [int(row["start_row"]) for row in self.shards]

    def get(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dynamic = np.empty((len(indices), 16, 4, 768), dtype=np.float32)
        static = np.empty_like(dynamic)
        by_shard = {}
        for out_index, source_index in enumerate(indices.astype(int)):
            shard_index = bisect.bisect_right(self.starts, int(source_index)) - 1
            shard = self.shards[shard_index]
            if not int(shard["start_row"]) <= source_index < int(shard["end_row"]):
                raise ValueError(f"dense feature cache lacks row {source_index}")
            by_shard.setdefault(shard_index, []).append((out_index, int(source_index) - int(shard["start_row"])))
        for shard_index, positions in by_shard.items():
            values = np.load(self.shards[shard_index]["array"], mmap_mode="r")
            for out_index, local in positions:
                dynamic[out_index] = values[local, 0]
                static[out_index] = values[local, 1]
        return dynamic, static


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("D3 dense inference requires an approved watcher child")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    with np.load(DENSE_INPUTS, allow_pickle=False) as z:
        data = {key: z[key].copy() for key in ("dense_id", "episode_index", "original_task_index", "anchor_frame", "anchor_timestamp", "history_standardized", "history_mask", "history_relative_delta_standardized", "anchor_command_native")}
    n_rows = len(data["dense_id"])
    if n_rows != 3718 or len(set(data["dense_id"].tolist())) != n_rows:
        raise ValueError(f"fixed D2 dense panel contract expected 3718 unique IDs; got {n_rows}")
    cache_manifest = json.loads((EXT_ROOT / "cache/D3_DENSE_EXTRACTION_RUN_MANIFEST.json").read_text())
    if cache_manifest.get("status") != "PASS" or int(cache_manifest.get("rows_processed", -1)) != n_rows:
        raise ValueError("D3 dense spatial cache is incomplete or not accepted")
    video_norm = json.loads((BENCH_ROOT / "d3/cache/D3_VIDEO_NORMALIZER.json").read_text())
    target_norm = json.loads((V3_ROOT / "data/V3_NORMALIZERS.json").read_text())
    mean = np.asarray(video_norm["mean"], dtype=np.float32)
    std = np.asarray(video_norm["safe_std"], dtype=np.float32)
    sigma = torch.tensor(np.asarray(target_norm["target_sigma"], dtype=np.float32), device=device).view(1, 1, 28)
    state = torch.tensor(data["history_standardized"], device=device)
    mask = torch.tensor(data["history_mask"], device=device)
    relative = torch.tensor(data["history_relative_delta_standardized"], device=device)
    anchor = torch.tensor(data["anchor_command_native"], device=device)
    store = GridStore(cache_manifest)
    PRED_ROOT.mkdir(parents=True, exist_ok=True)
    accepted_rows = []
    run_started = time.time()
    for marker in accepted_formals():
        config, horizon, seed = str(marker["config"]), int(marker["horizon_steps"]), int(marker["seed"])
        run_id = str(marker["run_id"])
        checkpoint = Path(marker["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        out = PRED_ROOT / f"{run_id}.npz"
        sidecar = PRED_ROOT / f"{run_id}.json"
        if args.resume and out.is_file() and sidecar.is_file():
            previous = json.loads(sidecar.read_text())
            if previous.get("status") == "PASS" and previous.get("dense_ids_sha256") == hashlib.sha256(data["dense_id"].astype("U96").tobytes()).hexdigest():
                accepted_rows.append(previous)
                continue
            raise ValueError(f"resume output exists but is not a matching accepted bundle: {out}")
        if args.resume and out.is_file() and not sidecar.is_file():
            with np.load(out, allow_pickle=False) as recovered:
                recovered_ids = recovered["dense_id"].astype("U96")
                recovered_pred = recovered["predictions_native"]
                recovered_key = (str(recovered["config"].item()), int(recovered["seed"].item()), int(recovered["horizon_steps"]))
            if set(recovered_ids.tolist()) != set(data["dense_id"].astype("U96").tolist()) or recovered_pred.shape != (n_rows, horizon, 28) or recovered_key != (config, seed, horizon) or not np.isfinite(recovered_pred).all():
                raise ValueError(f"orphaned D3 dense NPZ fails recovery acceptance: {out}")
            record = {"status": "PASS", "protocol_id": "CORE_B_D3_DENSE_EXTENSION_V1", "run_id": run_id, "config": config, "seed": seed, "horizon_steps": horizon, "rows": n_rows, "dense_ids_sha256": hashlib.sha256(data["dense_id"].astype("U96").tobytes()).hexdigest(), "checkpoint": str(checkpoint), "prediction_path": str(out), "prediction_sha256": sha256(out), "checkpoint_selection": "accepted D3 epoch-40 checkpoint; retry1 preferred for duplicate formal slot", "recovered_after_sidecar_interruption": True, "final_test_access": 0}
            atomic_json(sidecar, record)
            accepted_rows.append(record)
            continue
        if out.exists() or sidecar.exists():
            raise FileExistsError(f"partial or prior D3 dense bundle exists without --resume: {out}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = D3TimeSpacePredictor(horizon=horizon, seed=seed, video_mode=config).to(device)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        torch.cuda.reset_peak_memory_stats(device)
        predictions = np.empty((n_rows, horizon, 28), dtype=np.float32)
        for start in range(0, n_rows, args.batch_size):
            end = min(n_rows, start + args.batch_size)
            indices = np.arange(start, end, dtype=np.int64)
            dynamic, static = store.get(indices)
            dynamic = ((dynamic - mean[None, None, None, :]) / std[None, None, None, :]).astype(np.float32)
            static = ((static - mean[None, None, None, :]) / std[None, None, None, :]).astype(np.float32)
            with torch.inference_mode():
                residual = model(state[start:end], mask[start:end], relative[start:end], torch.tensor(dynamic, device=device), torch.tensor(static, device=device), anchor[start:end])
                predictions[start:end] = (anchor[start:end, None, :] + residual * sigma).cpu().numpy().astype(np.float32)
        if not np.isfinite(predictions).all():
            raise ValueError(f"non-finite dense output: {run_id}")
        part = out.with_suffix(".part")
        with part.open("wb") as handle:
            np.savez_compressed(handle, dense_id=data["dense_id"], episode_index=data["episode_index"], original_task_index=data["original_task_index"], anchor_frame=data["anchor_frame"], anchor_timestamp=data["anchor_timestamp"], predictions_native=predictions, horizon_steps=np.asarray(horizon), config=np.asarray(config), seed=np.asarray(seed))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(part, out)
        record = {"status": "PASS", "protocol_id": "CORE_B_D3_DENSE_EXTENSION_V1", "run_id": run_id, "config": config, "seed": seed, "horizon_steps": horizon, "rows": n_rows, "dense_ids_sha256": hashlib.sha256(data["dense_id"].astype("U96").tobytes()).hexdigest(), "checkpoint": str(checkpoint), "prediction_path": str(out), "prediction_sha256": sha256(out), "checkpoint_selection": "accepted D3 epoch-40 checkpoint; retry1 preferred for duplicate formal slot", "input_population": "fixed D2 task-balanced dense development panel", "final_test_access": 0}
        atomic_json(sidecar, record)
        accepted_rows.append(record)
        print(json.dumps({"status": "bundle_complete", "run_id": run_id, "rows": n_rows, "elapsed_total_seconds": time.time() - run_started, "peak_allocated_mib": int(torch.cuda.max_memory_allocated(device) / (1024 * 1024)), "peak_reserved_mib": int(torch.cuda.max_memory_reserved(device) / (1024 * 1024))}, sort_keys=True), flush=True)
        del model, payload
        torch.cuda.empty_cache()
    summary = {"status": "PASS" if len(accepted_rows) == 36 else "PARTIAL", "accepted_bundles": len(accepted_rows), "expected_bundles": 36, "dense_rows_per_bundle": n_rows, "prediction_files": [row["prediction_path"] for row in accepted_rows], "elapsed_wall_seconds": time.time() - run_started, "peak_allocated_mib": int(torch.cuda.max_memory_allocated(device) / (1024 * 1024)), "peak_reserved_mib": int(torch.cuda.max_memory_reserved(device) / (1024 * 1024)), "final_test_access": 0}
    atomic_json(EXT_ROOT / "D3_DENSE_PREDICTION_ACCEPTANCE.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if len(accepted_rows) == 36 else 2


if __name__ == "__main__":
    raise SystemExit(main())
