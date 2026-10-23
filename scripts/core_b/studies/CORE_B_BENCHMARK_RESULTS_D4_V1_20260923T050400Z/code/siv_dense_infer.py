"""Inference-only V3_STATE extension over the fixed D2 dense development panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pathlibs import BENCH_ROOT, NEXT_ROOT, STATE_ROOT, V3_ROOT

sys.path.insert(0, str(STATE_ROOT / "code"))
from state_video_model import StateVideoPredictor, state_hash  # noqa: E402

DENSE_INPUTS = BENCH_ROOT / "d2/D2_DENSE_INPUTS.npz"
OUT_ROOT = NEXT_ROOT / "v3_state_dense"
CELLS = tuple(f"{state}-{video}" for state in ("FULL", "OBS", "ANCHOR") for video in ("N", "D", "S"))


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


def dense_markers() -> list[dict]:
    selected = {}
    for marker_path in sorted((STATE_ROOT / "predictions").glob("*/PREDICTION_EXPORT_COMPLETE.json")):
        marker = json.loads(marker_path.read_text())
        if marker.get("status") != "PASS":
            continue
        key = (str(marker["cell"]), int(marker["seed"]), int(marker["horizon_steps"]))
        if key in selected:
            raise ValueError(f"duplicate accepted V3_STATE dense source cell: {key}")
        selected[key] = marker
    expected = {(cell, seed, horizon) for cell in CELLS for seed in (20260953, 20260954, 20260955) for horizon in (30, 60)}
    if set(selected) != expected:
        raise ValueError(f"accepted V3_STATE study cells={len(selected)}, expected 54")
    return sorted(selected.values(), key=lambda r: (int(r["horizon_steps"]), int(r["seed"]), str(r["cell"])))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("V3_STATE dense inference requires an approved watcher child")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    with np.load(DENSE_INPUTS, allow_pickle=False) as bundle:
        data = {key: bundle[key].copy() for key in ("dense_id", "episode_index", "original_task_index", "anchor_frame", "anchor_timestamp", "history_standardized", "history_mask", "history_relative_delta_standardized", "anchor_command_native")}
    n_rows = len(data["dense_id"])
    feature_path = BENCH_ROOT / "d2/D2_DENSE_TEMPORAL_FEATURES.npy"
    feature_manifest = json.loads((BENCH_ROOT / "d2/D2_DENSE_FEATURE_MANIFEST.json").read_text())
    if feature_manifest.get("status") != "PASS" or int(feature_manifest.get("rows", -1)) != n_rows or feature_manifest.get("shape") != [n_rows, 2, 16, 768]:
        raise ValueError("V3_STATE dense reuse feature manifest is not accepted for the fixed panel")
    features = np.load(feature_path, mmap_mode="r")
    if n_rows != 3718 or len(set(data["dense_id"].tolist())) != n_rows or features.shape != (n_rows, 2, 16, 768) or not np.isfinite(features).all():
        raise ValueError("V3_STATE dense feature/input shape or ID contract mismatch")
    data["features"] = features
    normalizer = json.loads((V3_ROOT / "data/V3_NORMALIZERS.json").read_text())
    video_norm = json.loads((V3_ROOT / "data/V3_VIDEO_NORMALIZER.json").read_text())
    target_sigma = torch.tensor(np.asarray(normalizer["target_sigma"], dtype=np.float32), device=device).view(1, 1, 28)
    video_mean = np.asarray(video_norm["mean"], dtype=np.float32)
    video_std = np.asarray(video_norm["safe_std"], dtype=np.float32)
    state = torch.tensor(data["history_standardized"], device=device)
    history_mask = torch.tensor(data["history_mask"], device=device)
    relative_delta = torch.tensor(data["history_relative_delta_standardized"], device=device)
    anchor = torch.tensor(data["anchor_command_native"], device=device)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    accepted = []
    run_started = time.time()
    for marker in dense_markers():
        cell = str(marker["cell"])
        horizon, seed = int(marker["horizon_steps"]), int(marker["seed"])
        run_id = str(marker["run_id"])
        checkpoint = Path(marker["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        out_dir = OUT_ROOT / "predictions" / run_id
        out = out_dir / "dense.npz"
        sidecar = out_dir / "PREDICTION_EXPORT_COMPLETE.json"
        dense_ids_hash = hashlib.sha256(data["dense_id"].astype("U96").tobytes()).hexdigest()
        if args.resume and out.is_file() and sidecar.is_file():
            old = json.loads(sidecar.read_text())
            if old.get("status") == "PASS" and old.get("dense_ids_sha256") == dense_ids_hash:
                accepted.append(old)
                continue
            raise ValueError(f"resume V3_STATE output is not an accepted same-panel bundle: {out}")
        if args.resume and out.is_file() and not sidecar.is_file():
            with np.load(out, allow_pickle=False) as recovered:
                ids = recovered["dense_id"].astype("U96")
                pred = recovered["predictions_native"]
                recovered_key = (str(recovered["cell"].item()), int(recovered["seed"].item()), int(recovered["horizon_steps"]))
            if set(ids.tolist()) != set(data["dense_id"].tolist()) or pred.shape != (n_rows, horizon, 28) or recovered_key != (cell, seed, horizon) or not np.isfinite(pred).all():
                raise ValueError(f"orphaned V3_STATE dense NPZ fails recovery acceptance: {out}")
            recovery_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            recovery_model = StateVideoPredictor(horizon=horizon, seed=seed)
            recovery_model.load_state_dict(recovery_payload["model_state_dict"], strict=True)
            recovered_state_hash = state_hash(recovery_model)
            record = {"status": "PASS", "protocol_id": "CORE_B_V3_STATE_INFORMATION_VIDEO_DENSE_EXTENSION_V1", "run_id": run_id, "cell": cell, "state_level": cell.split("-")[0], "video_level": cell.split("-")[1], "seed": seed, "horizon_steps": horizon, "primary_dense_shape": list(pred.shape), "dense_ids_sha256": dense_ids_hash, "checkpoint": str(checkpoint), "checkpoint_sha256": marker.get("checkpoint_sha256", "UNKNOWN"), "prediction_path": str(out), "prediction_sha256": sha256(out), "state_hash_before": recovered_state_hash, "state_hash_after": recovered_state_hash, "state_unchanged": True, "recovered_after_sidecar_interruption": True, "encoder_updated": False, "new_encoder_forward_for_extraction": False, "final_test_access": 0}
            atomic_json(sidecar, record)
            accepted.append(record)
            continue
        if out.exists() or sidecar.exists():
            raise FileExistsError(f"partial V3_STATE output exists without --resume: {out}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = StateVideoPredictor(horizon=horizon, seed=seed).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        before = state_hash(model)
        state_level, video_level = cell.split("-")
        torch.cuda.reset_peak_memory_stats(device)
        predictions = np.empty((n_rows, horizon, 28), dtype=np.float32)
        video_index = 0 if video_level == "D" else 1
        for start in range(0, n_rows, args.batch_size):
            end = min(n_rows, start + args.batch_size)
            video_values = data["features"][start:end, video_index].astype(np.float32)
            video = (torch.from_numpy((video_values - video_mean[None, None, :]) / video_std[None, None, :]).to(device))
            if video_level == "N":
                video = torch.zeros_like(video)
            with torch.inference_mode():
                residual = model.forward_locked(state[start:end], history_mask[start:end], relative_delta[start:end], video, anchor[start:end], state_level=state_level, video_level=video_level)
                predictions[start:end] = (anchor[start:end, None, :] + residual * target_sigma).cpu().numpy().astype(np.float32)
        if not np.isfinite(predictions).all():
            raise ValueError(f"non-finite V3_STATE dense predictions: {run_id}")
        after = state_hash(model)
        if before != after:
            raise ValueError(f"V3_STATE model weights changed during inference: {run_id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        part = out.with_suffix(".part")
        with part.open("wb") as handle:
            np.savez_compressed(handle, dense_id=data["dense_id"], episode_index=data["episode_index"], original_task_index=data["original_task_index"], anchor_frame=data["anchor_frame"], anchor_timestamp=data["anchor_timestamp"], predictions_native=predictions, horizon_steps=np.asarray(horizon), cell=np.asarray(cell), state_level=np.asarray(state_level), video_level=np.asarray(video_level), seed=np.asarray(seed))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(part, out)
        record = {"status": "PASS", "protocol_id": "CORE_B_V3_STATE_INFORMATION_VIDEO_DENSE_EXTENSION_V1", "run_id": run_id, "cell": cell, "state_level": state_level, "video_level": video_level, "seed": seed, "horizon_steps": horizon, "primary_dense_shape": list(predictions.shape), "dense_ids_sha256": dense_ids_hash, "checkpoint": str(checkpoint), "checkpoint_sha256": marker.get("checkpoint_sha256", "UNKNOWN"), "prediction_path": str(out), "prediction_sha256": sha256(out), "state_hash_before": before, "state_hash_after": after, "state_unchanged": True, "encoder_updated": False, "new_encoder_forward_for_extraction": False, "final_test_access": 0}
        atomic_json(sidecar, record)
        accepted.append(record)
        print(json.dumps({"status": "bundle_complete", "run_id": run_id, "rows": n_rows, "elapsed_total_seconds": time.time() - run_started, "peak_allocated_mib": int(torch.cuda.max_memory_allocated(device) / (1024 * 1024)), "peak_reserved_mib": int(torch.cuda.max_memory_reserved(device) / (1024 * 1024))}, sort_keys=True), flush=True)
        del model, payload
        torch.cuda.empty_cache()
    summary = {"status": "PASS" if len(accepted) == 54 else "PARTIAL", "accepted_bundles": len(accepted), "expected_bundles": 54, "dense_rows_per_bundle": n_rows, "elapsed_wall_seconds": time.time() - run_started, "peak_allocated_mib": int(torch.cuda.max_memory_allocated(device) / (1024 * 1024)), "peak_reserved_mib": int(torch.cuda.max_memory_reserved(device) / (1024 * 1024)), "final_test_access": 0}
    atomic_json(OUT_ROOT / "V3_STATE_DENSE_PREDICTION_ACCEPTANCE.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if len(accepted) == 54 else 2


if __name__ == "__main__":
    raise SystemExit(main())
