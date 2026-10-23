#!/usr/bin/env python3
"""Replay an accepted pooled feature through one locked B3 frozen head."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn


class PooledRegression(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.GELU(), nn.Linear(256, 30 * 28))

    def forward(self, value: Tensor) -> Tensor:
        return self.net(value).reshape(-1, 30, 28)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_registry(path: Path, model: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["model"] == model:
                return row
    raise KeyError(f"model not found in checkpoint registry: {model}")


def compare_by_id(
    *,
    ids: np.ndarray,
    candidate: np.ndarray,
    expected_ids: np.ndarray,
    expected: np.ndarray,
    selected_ids: set[int] | None,
) -> dict[str, object]:
    expected_index = {int(value): index for index, value in enumerate(expected_ids.tolist())}
    selected_indices = [index for index, value in enumerate(ids.tolist()) if selected_ids is None or int(value) in selected_ids]
    if not selected_indices:
        raise ValueError("no requested episode IDs are present in the development population")
    diffs: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for index in selected_indices:
        episode = int(ids[index])
        if episode not in expected_index:
            raise ValueError(f"canonical prediction is missing episode {episode}")
        delta = candidate[index] - expected[expected_index[episode]]
        diffs.append(np.abs(delta))
        rows.append({
            "episode_id": episode,
            "max_abs": float(np.max(np.abs(delta))),
            "mean_abs": float(np.mean(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        })
    stacked = np.concatenate([item.reshape(-1) for item in diffs])
    max_location = int(np.argmax(stacked))
    return {
        "selected_N": len(selected_indices),
        "selected_episode_ids": [int(ids[index]) for index in selected_indices],
        "max_abs": float(np.max(stacked)),
        "mean_abs": float(np.mean(stacked)),
        "rmse": float(np.sqrt(np.mean(np.square(stacked)))),
        "first_max_abs_flat_index": max_location,
        "per_episode": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("B3-FUS-D-F", "B3-V-D-F"), required=True)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--b23-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, action="append")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    registry = load_registry(args.route_root / "ROUTE_C_CHECKPOINT_REGISTRY.csv", args.model)
    with np.load(args.b23_root / "inputs/B2_INPUTS_0125.npz", allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    with np.load(Path(registry["canonical_prediction_path"]), allow_pickle=False) as archive:
        canonical = {key: archive[key].copy() for key in archive.files}

    dev = data["split_code"] == 1
    ids = np.asarray(data["episode_index"][dev], dtype=np.int64)
    pooled = np.asarray(data["dynamic_pooled"][dev], dtype=np.float32)
    state = np.asarray(data["state_action_history"][dev], dtype=np.float32)
    if pooled.shape != (len(ids), 768) or state.shape != (len(ids), 568):
        raise ValueError(f"unexpected feature/state shape: {pooled.shape} {state.shape}")
    if not np.array_equal(ids, np.asarray(canonical["episode_index"], dtype=np.int64)):
        raise ValueError("canonical IDs are not in the same order; ID join is required")

    branch = "fusion" if args.model.startswith("B3-FUS") else "video_only"
    values = np.concatenate([pooled, state], axis=1) if branch == "fusion" else pooled
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for canonical precision replay but is unavailable")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    head = PooledRegression(values.shape[1]).to(device)
    payload = torch.load(Path(registry["checkpoint_path"]), map_location="cpu", weights_only=False)
    state_dict = payload.get("head_state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint has no head/model state dict")
    head.load_state_dict(state_dict, strict=True)
    head.eval()
    prediction_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), args.batch_size):
            value_tensor = torch.from_numpy(values[start : start + args.batch_size]).to(device)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    part = head(value_tensor).float().cpu().numpy()
            else:
                part = head(value_tensor).float().cpu().numpy()
            prediction_parts.append(part)
    prediction = np.concatenate(prediction_parts, axis=0)

    comparison = compare_by_id(
        ids=ids,
        candidate=prediction,
        expected_ids=np.asarray(canonical["episode_index"], dtype=np.int64),
        expected=np.asarray(canonical["predictions_standardized"], dtype=np.float32),
        selected_ids=set(args.episode_id) if args.episode_id else None,
    )
    result = {
        "status": "pass" if comparison["max_abs"] <= 1e-5 else "fail",
        "model": args.model,
        "branch": branch,
        "source_input_archive": str((args.b23_root / "inputs/B2_INPUTS_0125.npz").resolve()),
        "source_input_archive_sha256": sha256(args.b23_root / "inputs/B2_INPUTS_0125.npz"),
        "checkpoint_path": registry["checkpoint_path"],
        "checkpoint_sha256": registry["checkpoint_sha256"],
        "canonical_prediction_path": registry["canonical_prediction_path"],
        "canonical_prediction_sha256": registry["canonical_prediction_sha256"],
        "feature_space": "accepted train-only standardized dynamic_pooled plus state_action_history",
        "device": str(device),
        "precision": "CUDA BF16 autocast" if device.type == "cuda" else "FP32",
        "batch_size": args.batch_size,
        "prediction_atol": 1e-5,
        "prediction_rtol": 1e-5,
        "comparison": comparison,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
