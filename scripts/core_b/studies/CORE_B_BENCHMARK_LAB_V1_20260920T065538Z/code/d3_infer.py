"""GPU inference for the accepted D3 formal 36-cell matrix."""

from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
import torch

from common import BENCH_ROOT, REPO_ROOT, atomic_write_json, read_json
from d3_model import D3TimeSpacePredictor


V3 = REPO_ROOT / "reports/core_b/CORE_B_V3_MULTI_ANCHOR_TEMPORAL_V1_20260919T103147Z"


class GridStore:
    def __init__(self, manifest: dict[str, object]) -> None:
        self.shards = sorted(manifest["shards"], key=lambda item: int(item["start_row"]))
        self.starts = [int(item["start_row"]) for item in self.shards]

    def get(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dynamic = np.empty((len(indices), 16, 4, 768), dtype=np.float32)
        static = np.empty_like(dynamic)
        grouped: dict[int, list[tuple[int, int]]] = {}
        for output_index, source_index in enumerate(indices.astype(int)):
            shard_index = bisect.bisect_right(self.starts, int(source_index)) - 1
            shard = self.shards[shard_index]
            grouped.setdefault(shard_index, []).append((output_index, int(source_index) - int(shard["start_row"])))
        for shard_index, pairs in grouped.items():
            values = np.load(self.shards[shard_index]["array"], mmap_mode="r")
            for output_index, local in pairs:
                dynamic[output_index] = values[local, 0]
                static[output_index] = values[local, 1]
        return dynamic, static


def formal_runs() -> list[tuple[str, Path, int, int, str]]:
    found: dict[tuple[str, int, int], tuple[str, Path, int, int, str]] = {}
    for marker in sorted((BENCH_ROOT / "d3/runs").glob("D3-*-FORMAL*/RUN_COMPLETE.json")):
        data = read_json(marker)
        if data.get("status") != "PASS":
            continue
        key = (str(data["config"]), int(data["horizon"]), int(data["seed"]))
        checkpoint = marker.parent / "checkpoint_last.pt"
        if not checkpoint.exists():
            continue
        # Prefer a RETRY1 run when both a stale/base and retry artifact exist.
        previous = found.get(key)
        if previous is None or "RETRY" in marker.parent.name:
            found[key] = (marker.parent.name, checkpoint, key[1], key[2], key[0])
    return sorted(found.values(), key=lambda item: (item[2], item[3], item[4]))


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("D3 inference requires approved CUDA")
    device = torch.device("cuda:0")
    with np.load(V3 / "data/V3_MAIN_INPUTS.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    dev_indices = np.flatnonzero(data["split_code"] == 1).astype(np.int64)
    cache_manifest = read_json(BENCH_ROOT / "d3/cache/D3_CACHE_MANIFEST.json")
    video_norm = read_json(BENCH_ROOT / "d3/cache/D3_VIDEO_NORMALIZER.json")
    normalizers = read_json(V3 / "data/V3_NORMALIZERS.json")
    store = GridStore(cache_manifest)
    mean = np.asarray(video_norm["mean"], dtype=np.float32)
    std = np.asarray(video_norm["safe_std"], dtype=np.float32)
    sigma = torch.tensor(np.asarray(normalizers["target_sigma"], dtype=np.float32), device=device).view(1, 1, 28)
    state = torch.tensor(data["history_standardized"][dev_indices], device=device)
    mask = torch.tensor(data["history_mask"][dev_indices], device=device)
    relative = torch.tensor(data["history_relative_delta_standardized"][dev_indices], device=device)
    anchor = torch.tensor(data["anchor_command_native"][dev_indices], device=device)
    model_runs = formal_runs()
    if len(model_runs) != 36:
        raise RuntimeError(f"expected 36 accepted D3 formal runs, found {len(model_runs)}")
    output_root = BENCH_ROOT / "d3/predictions"
    output_root.mkdir(parents=True, exist_ok=True)
    for run_name, checkpoint, horizon, seed, config in model_runs:
        out = output_root / f"{run_name}.npz"
        if out.exists():
            continue
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = D3TimeSpacePredictor(horizon=horizon, seed=seed, video_mode=config).to(device)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        predictions = np.empty((len(dev_indices), horizon, 28), dtype=np.float32)
        for start in range(0, len(dev_indices), 64):
            end = min(len(dev_indices), start + 64)
            dynamic, static = store.get(dev_indices[start:end])
            dynamic = ((dynamic - mean[None, None, None, :]) / std[None, None, None, :]).astype(np.float32)
            static = ((static - mean[None, None, None, :]) / std[None, None, None, :]).astype(np.float32)
            with torch.inference_mode():
                residual = model(state[start:end], mask[start:end], relative[start:end], torch.tensor(dynamic, device=device), torch.tensor(static, device=device), anchor[start:end])
                predictions[start:end] = (anchor[start:end, None, :] + residual * sigma).cpu().numpy().astype(np.float32)
        part = out.with_suffix(".part")
        with part.open("wb") as handle:
            np.savez_compressed(handle, anchor_id=data["anchor_id"][dev_indices], episode_index=data["episode_index"][dev_indices], original_task_index=data["original_task_index"][dev_indices], anchor_frame=data["anchor_frame"][dev_indices], anchor_timestamp=data["anchor_timestamp"][dev_indices], predictions_native=predictions, horizon_steps=np.asarray(horizon), config=np.asarray(config), seed=np.asarray(seed))
        part.replace(out)
        atomic_write_json(out.with_suffix(".json"), {"status": "PASS", "run_id": run_name, "config": config, "horizon_steps": horizon, "seed": seed, "rows": len(predictions), "checkpoint": str(checkpoint), "final_test_access": 0})
    atomic_write_json(BENCH_ROOT / "d3/D3_PREDICTION_ACCEPTANCE.json", {"status": "PASS", "accepted": len(list(output_root.glob("D3-*.npz"))) // 1, "expected": 36, "primary_dev_rows": len(dev_indices), "final_test_access": 0})
    print(json.dumps({"status": "PASS", "formal_runs": len(model_runs), "prediction_files": len(list(output_root.glob("D3-*.npz"))), "dev_rows": len(dev_indices)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
