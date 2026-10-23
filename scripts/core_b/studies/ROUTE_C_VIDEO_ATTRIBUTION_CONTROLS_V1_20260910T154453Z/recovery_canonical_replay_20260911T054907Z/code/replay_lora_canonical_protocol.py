#!/usr/bin/env python3
"""Use the original B3 raw-LoRA make_forward for a bounded identity probe."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--b23-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, default=376)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    protocol_path = args.b23_root / "protocol/run_b3_raw_lora.py"
    protocol = load_module(protocol_path, "canonical_b3_raw_lora_protocol")
    route_registry = args.route_root / "ROUTE_C_CHECKPOINT_REGISTRY.csv"
    registry = next(row for row in csv.DictReader(route_registry.open(newline="", encoding="utf-8")) if row["model"] == "B3-FUS-D-L")
    with np.load(args.b23_root / "inputs/B2_INPUTS_0125.npz", allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    dev = data["split_code"] == 1
    ids = np.asarray(data["episode_index"][dev], dtype=np.int64)
    state = np.asarray(data["state_action_history"][dev], dtype=np.float32)
    index = {int(value): position for position, value in enumerate(ids.tolist())}
    selected = index[args.episode_id]
    start = (selected // args.batch_size) * args.batch_size
    context_indices = list(range(start, min(start + args.batch_size, len(ids))))

    sys.path.insert(0, str((args.project_root / "repo/ee6008-humanoid-vjepa/src").resolve()))
    sys.path.insert(0, str((args.project_root / "repo/third_party/vjepa2").resolve()))
    from ee6008.data import clips as clips_module
    from ee6008.data.clips import decode_prefix, preprocess_vjepa
    from ee6008.lora import LoRAConfig, inject_top_block_lora
    from ee6008.models import load_vjepa2_1_base

    device = torch.device("cuda:0")
    clips_module._DECORD_USABLE = False
    encoder, _info = load_vjepa2_1_base(
        source_root=args.project_root / "repo/third_party/vjepa2",
        checkpoint=Path(registry["base_checkpoint_path"]),
        device=device,
        num_frames=32,
        resolution=384,
    )
    inserted = inject_top_block_lora(encoder, config=LoRAConfig())
    head = protocol.PooledRegression(768 + 568)
    model = protocol.B3Model(encoder, head, True).to(device)
    payload = torch.load(Path(registry["checkpoint_path"]), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    norm = json.loads(Path(registry["feature_normalization_path"]).read_text(encoding="utf-8"))["dynamic_pooled"]
    feature_mean = torch.tensor(norm["mean"], dtype=torch.float32, device=device)
    feature_std = torch.tensor(norm["safe_std"], dtype=torch.float32, device=device)
    manifest_path = Path(registry["data_manifest"])
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = {int(row["episode_index"]): row for row in csv.DictReader(handle)}
    data_root = args.project_root / "data_raw/HumanoidEveryday/G1/USC-PSI-Lab-Humanoid-Everyday-G1__6b6b599d"
    clips = []
    for index_value in context_indices:
        episode = int(ids[index_value])
        sample = decode_prefix(data_root / manifest[episode]["video_path"], duration_seconds=4.0, sample_fps=8, expected_source_fps=30.0)
        clips.append(preprocess_vjepa(sample.frames, resolution=384))
    clip_batch = torch.stack(clips).to(device)
    with torch.inference_mode(), protocol.autocast_context(device):
        output = protocol.make_forward(model, clip_batch, torch.tensor(state[context_indices], device=device), None, train=False, feature_mean=feature_mean, feature_std=feature_std).float().cpu().numpy()
    canonical = np.load(Path(registry["canonical_prediction_path"]), allow_pickle=False)
    canonical_index = {int(value): position for position, value in enumerate(canonical["episode_index"].tolist())}
    local = context_indices.index(selected)
    delta = output[local] - canonical["predictions_standardized"][canonical_index[args.episode_id]]
    result = {
        "status": "pass" if float(np.max(np.abs(delta))) <= 1e-5 + 1e-5 * max(1.0, float(np.max(np.abs(canonical["predictions_standardized"])))) else "fail",
        "episode_id": args.episode_id,
        "context_episode_ids": [int(ids[index_value]) for index_value in context_indices],
        "batch_size": args.batch_size,
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "inserted_lora_modules": inserted,
        "checkpoint_sha256": registry["checkpoint_sha256"],
        "canonical_prediction_sha256": registry["canonical_prediction_sha256"],
        "precision": "CUDA BF16 autocast",
        "final_test_accessed": 0,
        "optimizer_steps": 0,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
