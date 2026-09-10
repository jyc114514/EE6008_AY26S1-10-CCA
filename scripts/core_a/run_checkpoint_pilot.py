"""Extract train/validation-only pilot features for local V-JEPA checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ee6008.cache import FeatureCache
from ee6008.config import load_config
from ee6008.data.clips import decode_prefix, preprocess_vjepa
from ee6008.data.selection import benchmark_primary_mask
from ee6008.models import encode_clip

MODEL_CONTRACTS: dict[str, dict[str, Any]] = {
    "vjepa2_1_l": {
        "checkpoint_key": "ema_encoder",
        "resolution": 384,
        "source_family": "app.vjepa_2_1.models.vision_transformer",
        "model_kwargs": {
            "patch_size": 16,
            "tubelet_size": 2,
            "use_sdpa": True,
            "use_silu": False,
            "wide_silu": True,
            "uniform_power": True,
            "use_rope": True,
            "img_temporal_dim_size": 1,
            "interpolate_rope": False,
        },
    },
    "vjepa2_l": {
        "checkpoint_key": "target_encoder",
        "resolution": 256,
        "source_family": "src.models.vision_transformer",
        "model_kwargs": {
            "patch_size": 16,
            "tubelet_size": 2,
            "use_sdpa": True,
            "uniform_power": True,
            "use_rope": True,
        },
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state_dict.items()
    }


def load_checkpoint_model(
    model_name: str,
    source_root: Path,
    checkpoint: Path,
    device: torch.device,
    *,
    num_frames: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    contract = MODEL_CONTRACTS[model_name]
    source_root = source_root.resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    if model_name == "vjepa2_1_l":
        from app.vjepa_2_1.models import vision_transformer as vit

        model = vit.vit_large(
            img_size=(contract["resolution"], contract["resolution"]),
            num_frames=num_frames,
            **contract["model_kwargs"],
        )
    elif model_name == "vjepa2_l":
        from src.models import vision_transformer as vit

        model = vit.vit_large(
            img_size=(contract["resolution"], contract["resolution"]),
            num_frames=num_frames,
            **contract["model_kwargs"],
        )
    else:
        raise ValueError(f"unsupported checkpoint model: {model_name}")

    checkpoint_obj = torch.load(checkpoint, map_location="cpu", weights_only=True)
    checkpoint_key = contract["checkpoint_key"]
    if checkpoint_key not in checkpoint_obj:
        raise KeyError(f"checkpoint has no {checkpoint_key} key")
    state_dict = clean_state_dict(checkpoint_obj[checkpoint_key])
    message = model.load_state_dict(state_dict, strict=True)
    if message.missing_keys or message.unexpected_keys:
        raise RuntimeError(f"checkpoint load was not exact: {message}")
    del checkpoint_obj, state_dict
    gc.collect()
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "model": model_name,
        "checkpoint_key": checkpoint_key,
        "resolution": contract["resolution"],
        "num_frames": num_frames,
        "source_family": contract["source_family"],
        "model_kwargs": contract["model_kwargs"],
        "embed_dim": int(model.embed_dim),
    }


def select_rows(config: Any, selection_path: Path, task_set: str) -> pd.DataFrame:
    if task_set != "pilot":
        raise ValueError("checkpoint comparison is intentionally limited to pilot")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    task_ids = [int(value) for value in selection["pilot_tasks"]]
    split_table = pd.read_parquet(config.manifest_dir / "splits.parquet")
    rows = split_table[
        benchmark_primary_mask(split_table)
        & split_table["original_task_index"].isin(task_ids)
        & split_table["split"].isin(("train", "validation"))
    ].sort_values("episode_index", kind="mergesort")
    if rows.empty or set(rows["split"].unique()) != {"train", "validation"}:
        raise RuntimeError("pilot checkpoint comparison requires train and validation rows")
    if rows["episode_index"].duplicated().any():
        raise RuntimeError("duplicate pilot episode row")
    if rows["split"].eq("final_test").any():
        raise RuntimeError("final_test row reached checkpoint pilot")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--task-set", choices=("pilot",), default="pilot")
    parser.add_argument("--model", choices=tuple(MODEL_CONTRACTS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    rows = select_rows(config, args.selection, args.task_set)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")
    if device.type == "cpu" and not args.allow_cpu:
        raise PermissionError("CPU checkpoint extraction requires explicit --allow-cpu")

    run_dir = config.feature_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    cache_contract = MODEL_CONTRACTS[args.model]
    cache_config = {
        **config.to_dict(),
        "run_id": args.run_id,
        "model": args.model,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_key": cache_contract["checkpoint_key"],
        "input_resolution": cache_contract["resolution"],
        "input_frames": config.frame_count,
        "task_set": args.task_set,
        "splits": ["train", "validation"],
        "final_test_access": "none",
    }
    cache = FeatureCache(run_dir / args.model / "entries", cache_config)
    model, model_info = load_checkpoint_model(
        args.model,
        config.vjepa_source_root,
        checkpoint,
        device,
        num_frames=config.frame_count,
    )
    results: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for position, row in enumerate(rows.itertuples(index=False), 1):
        episode_index = int(row.episode_index)
        video_path = config.g1_root / str(row.video_path)
        sample = decode_prefix(
            video_path,
            duration_seconds=config.observation_seconds,
            sample_fps=config.sample_fps,
            expected_source_fps=config.source_fps,
        )
        clip = preprocess_vjepa(sample.frames, cache_contract["resolution"])
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            tokens, pooled = encode_clip(
                model, clip.to(device, non_blocking=True)
            )
        metadata = {
            "split": str(row.split),
            "video_path": str(row.video_path),
            "frame_indices": sample.frame_indices.tolist(),
            "timestamps_seconds": sample.timestamps_seconds.tolist(),
            "source_fps_observed": sample.source_fps,
            "decoder": sample.decoder,
            "model": args.model,
        }
        status = cache.write(
            episode_index=episode_index,
            pooled=pooled.float(),
            tokens=tokens.half(),
            metadata=metadata,
        )
        entry_path = cache.path_for(episode_index)
        manifest_rows.append(
            {
                "episode_index": episode_index,
                "split": str(row.split),
                "original_task_index": int(row.original_task_index),
                "feature_path": str(entry_path),
                "pooled_shape": list(pooled.shape),
                "token_shape": list(tokens.shape),
                "feature_bytes": entry_path.stat().st_size,
            }
        )
        results.append({"episode_index": episode_index, "status": status})
        if position == 1 or position == len(rows) or position % 10 == 0:
            print(
                f"CHECKPOINT_PILOT_PROGRESS {args.model} {position}/{len(rows)}",
                flush=True,
            )

    manifest = pd.DataFrame(manifest_rows).sort_values("episode_index", kind="mergesort")
    manifest.to_parquet(run_dir / "feature_manifest.parquet", index=False)
    (run_dir / "dataset_revision.txt").write_text(
        config.dataset_revision + "\n", encoding="utf-8"
    )
    (run_dir / "source_commits.json").write_text(
        json.dumps({"vjepa2": config.source_commit}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "pass",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "account": os.environ.get("USER", "unknown"),
        "model": args.model,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_key": cache_contract["checkpoint_key"],
        "config_path": str(args.config),
        "config_sha256": sha256_file(args.config),
        "dataset_revision": config.dataset_revision,
        "source_commit": config.source_commit,
        "task_set": args.task_set,
        "selected_splits": ["train", "validation"],
        "selected_rows": len(rows),
        "train_count": int(rows["split"].eq("train").sum()),
        "validation_count": int(rows["split"].eq("validation").sum()),
        "model_contract": model_info,
        "cache_config_sha256": sha256_file(run_dir / args.model / "entries/cache_config.json"),
        "feature_manifest_sha256": sha256_file(run_dir / "feature_manifest.parquet"),
        "device": str(device),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "new_or_existing": results,
        "final_test_access": "none",
        "label_scope": "train and validation only; no final-test labels or metrics accessed",
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
