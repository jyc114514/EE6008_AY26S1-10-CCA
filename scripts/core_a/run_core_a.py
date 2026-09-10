"""Run deterministic Core-A feature extraction with atomic resumable cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ee6008.baselines import encode_r3d18, load_r3d18
from ee6008.cache import FeatureCache
from ee6008.config import load_config
from ee6008.data.clips import decode_prefix, preprocess_vjepa
from ee6008.data.selection import benchmark_primary_mask
from ee6008.models import encode_clip, load_vjepa2_1_base


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:  # provenance remains readable on minimal PATHs
        return f"unavailable:{type(exc).__name__}"


def resolve_task_ids(selection_path: Path | None, task_set: str) -> list[int] | None:
    if task_set == "all":
        return None
    if selection_path is None:
        raise ValueError("--selection is required for smoke/pilot task sets")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    key = f"{task_set}_tasks"
    if key not in selection:
        raise KeyError(f"selection manifest has no {key}")
    return [int(value) for value in selection[key]]


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument(
        "--task-set", choices=("smoke", "pilot", "all"), default="smoke"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "final_test"),
        default=["train", "validation"],
    )
    parser.add_argument(
        "--model", choices=("vjepa2_1_b", "r3d18"), default="vjepa2_1_b"
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--episode-index",
        type=int,
        nargs="+",
        help="restrict extraction to an explicit episode-index list",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--repeated-first-frame", action="store_true")
    parser.add_argument(
        "--repeat-all-frames",
        action="store_true",
        help="replace every sampled frame after frame zero with frame zero",
    )
    parser.add_argument(
        "--pooled-only",
        action="store_true",
        help="store pooled features without the large token tensor",
    )
    parser.add_argument(
        "--r3d-weights",
        choices=("none", "kinetics400_v1"),
        default="none",
        help="R3D-18 weight contract; ignored for V-JEPA",
    )
    parser.add_argument(
        "--allow-final-test-features",
        action="store_true",
        help="explicit future unlock; final-test metrics remain prohibited",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.repeat_all_frames and args.model != "vjepa2_1_b":
        raise ValueError("--repeat-all-frames is only defined for V-JEPA 2.1-B")
    if args.repeat_all_frames and args.repeated_first_frame:
        raise ValueError("choose either --repeat-all-frames or --repeated-first-frame")
    if "final_test" in args.splits and not args.allow_final_test_features:
        raise PermissionError(
            "final_test feature extraction is locked; use the separately "
            "approved --allow-final-test-features unlock"
        )
    device = choose_device(args.device)
    if device.type == "cpu" and not args.allow_cpu:
        raise RuntimeError(
            "CPU extraction is disabled by default; use --allow-cpu for a bounded diagnostic"
        )
    split_table = pd.read_parquet(config.manifest_dir / "splits.parquet")
    task_ids = resolve_task_ids(args.selection, args.task_set)
    rows = split_table[split_table["split"].isin(args.splits)].copy()
    if any(split != "final_test" for split in args.splits):
        rows = rows[benchmark_primary_mask(rows)]
    if task_ids is not None:
        rows = rows[rows["original_task_index"].isin(task_ids)]
    if args.episode_index is not None:
        selected_episode_indices = set(args.episode_index)
        rows = rows[rows["episode_index"].isin(selected_episode_indices)]
    rows = rows[rows["validation_status"] == "valid"]
    if args.limit is not None:
        rows = rows.head(args.limit)
    if rows.empty:
        raise RuntimeError("no episodes selected")
    if "final_test" in args.splits:
        final_test_note = (
            "explicitly unlocked label-free frozen feature extraction only; "
            "no label metrics"
        )
    else:
        final_test_note = "not selected"

    run_dir = config.feature_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    input_mode = "repeated_first_frame" if args.repeat_all_frames else "normal"
    cache_config = {
        **config.to_dict(),
        "model": args.model,
        "input_mode": input_mode,
        "pooled_only": args.pooled_only,
        "r3d_weights": args.r3d_weights if args.model == "r3d18" else None,
    }
    cache = FeatureCache(
        run_dir / args.model / "entries", cache_config
    )
    write_json(
        run_dir / "resolved_config.json",
        {
            **config.to_dict(),
            "model": args.model,
            "splits": args.splits,
            "task_set": args.task_set,
            "episode_index": args.episode_index,
            "input_mode": input_mode,
            "pooled_only": args.pooled_only,
            "r3d_weights": args.r3d_weights if args.model == "r3d18" else None,
        },
    )
    provenance_checkpoint = (
        str(config.checkpoint)
        if args.model == "vjepa2_1_b"
        else "torchvision://R3D_18_Weights.KINETICS400_V1"
    )
    write_json(
        run_dir / "provenance.json",
        {
            "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "dataset_revision": config.dataset_revision,
            "source_commit": config.source_commit,
            "source_git_revision_observed": git_revision(config.vjepa_source_root),
            "checkpoint": provenance_checkpoint,
            "checkpoint_sha256": (
                sha256_file(config.checkpoint) if args.model == "vjepa2_1_b" else None
            ),
            "selected_rows": len(rows),
            "selected_splits": args.splits,
            "episode_index": args.episode_index,
            "final_test_policy": final_test_note,
            "input_mode": input_mode,
            "pooled_only": args.pooled_only,
            "r3d_weights": args.r3d_weights if args.model == "r3d18" else None,
        },
    )
    (run_dir / "dataset_revision.txt").write_text(
        config.dataset_revision + "\n", encoding="utf-8"
    )
    (run_dir / "source_commits.json").write_text(
        json.dumps({"vjepa2": config.source_commit}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.model == "vjepa2_1_b":
        model, model_info = load_vjepa2_1_base(
            source_root=config.vjepa_source_root,
            checkpoint=config.checkpoint,
            device=device,
            num_frames=config.frame_count,
            resolution=config.input_resolution,
        )
    else:
        model = load_r3d18(device, weights=args.r3d_weights)
        model_info = {
            "model": "torchvision.r3d_18",
            "weights": args.r3d_weights,
            "weights_url": "https://download.pytorch.org/models/r3d_18-b3b3357e.pth"
            if args.r3d_weights == "kinetics400_v1"
            else None,
            "feature_dim": 512,
        }
    model_info.update({"input_mode": input_mode, "pooled_only": args.pooled_only})
    write_json(run_dir / "model_contract.json", model_info)

    results: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    started = time.monotonic()
    for position, row in enumerate(rows.itertuples(index=False), 1):
        episode_index = int(row.episode_index)
        if cache.has_valid(episode_index):
            results.append({"episode_index": episode_index, "status": "existing"})
            continue
        video_path = config.g1_root / str(row.video_path)
        sample = decode_prefix(
            video_path,
            duration_seconds=config.observation_seconds,
            sample_fps=config.sample_fps,
            expected_source_fps=config.source_fps,
        )
        clip = preprocess_vjepa(sample.frames, config.input_resolution)
        if args.repeat_all_frames:
            clip[:, 1:] = clip[:, :1]
        clip_device = clip.to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            if args.model == "vjepa2_1_b":
                tokens, pooled = encode_clip(model, clip_device)
            else:
                pooled = encode_r3d18(model, clip_device)
                tokens = pooled.unsqueeze(0)
        stored_tokens = (
            torch.empty((0,), dtype=torch.float16)
            if args.pooled_only and args.model == "vjepa2_1_b"
            else tokens.half() if args.model == "vjepa2_1_b" else tokens.float()
        )
        metadata = {
            "split": str(row.split),
            "video_path": str(row.video_path),
            "frame_indices": sample.frame_indices.tolist(),
            "timestamps_seconds": sample.timestamps_seconds.tolist(),
            "source_fps_observed": sample.source_fps,
            "decoder": sample.decoder,
            "clip_shape_cthw": list(clip.shape),
            "encoder_token_shape": list(tokens.shape),
            "stored_token_shape": list(stored_tokens.shape),
            "pooled_shape": list(pooled.shape),
            "input_mode": input_mode,
        }
        if str(row.split) != "final_test":
            metadata.update(
                {
                    "original_task_index": int(row.original_task_index),
                    "contiguous_label_index": int(row.contiguous_label_index),
                    "classification_inclusion": str(row.classification_inclusion),
                    "task_name_normalized": str(row.task_name_normalized),
                    "category_normalized": str(row.category_normalized),
                }
            )
        status = cache.write(
            episode_index=episode_index,
            pooled=pooled.float(),
            tokens=stored_tokens,
            metadata=metadata,
        )
        results.append({"episode_index": episode_index, "status": status})
        if args.repeated_first_frame and position <= 4 and args.model == "vjepa2_1_b":
            repeated = clip.clone()
            repeated[:, 1:] = repeated[:, :1]
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                _, repeated_pooled = encode_clip(
                    model, repeated.to(device, non_blocking=True)
                )
            cosine = torch.nn.functional.cosine_similarity(
                pooled.float(), repeated_pooled.float(), dim=0
            ).item()
            diagnostics.append(
                {
                    "episode_index": episode_index,
                    "pooled_cosine_original_vs_repeated_first": cosine,
                    "original_norm": float(pooled.float().norm().item()),
                    "repeated_norm": float(repeated_pooled.float().norm().item()),
                }
            )
        if position % 10 == 0 or position == len(rows):
            print(f"CORE_A_PROGRESS {position}/{len(rows)}", flush=True)

    manifest_rows: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        path = cache.path_for(int(row.episode_index))
        if not path.exists() or not cache.has_valid(int(row.episode_index)):
            continue
        record = cache.load(int(row.episode_index))
        manifest_row = {
            "episode_index": int(row.episode_index),
            "split": str(row.split),
            "feature_path": str(path),
            "feature_bytes": path.stat().st_size,
            "feature_sha256": sha256_file(path),
            "pooled_shape": json.dumps(list(record["pooled"].shape)),
            "token_shape": json.dumps(list(record["tokens"].shape)),
        }
        if str(row.split) != "final_test":
            manifest_row.update(
                {
                    "original_task_index": int(row.original_task_index),
                    "contiguous_label_index": int(row.contiguous_label_index),
                    "classification_inclusion": str(row.classification_inclusion),
                }
            )
        manifest_rows.append(manifest_row)
    manifest_path = run_dir / "feature_manifest.parquet"
    manifest_tmp = Path(f"{manifest_path}.part")
    pd.DataFrame(manifest_rows).sort_values("episode_index").to_parquet(
        manifest_tmp, index=False
    )
    os.replace(manifest_tmp, manifest_path)
    write_json(
        run_dir / "run_summary.json",
        {
            "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": args.model,
            "input_mode": input_mode,
            "pooled_only": args.pooled_only,
            "r3d_weights": args.r3d_weights if args.model == "r3d18" else None,
            "device": str(device),
            "selected_rows": len(rows),
            "manifest_rows": len(manifest_rows),
            "new_or_existing": results,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "diagnostics": diagnostics,
            "final_test_policy": final_test_note,
        },
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "rows": len(manifest_rows),
                "device": str(device),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
