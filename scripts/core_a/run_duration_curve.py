"""Extract pooled V-JEPA 2.1-B train/validation features for duration ablations."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ee6008.cache import FeatureCache
from ee6008.config import load_config
from ee6008.data.clips import ShortEpisodeError, decode_prefix, preprocess_vjepa
from ee6008.data.selection import benchmark_primary_mask
from ee6008.models import encode_clip, load_vjepa2_1_base


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_rows(config: Any) -> pd.DataFrame:
    split_table = pd.read_parquet(config.manifest_dir / "splits.parquet")
    rows = split_table[
        benchmark_primary_mask(split_table)
        & split_table["split"].isin(("train", "validation"))
    ].sort_values("episode_index", kind="mergesort")
    if rows.empty or set(rows["split"].unique()) != {"train", "validation"}:
        raise RuntimeError("duration curve requires train and validation rows")
    if rows["split"].eq("final_test").any():
        raise RuntimeError("final_test row reached duration curve")
    return rows


def extract_duration(
    config: Any,
    rows: pd.DataFrame,
    seconds: float,
    *,
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    frame_count = round(seconds * config.sample_fps)
    run_dir = output_root / f"{seconds:g}s"
    run_dir.mkdir(parents=True, exist_ok=False)
    cache_config = {
        **config.to_dict(),
        "duration_seconds": seconds,
        "frame_count": frame_count,
        "model": "vjepa2_1_b",
        "pooled_only": True,
        "split_filter": ["train", "validation"],
        "final_test_access": "none",
    }
    cache = FeatureCache(run_dir / "vjepa2_1_b" / "entries", cache_config)
    model, model_info = load_vjepa2_1_base(
        source_root=config.vjepa_source_root,
        checkpoint=config.checkpoint,
        device=device,
        num_frames=frame_count,
        resolution=config.input_resolution,
    )
    results: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    started = time.monotonic()
    for position, row in enumerate(rows.itertuples(index=False), 1):
        episode_index = int(row.episode_index)
        video_path = config.g1_root / str(row.video_path)
        try:
            sample = decode_prefix(
                video_path,
                duration_seconds=seconds,
                sample_fps=config.sample_fps,
                expected_source_fps=config.source_fps,
            )
        except ShortEpisodeError as exc:
            skipped.append({"episode_index": episode_index, "reason": str(exc)})
            continue
        clip = preprocess_vjepa(sample.frames, config.input_resolution)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            _tokens, pooled = encode_clip(model, clip.to(device, non_blocking=True))
        metadata = {
            "split": str(row.split),
            "video_path": str(row.video_path),
            "frame_indices": sample.frame_indices.tolist(),
            "timestamps_seconds": sample.timestamps_seconds.tolist(),
            "source_fps_observed": sample.source_fps,
            "decoder": sample.decoder,
            "duration_seconds": seconds,
        }
        status = cache.write(
            episode_index=episode_index,
            pooled=pooled.float(),
            tokens=torch.empty((0,), dtype=torch.float16),
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
                "token_shape": [],
                "feature_bytes": entry_path.stat().st_size,
            }
        )
        results.append({"episode_index": episode_index, "status": status})
        if position == 1 or position == len(rows) or position % 100 == 0:
            print(
                f"DURATION_PROGRESS {seconds:g}s {position}/{len(rows)}",
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
        "duration_seconds": seconds,
        "frame_count": frame_count,
        "model": "vjepa2_1_b",
        "checkpoint": str(config.checkpoint),
        "checkpoint_sha256": sha256_file(config.checkpoint),
        "config_path": str(config),
        "dataset_revision": config.dataset_revision,
        "source_commit": config.source_commit,
        "selected_splits": ["train", "validation"],
        "selected_rows": len(rows),
        "manifest_rows": len(manifest_rows),
        "train_count": int(manifest["split"].eq("train").sum()),
        "validation_count": int(manifest["split"].eq("validation").sum()),
        "skipped_short_count": len(skipped),
        "skipped_short_episode_indices": [item["episode_index"] for item in skipped],
        "model_contract": model_info,
        "feature_manifest_sha256": sha256_file(run_dir / "feature_manifest.parquet"),
        "device": str(device),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "new_or_existing": results,
        "final_test_access": "none",
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--durations", type=float, nargs="+", default=[1.0, 2.0, 8.0])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    config = load_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")
    if device.type == "cpu" and not args.allow_cpu:
        raise PermissionError("CPU duration extraction requires explicit --allow-cpu")
    rows = select_rows(config)
    args.output_root.mkdir(parents=True, exist_ok=False)
    reports = []
    for seconds in args.durations:
        if seconds <= 0 or seconds == 4.0:
            raise ValueError("duration must be positive and 4.0s must use the existing cache")
        reports.append(
            extract_duration(
                config,
                rows,
                seconds,
                output_root=args.output_root,
                device=device,
            )
        )
    output = {
        "status": "pass",
        "config_path": str(args.config),
        "durations": reports,
        "four_second_reference": "existing full-vjepa-b-opencv-20260826 cache; not regenerated",
        "final_test_access": "none",
    }
    (args.output_root / "duration_curve_summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
