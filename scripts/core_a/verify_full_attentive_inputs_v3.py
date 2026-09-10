"""Verify matched v3 dynamic/static token-cache inputs without final-test access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ee6008.config import load_config
from ee6008.data.clips import decode_prefix, preprocess_vjepa
from ee6008.full_attentive import repeat_first_frame
from ee6008.full_attentive_data import alignment_report, load_v3_population


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def raw_repeat_check(config_path: Path, sample_count: int = 10) -> dict[str, Any]:
    config = load_config(config_path)
    population = load_v3_population(config, full=True)
    video_table = pd.read_parquet(
        config.manifest_dir / "splits.parquet",
        columns=["episode_index", "video_path"],
        filters=[("split", "in", ["train", "validation"])],
    )
    rows = population.rows.head(sample_count).merge(
        video_table, on="episode_index", how="left", validate="one_to_one"
    )
    records = []
    first_hashes = []
    for row in rows.itertuples(index=False):
        sample = decode_prefix(
            config.g1_root / str(row.video_path),
            duration_seconds=config.observation_seconds,
            sample_fps=config.sample_fps,
            expected_source_fps=config.source_fps,
        )
        clip = preprocess_vjepa(sample.frames, config.input_resolution)
        repeated = repeat_first_frame(clip)
        first_hash = hashlib.sha256(sample.frames[0].tobytes()).hexdigest()
        first_hashes.append(first_hash)
        records.append(
            {
                "episode_index": int(row.episode_index),
                "decoder": sample.decoder,
                "decoded_frames": int(sample.frames.shape[0]),
                "repeated_clip_shape": list(repeated.shape),
                "all_32_equal_after_repeat": bool(
                    np.array_equal(
                        repeated[:, :1].expand_as(repeated).numpy(), repeated.numpy()
                    )
                ),
                "raw_first_frame_sha256": first_hash,
            }
        )
    return {
        "sample_count": len(records),
        "all_samples_pass": all(item["all_32_equal_after_repeat"] for item in records),
        "unique_raw_first_frame_hashes": len(set(first_hashes)),
        "samples": records,
    }


def cache_provenance_check(
    config_path: Path, dynamic_run: Path, static_run: Path
) -> dict[str, Any]:
    config = load_config(config_path)
    expected = {
        "dataset_revision": config.dataset_revision,
        "source_commit": config.source_commit,
        "checkpoint": str(config.checkpoint),
        "checkpoint_key": config.checkpoint_key,
        "input_resolution": config.input_resolution,
        "frame_count": config.frame_count,
    }
    result: dict[str, Any] = {}
    for name, run in (("dynamic", dynamic_run), ("static", static_run)):
        resolved_path = run / "resolved_config.json"
        provenance_path = run / "provenance.json"
        if not resolved_path.is_file() or not provenance_path.is_file():
            raise FileNotFoundError(f"cache provenance is incomplete: {run}")
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        for key in (
            "dataset_revision",
            "source_commit",
            "checkpoint",
            "checkpoint_key",
            "input_resolution",
        ):
            if str(resolved.get(key)) != str(expected[key]):
                raise ValueError(f"{name} resolved config mismatch for {key}")
        if name == "static" and (
            resolved.get("input_mode") != "repeated_first_frame"
            or resolved.get("pooled_only") is True
        ):
            raise ValueError(
                "static cache provenance is not repeated-first-frame token mode"
            )
        allowed_policies = {"not selected", "train/validation only"}
        if name == "dynamic":
            allowed_policies.add("label-blind frozen feature extraction only")
        if provenance.get("final_test_policy") not in allowed_policies:
            raise ValueError(f"{name} provenance does not deny final-test selection")
        result[name] = {
            "resolved_config_sha256": hashlib.sha256(
                resolved_path.read_bytes()
            ).hexdigest(),
            "provenance_sha256": hashlib.sha256(
                provenance_path.read_bytes()
            ).hexdigest(),
            "dataset_revision": resolved.get("dataset_revision"),
            "source_commit": resolved.get("source_commit"),
            "checkpoint_key": resolved.get("checkpoint_key"),
            "input_mode": resolved.get("input_mode", "normal/implicit"),
            "pooled_only": resolved.get("pooled_only", False),
            "final_test_policy": provenance.get("final_test_policy"),
            "historical_final_test_entries_present": "final_test"
            in provenance.get("selected_splits", []),
            "runner_final_test_selected": False,
        }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dynamic-run", type=Path, required=True)
    parser.add_argument("--static-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing to overwrite audit output: {args.output}")
    config = load_config(args.config)
    report = alignment_report(
        config, args.dynamic_run.resolve(), args.static_run.resolve()
    )
    report["cache_provenance"] = cache_provenance_check(
        args.config, args.dynamic_run.resolve(), args.static_run.resolve()
    )
    report["raw_repeat_check"] = raw_repeat_check(args.config)
    if not report["raw_repeat_check"]["all_samples_pass"]:
        raise RuntimeError("repeated-first-frame contract failed")
    report["status"] = "pass"
    report["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "FULL_INPUT_PREFLIGHT.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
