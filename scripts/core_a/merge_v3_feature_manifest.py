"""Build a v3 train/validation feature manifest from disjoint cache manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ee6008.data.selection import benchmark_primary_mask


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    split_table = pd.read_parquet(args.split_manifest)
    target_mask = benchmark_primary_mask(split_table) & split_table["split"].isin(
        ("train", "validation")
    )
    target = split_table.loc[
        target_mask,
        [
            "episode_index",
            "split",
            "original_task_index",
            "contiguous_label_index",
            "classification_inclusion",
            "v3_benchmark_inclusion",
        ],
    ].copy()
    if target["episode_index"].duplicated().any():
        raise RuntimeError("v3 split manifest contains duplicate episode rows")

    source_frames: list[pd.DataFrame] = []
    source_info: list[dict[str, str | int]] = []
    for source_path in args.source_manifest:
        frame = pd.read_parquet(source_path)
        required = {"episode_index", "feature_path"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"{source_path} missing columns: {sorted(missing)}")
        if frame["episode_index"].duplicated().any():
            raise RuntimeError(f"{source_path} contains duplicate episode rows")
        source_frames.append(frame)
        source_info.append(
            {
                "path": str(source_path),
                "sha256": sha256_file(source_path),
                "rows": len(frame),
            }
        )

    features = pd.concat(source_frames, ignore_index=True)
    if features["episode_index"].duplicated().any():
        overlap = features.loc[
            features["episode_index"].duplicated(keep=False), "episode_index"
        ].astype(int).unique()
        raise RuntimeError(f"source manifests overlap episode indices: {sorted(overlap)}")
    feature_columns = [
        column
        for column in (
            "episode_index",
            "feature_path",
            "feature_bytes",
            "feature_sha256",
            "pooled_shape",
            "token_shape",
        )
        if column in features.columns
    ]
    merged = target.merge(
        features[feature_columns],
        on="episode_index",
        how="left",
        validate="one_to_one",
    ).sort_values("episode_index", kind="mergesort")
    if len(merged) != len(target) or merged["feature_path"].isna().any():
        missing = merged.loc[merged["feature_path"].isna(), "episode_index"].astype(int).tolist()
        raise RuntimeError(f"combined feature manifest is missing episodes: {missing}")
    missing_paths = [
        str(path)
        for path in merged["feature_path"]
        if not Path(str(path)).is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(missing_paths[:10])

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_manifest = args.output_dir / "feature_manifest.parquet"
    merged.to_parquet(output_manifest, index=False)
    report = {
        "status": "pass",
        "model": args.model,
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "source_manifests": source_info,
        "rows": len(merged),
        "split_counts": merged["split"].value_counts().sort_index().to_dict(),
        "task_count": int(merged["original_task_index"].nunique()),
        "feature_manifest": str(output_manifest),
        "feature_manifest_sha256": sha256_file(output_manifest),
        "final_test_access": "none",
    }
    (args.output_dir / "combination.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
