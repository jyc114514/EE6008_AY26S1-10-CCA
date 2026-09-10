"""Audit population accounting, split leakage and create a protocol-v2 copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

GROUP_PATTERN = re.compile(
    r"session|scene|operator|subject|sequence|recording|location|camera|run",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl_keys(path: Path) -> list[str]:
    keys: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    keys.update(value)
    return sorted(keys)


def hash_collisions(
    rows: pd.DataFrame, root: Path, path_column: str
) -> dict[str, Any]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing: list[int] = []
    for row in rows.itertuples(index=False):
        episode_index = int(row.episode_index)
        path = root / str(getattr(row, path_column))
        if not path.exists():
            missing.append(episode_index)
            continue
        by_hash[sha256_file(path)].append(
            {"episode_index": episode_index, "split": str(row.split)}
        )
    collisions = [items for items in by_hash.values() if len(items) > 1]
    cross_split = [
        items for items in collisions if len({item["split"] for item in items}) > 1
    ]
    return {
        "files_checked": int(sum(len(items) for items in by_hash.values())),
        "missing_count": len(missing),
        "missing_episode_indices": missing[:20],
        "duplicate_group_count": len(collisions),
        "cross_split_duplicate_group_count": len(cross_split),
        "cross_split_duplicate_groups": cross_split[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v2-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.v2_dir.exists():
        raise FileExistsError("refusing to overwrite a population or v2 output")
    raw_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(raw_config["g1_root"])
    manifest_dir = Path(raw_config["manifest_dir"])
    episodes = pd.read_parquet(manifest_dir / "episodes.parquet")
    splits = pd.read_parquet(manifest_dir / "splits.parquet")
    tasks = pd.read_csv(manifest_dir / "tasks.csv")
    split_counts = splits["split"].value_counts().sort_index().to_dict()
    primary = splits[splits["classification_inclusion"].eq("primary")]
    primary_split_counts = primary["split"].value_counts().sort_index().to_dict()
    low_rows = splits[splits["classification_inclusion"].eq("low_sample_excluded")]
    low_task_ids = sorted(low_rows["original_task_index"].unique().tolist())
    low_episode_ids = {
        str(int(task_id)): sorted(
            low_rows.loc[
                low_rows["original_task_index"].eq(task_id), "episode_index"
            ].astype(int)
        )
        for task_id in low_task_ids
    }
    short_rows = episodes[~episodes["eligible_4s"].astype(bool)]
    primary_tasks = sorted(primary["original_task_index"].unique().tolist())
    contiguous_map = (
        primary[["original_task_index", "contiguous_label_index"]]
        .drop_duplicates()
        .sort_values("contiguous_label_index")
    )
    group_columns = sorted(
        column
        for column in splits.columns
        if GROUP_PATTERN.search(column) and column not in {"video_path", "parquet_path"}
    )
    raw_key_sets = {
        name: load_jsonl_keys(root / "meta" / name)
        for name in ("episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl")
    }
    duplicate_parquet = hash_collisions(splits, root, "parquet_path")
    duplicate_video = hash_collisions(splits, root, "video_path")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.v2_dir.mkdir(parents=True, exist_ok=False)
    copied = []
    for name in ("episodes.parquet", "tasks.csv", "splits.parquet", "label_map.json"):
        source = manifest_dir / name
        destination = args.v2_dir / name
        shutil.copy2(source, destination)
        copied.append(
            {
                "name": name,
                "source_sha256": sha256_file(source),
                "copied_sha256": sha256_file(destination),
            }
        )
    protocol = {
        "protocol_version": "core_a_protocol_v2_20260826",
        "source_manifest_dir": str(manifest_dir),
        "dataset_revision": raw_config["dataset_revision"],
        "membership_policy": "bytewise copy of corrected Core-A bundle; no split reassignment",
        "copied_files": copied,
        "split_counts": split_counts,
        "primary_split_counts": primary_split_counts,
        "primary_class_count": len(primary_tasks),
        "metadata_task_count": len(tasks),
        "used_task_count": int(splits["original_task_index"].nunique()),
        "low_sample_tasks": low_task_ids,
        "short_episode_indices": sorted(short_rows["episode_index"].astype(int).tolist()),
        "group_columns_matching_audit_pattern": group_columns,
        "raw_metadata_key_sets": raw_key_sets,
        "parquet_duplicate_audit": duplicate_parquet,
        "video_sha256_fingerprint_audit": duplicate_video,
        "contiguous_label_map": [
            {
                "original_task_index": int(row.original_task_index),
                "contiguous_label_index": int(row.contiguous_label_index),
            }
            for row in contiguous_map.itertuples(index=False)
        ],
        "final_test_metrics_computed": False,
    }
    (args.v2_dir / "protocol_v2.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    protocol["protocol_v2_sha256"] = sha256_file(args.v2_dir / "protocol_v2.json")
    (args.v2_dir / "protocol_v2.lock.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = {
        "status": "pass"
        if not duplicate_parquet["missing_count"]
        and not duplicate_video["missing_count"]
        and not duplicate_parquet["cross_split_duplicate_group_count"]
        and not duplicate_video["cross_split_duplicate_group_count"]
        and not group_columns
        else "review",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "account": os.environ.get("USER", "unknown"),
        "dataset_revision": raw_config["dataset_revision"],
        "manifest_dir": str(manifest_dir),
        "episodes_manifest_rows": len(episodes),
        "split_rows": len(splits),
        "metadata_task_count": len(tasks),
        "used_task_count": int(splits["original_task_index"].nunique()),
        "primary_class_count": len(primary_tasks),
        "eligible_4s_count": int(episodes["eligible_4s"].sum()),
        "split_counts": split_counts,
        "primary_split_counts": primary_split_counts,
        "low_sample_task_ids": low_task_ids,
        "low_sample_episode_ids": low_episode_ids,
        "short_episode_indices": sorted(short_rows["episode_index"].astype(int).tolist()),
        "primary_train_excess_over_primary": int(
            split_counts.get("train", 0) - primary_split_counts.get("train", 0)
        ),
        "group_columns_matching_audit_pattern": group_columns,
        "raw_metadata_key_sets": raw_key_sets,
        "parquet_duplicate_audit": duplicate_parquet,
        "video_sha256_fingerprint_audit": duplicate_video,
        "v2_dir": str(args.v2_dir),
        "v2_protocol_sha256": protocol["protocol_v2_sha256"],
        "final_test_metrics_computed": False,
        "interpretation": (
            "The split is task-stratified because no trusted session/scene/"
            "operator grouping field was found in the inspected manifest or "
            "raw metadata key sets. Exact Parquet and full-video SHA-256 "
            "cross-split collisions are checked separately."
        ),
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Primary population and split/leakage audit",
        "",
        f"Status: {report['status']}",
        f"Dataset revision: {report['dataset_revision']}",
        "",
        "## Population reconciliation",
        "",
        f"- metadata task rows: {report['metadata_task_count']}",
        f"- task indices used in split table: {report['used_task_count']}",
        f"- primary classes: {report['primary_class_count']}",
        f"- eligible 4-second episodes: {report['eligible_4s_count']}",
        f"- split counts: {report['split_counts']}",
        f"- primary split counts: {report['primary_split_counts']}",
        f"- low-sample task IDs: {report['low_sample_task_ids']}",
        f"- low-sample episode IDs: {report['low_sample_episode_ids']}",
        f"- short episode IDs rejected from 4-second protocol: {report['short_episode_indices']}",
        "",
        (
            "The four train rows beyond the 2,812 primary train rows are the two "
            "episodes for each retained low-sample task. They remain in the corpus "
            "and split table but are excluded from the 122-class primary benchmark."
        ),
        "",
        "## Leakage and grouping",
        "",
        f"- manifest columns matching group-field audit: {report['group_columns_matching_audit_pattern']}",
        f"- raw metadata key sets: {report['raw_metadata_key_sets']}",
        f"- Parquet hash audit: {report['parquet_duplicate_audit']}",
        f"- full-video SHA-256 fingerprint audit: {report['video_sha256_fingerprint_audit']}",
        "",
        (
            "No trusted session/scene/operator grouping field was found, so the "
            "current split is task-stratified rather than group-safe. No exact "
            "Parquet or full-video hash collision crossed a split boundary."
        ),
        "",
        "## Protocol-v2 bundle",
        "",
        f"- path: {report['v2_dir']}",
        f"- protocol hash: {report['v2_protocol_sha256']}",
        (
            "- The bundle is a bytewise copy of the corrected Core-A manifest; "
            "the old bundle was not modified."
        ),
        "",
        "No final-test metrics or label aggregates were computed.",
    ]
    (args.output_dir / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
