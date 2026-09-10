"""Repair exact cross-split video duplicates with deterministic group assignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_groups(rows: pd.DataFrame, root: Path, column: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total = len(rows)
    for position, row in enumerate(rows.itertuples(index=False), 1):
        path = root / str(getattr(row, column))
        if not path.exists():
            raise FileNotFoundError(path)
        groups[sha256_file(path)].append(
            {
                "episode_index": int(row.episode_index),
                "split": str(row.split),
                "original_task_index": int(row.original_task_index),
            }
        )
        if position % 500 == 0 or position == total:
            print(f"LEAKAGE_HASH_PROGRESS {column} {position}/{total}", flush=True)
    return dict(groups)


def cross_split_groups(groups: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {
        digest: sorted(items, key=lambda item: item["episode_index"])
        for digest, items in groups.items()
        if len(items) > 1 and len({item["split"] for item in items}) > 1
    }


def counts(rows: pd.DataFrame, column: str = "split") -> dict[str, int]:
    return {str(key): int(value) for key, value in rows[column].value_counts().sort_index().items()}


def primary_counts(rows: pd.DataFrame) -> dict[str, int]:
    return counts(rows[rows["classification_inclusion"].eq("primary")])


def task_split_counts(rows: pd.DataFrame) -> dict[str, dict[str, int]]:
    primary = rows[rows["classification_inclusion"].eq("primary")]
    result: dict[str, dict[str, int]] = {}
    for task, frame in primary.groupby("original_task_index"):
        result[str(int(task))] = counts(frame)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v2-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.v2_dir.exists():
        raise FileExistsError("refusing to overwrite split-repair outputs")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(config["g1_root"])
    source_dir = args.source_manifest
    rows = pd.read_parquet(source_dir / "splits.parquet").sort_values(
        "episode_index", kind="mergesort"
    ).reset_index(drop=True)
    if rows["episode_index"].duplicated().any():
        raise AssertionError("source split contains duplicate episode indices")

    video_groups = find_groups(rows, root, "video_path")
    parquet_groups = find_groups(rows, root, "parquet_path")
    cross_video = cross_split_groups(video_groups)
    cross_parquet = cross_split_groups(parquet_groups)
    repaired = rows.copy()
    repaired["video_sha256_fingerprint"] = repaired["episode_index"].map(
        {
            item["episode_index"]: digest
            for digest, items in video_groups.items()
            for item in items
        }
    )
    moves: list[dict[str, Any]] = []
    for digest, items in sorted(cross_video.items()):
        target = items[0]["split"]
        for item in items:
            if item["split"] != target:
                repaired.loc[
                    repaired["episode_index"].eq(item["episode_index"]), "split"
                ] = target
                moves.append(
                    {
                        "episode_index": item["episode_index"],
                        "from_split": item["split"],
                        "to_split": target,
                        "video_sha256": digest,
                        "original_task_index": item["original_task_index"],
                    }
                )

    repaired = repaired.sort_values("episode_index", kind="mergesort").reset_index(drop=True)
    repaired_split_by_episode = dict(
        zip(
            repaired["episode_index"].astype(int),
            repaired["split"].astype(str),
            strict=True,
        )
    )
    repaired_groups = {
        digest: [
            {**item, "split": repaired_split_by_episode[item["episode_index"]]}
            for item in items
        ]
        for digest, items in video_groups.items()
    }
    repaired_cross_video = cross_split_groups(repaired_groups)
    print(
        f"LEAKAGE_REPAIR_GROUP_CHECK before={len(cross_video)} after={len(repaired_cross_video)}",
        flush=True,
    )
    if repaired_cross_video:
        raise AssertionError(f"video duplicate groups still cross splits: {len(repaired_cross_video)}")
    missing_primary_coverage = {
        task: sorted(set({"train", "validation", "final_test"}) - set(split_counts))
        for task, split_counts in task_split_counts(repaired).items()
        if set(split_counts) != {"train", "validation", "final_test"}
    }
    if missing_primary_coverage:
        raise RuntimeError(
            f"group repair removed a primary split for tasks: {missing_primary_coverage}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.v2_dir.mkdir(parents=True, exist_ok=False)
    for name in ("episodes.parquet", "tasks.csv", "label_map.json"):
        shutil.copy2(source_dir / name, args.v2_dir / name)
    repaired.to_parquet(args.v2_dir / "splits.parquet", index=False)
    source_summary = json.loads((source_dir / "split_summary.json").read_text(encoding="utf-8"))
    source_summary.update(
        {
            "protocol_version": "core_a_protocol_v2_group_safe_20260826",
            "split_policy": (
                "task-stratified episode split with exact full-video duplicate "
                "groups constrained to one split; no trusted session/scene/operator "
                "group field was present"
            ),
            "split_counts": counts(repaired),
            "primary_split_counts": primary_counts(repaired),
            "per_primary_task_split_counts": task_split_counts(repaired),
            "cross_split_video_duplicate_groups_after_repair": 0,
            "final_test_policy": (
                "label-blind feature release and cache validation only; no label metrics"
            ),
        }
    )
    (args.v2_dir / "split_summary.json").write_text(
        json.dumps(source_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    repair = {
        "repair_version": "core_a_protocol_v2_group_safe_20260826",
        "source_manifest": str(source_dir),
        "source_splits_sha256": sha256_file(source_dir / "splits.parquet"),
        "video_groups_total": int(sum(1 for items in video_groups.values() if len(items) > 1)),
        "video_groups_cross_split_before": len(cross_video),
        "video_groups_cross_split_after": len(repaired_cross_video),
        "parquet_groups_cross_split": len(cross_parquet),
        "assignment_rule": (
            "for each exact duplicate video group crossing splits, assign all "
            "members to the split of the lowest episode_index member"
        ),
        "moves": moves,
        "source_split_counts": counts(rows),
        "repaired_split_counts": counts(repaired),
        "source_primary_split_counts": primary_counts(rows),
        "repaired_primary_split_counts": primary_counts(repaired),
        "primary_coverage_missing_after_repair": missing_primary_coverage,
        "final_test_metrics_computed": False,
    }
    (args.v2_dir / "leakage_repair.json").write_text(
        json.dumps(repair, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    hash_lines = []
    for path in sorted(args.v2_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            hash_lines.append(f"{sha256_file(path)}  {path.name}")
    (args.v2_dir / "SHA256SUMS").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")

    report: dict[str, Any] = {
        "status": "pass"
        if not cross_parquet and not repaired_cross_video and not missing_primary_coverage
        else "review",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "account": os.environ.get("USER", "unknown"),
        "dataset_revision": config["dataset_revision"],
        "source_manifest": str(source_dir),
        "repaired_manifest": str(args.v2_dir),
        "source_split_counts": counts(rows),
        "repaired_split_counts": counts(repaired),
        "source_primary_split_counts": primary_counts(rows),
        "repaired_primary_split_counts": primary_counts(repaired),
        "video_groups_total": repair["video_groups_total"],
        "video_groups_cross_split_before": len(cross_video),
        "video_groups_cross_split_after": len(repaired_cross_video),
        "parquet_groups_cross_split": len(cross_parquet),
        "move_count": len(moves),
        "moves": moves,
        "missing_primary_coverage_after_repair": missing_primary_coverage,
        "source_splits_sha256": repair["source_splits_sha256"],
        "repaired_splits_sha256": sha256_file(args.v2_dir / "splits.parquet"),
        "repaired_bundle_files": {
            path.name: sha256_file(path)
            for path in sorted(args.v2_dir.iterdir())
            if path.is_file()
        },
        "final_test_metrics_computed": False,
        "interpretation": (
            "The original task-stratified split had exact full-video duplicate "
            "groups crossing split boundaries. The repaired bundle assigns each "
            "such group to one deterministic split and preserves one train, "
            "validation and final_test row for every primary class. This is "
            "safe against exact video duplication, not proof of scene/session "
            "independence because no trusted group field was available."
        ),
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Split and leakage audit",
        "",
        f"Status: {report['status']}",
        f"Dataset revision: {report['dataset_revision']}",
        "",
        "## Exact duplicate result",
        "",
        f"- duplicate video groups: {report['video_groups_total']}",
        f"- cross-split video duplicate groups before repair: {report['video_groups_cross_split_before']}",
        f"- cross-split video duplicate groups after repair: {report['video_groups_cross_split_after']}",
        f"- cross-split Parquet duplicate groups: {report['parquet_groups_cross_split']}",
        f"- moved rows: {report['move_count']}",
        "",
        f"- source split counts: {report['source_split_counts']}",
        f"- repaired split counts: {report['repaired_split_counts']}",
        f"- source primary split counts: {report['source_primary_split_counts']}",
        f"- repaired primary split counts: {report['repaired_primary_split_counts']}",
        "",
        (
            "The repair rule assigns every exact duplicate video group to the split "
            "of its lowest episode index. Primary class coverage remains present in "
            "all three splits. This does not establish session/scene independence."
        ),
        "",
        "## Versioned repaired manifest",
        "",
        f"- path: {args.v2_dir}",
        f"- splits SHA-256: {report['repaired_splits_sha256']}",
        f"- missing primary coverage after repair: {report['missing_primary_coverage_after_repair']}",
        "",
        "No final-test labels were aggregated and no final-test metric was computed.",
    ]
    (args.output_dir / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
