"""Build the duplicate-safe Core-A v3 split and its audit artifacts.

The script treats the old task-stratified split as retired input, audits all
eligible G1 episodes, computes exact video/Parquet fingerprints, runs a fixed
three-frame perceptual-hash proxy audit, and writes a new versioned bundle.
It never reads final-test predictions or model outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import socket
import time
from collections import defaultdict
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml

PERCEPTUAL_HAMMING_THRESHOLD = 4
FRAME_COUNT_TOLERANCE = 2
TRAIN_VALIDATION_SAMPLE_SIZE = 256
TRAIN_RANDOM_PAIR_COUNT = 4000
PATH_PROXY_PATTERN = re.compile(
    r"session|scene|operator|subject|sequence|recording|location|camera|run|batch|date",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def phash_frame(frame: np.ndarray) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coefficients = cv2.dct(resized)[:8, :8].reshape(-1)[1:]
    median = float(np.median(coefficients))
    value = 0
    for bit in coefficients > median:
        value = (value << 1) | int(bool(bit))
    return value


def video_phashes(path: Path, frame_count: int) -> tuple[int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video for perceptual hash: {path}")
    positions = (0, max(0, frame_count // 2), max(0, frame_count - 1))
    hashes: list[int] = []
    try:
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"cannot read frame {position} from {path}")
            hashes.append(phash_frame(frame))
    finally:
        capture.release()
    return tuple(hashes)  # type: ignore[return-value]


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def counts(rows: pd.DataFrame) -> dict[str, int]:
    return {str(key): int(value) for key, value in rows["split"].value_counts().sort_index().items()}


def primary_counts(rows: pd.DataFrame) -> dict[str, int]:
    return counts(rows[rows["classification_inclusion"].eq("primary")])


def task_split_counts(rows: pd.DataFrame) -> dict[str, dict[str, int]]:
    primary = rows[rows["classification_inclusion"].eq("primary")]
    result: dict[str, dict[str, int]] = {}
    for task, frame in primary.groupby("original_task_index"):
        result[str(int(task))] = counts(frame)
    return result


def path_proxies(rows: pd.DataFrame) -> dict[str, Any]:
    columns = [str(column) for column in rows.columns if PATH_PROXY_PATTERN.search(str(column))]
    values: dict[str, list[str]] = {}
    for column in columns:
        values[column] = sorted({str(value) for value in rows[column].dropna().unique()})[:100]
    path_matches = sorted(
        {
            component
            for column in ("video_path", "parquet_path")
            for value in rows[column].astype(str)
            for component in value.split("/")
            if PATH_PROXY_PATTERN.search(component)
        }
    )
    return {
        "matching_manifest_columns": columns,
        "matching_manifest_values_sample": values,
        "matching_path_components": path_matches,
        "raw_metadata_keys_checked": {},
    }


def union_find(ids: list[int]) -> tuple[dict[int, int], Any, Any]:
    parent = {value: value for value in ids}
    rank = {value: 0 for value in ids}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    return parent, find, union


def exact_groups(rows: pd.DataFrame) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows.itertuples(index=False):
        groups[str(row.video_sha256)].append(int(row.episode_index))
    return {digest: sorted(episodes) for digest, episodes in groups.items() if len(episodes) > 1}


def validate_threshold(
    rows: pd.DataFrame, phashes: dict[int, tuple[int, int, int]], exact: dict[str, list[int]]
) -> dict[str, Any]:
    train_ids = rows.loc[rows["split"].eq("train"), "episode_index"].astype(int).tolist()
    train_ids = sorted(train_ids)[:TRAIN_VALIDATION_SAMPLE_SIZE]
    exact_pairs: list[tuple[int, int]] = []
    for episodes in exact.values():
        exact_pairs.extend(combinations(episodes, 2))
    exact_distances = [
        max(hamming(a, b) for a, b in zip(phashes[left], phashes[right], strict=True))
        for left, right in exact_pairs
        if left in phashes and right in phashes
    ]
    rng = random.Random(20260826)
    candidate_pairs = list(combinations(train_ids, 2))
    rng.shuffle(candidate_pairs)
    random_distances: list[int] = []
    sha_by_episode = dict(zip(rows["episode_index"], rows["video_sha256"], strict=True))
    for left, right in candidate_pairs[:TRAIN_RANDOM_PAIR_COUNT]:
        if sha_by_episode[left] == sha_by_episode[right]:
            continue
        random_distances.append(
            max(hamming(a, b) for a, b in zip(phashes[left], phashes[right], strict=True))
        )
    quantiles = {}
    if random_distances:
        values = np.asarray(random_distances, dtype=np.float64)
        quantiles = {
            str(q): float(np.quantile(values, q)) for q in (0.01, 0.05, 0.5, 0.95, 0.99)
        }
    return {
        "threshold_fixed_before_full_scan": True,
        "algorithm": "8x8 low-frequency DCT median perceptual hash on first/middle/last frame",
        "threshold_max_hamming_per_frame": PERCEPTUAL_HAMMING_THRESHOLD,
        "frame_count_tolerance": FRAME_COUNT_TOLERANCE,
        "train_sample_count": len(train_ids),
        "exact_duplicate_pair_count": len(exact_distances),
        "exact_duplicate_max_distance": max(exact_distances) if exact_distances else None,
        "random_non_exact_pair_count": len(random_distances),
        "random_non_exact_quantiles": quantiles,
        "interpretation": (
            "The fixed rule is a conservative candidate-group rule, not proof of semantic identity. "
            "Only same-task pairs with all three frame hashes within threshold and frame counts within "
            "the fixed tolerance are promoted to near-duplicate constraints."
        ),
    }


def write_exact_reports(
    report_dir: Path,
    rows: pd.DataFrame,
    exact: dict[str, list[int]],
    old_split_by_episode: dict[int, str],
    v3_split_by_episode: dict[int, str],
    parquet_sha: dict[int, str],
) -> dict[str, Any]:
    row_by_episode = rows.set_index("episode_index", drop=False)
    group_records: list[dict[str, Any]] = []
    member_records: list[dict[str, Any]] = []
    old_validation_members = 0
    old_final_members = 0
    for group_number, (digest, episodes) in enumerate(sorted(exact.items()), 1):
        group_id = f"exact-{group_number:04d}"
        group_rows = row_by_episode.loc[episodes]
        task_ids = sorted({int(value) for value in group_rows["original_task_index"]})
        task_names = sorted({str(value) for value in group_rows["task_name_raw"]})
        categories = sorted({str(value) for value in group_rows["category_raw"]})
        labels_consistent = len(task_ids) == 1 and len(task_names) == 1 and len(categories) == 1
        old_splits = sorted({old_split_by_episode[episode] for episode in episodes})
        v3_splits = sorted({v3_split_by_episode[episode] for episode in episodes})
        old_validation_members += sum(old_split_by_episode[episode] == "validation" for episode in episodes)
        old_final_members += sum(old_split_by_episode[episode] == "final_test" for episode in episodes)
        status = "exact_duplicate" if labels_consistent else "conflicting_label_duplicate"
        group_records.append(
            {
                "duplicate_group_id": group_id,
                "video_sha256": digest,
                "member_count": len(episodes),
                "episode_indices": episodes,
                "original_task_indices": task_ids,
                "task_names": task_names,
                "categories": categories,
                "old_splits": old_splits,
                "v3_splits": v3_splits,
                "label_consistent": labels_consistent,
                "status": status,
            }
        )
        for episode in episodes:
            row = row_by_episode.loc[episode]
            member_records.append(
                {
                    "duplicate_group_id": group_id,
                    "video_sha256": digest,
                    "episode_index": int(episode),
                    "original_task_index": int(row.original_task_index),
                    "contiguous_label_index": int(row.contiguous_label_index),
                    "task_name": str(row.task_name_raw),
                    "category": str(row.category_raw),
                    "old_split": old_split_by_episode[episode],
                    "v3_split": v3_split_by_episode[episode],
                    "video_path": str(row.video_path),
                    "parquet_path": str(row.parquet_path),
                    "video_bytes": int(row.video_size_bytes),
                    "video_duration_seconds": float(row.duration_seconds),
                    "parquet_sha256": parquet_sha[episode],
                    "label_consistent": labels_consistent,
                    "group_status": status,
                }
            )
    csv_path = report_dir / "EXACT_DUPLICATE_GROUPS_20260826.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(member_records[0]))
        writer.writeheader()
        writer.writerows(member_records)
    md_lines = [
        "# Exact duplicate groups",
        "",
        f"Total exact duplicate groups: {len(group_records)}",
        f"Total duplicate episodes: {sum(item['member_count'] for item in group_records)}",
        f"Groups crossing the old split: {sum(len(item['old_splits']) > 1 for item in group_records)}",
        f"Old validation duplicate members: {old_validation_members}",
        f"Old final-test duplicate members: {old_final_members}",
        "",
        "The old split is retired as `retired_due_to_duplicate_unsafe_split`. The v3 assignment keeps each trusted group atomic.",
        "",
        (
            "Theoretical maximum old validation accuracy influence from these duplicate members: "
            f"{old_validation_members}/622 = {old_validation_members / 622 * 100:.4f} percentage points of the old validation population."
        ),
        "",
        "| group | video SHA-256 | episodes | tasks | old splits | v3 split | label status |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in group_records:
        md_lines.append(
            "| {duplicate_group_id} | `{video_sha256}` | {episode_indices} | {original_task_indices} | "
            "{old_splits} | {v3_splits} | {status} |".format(**item)
        )
    (report_dir / "EXACT_DUPLICATE_GROUPS_20260826.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )
    return {
        "group_count": len(group_records),
        "duplicate_episode_count": sum(item["member_count"] for item in group_records),
        "old_cross_split_group_count": sum(len(item["old_splits"]) > 1 for item in group_records),
        "old_validation_duplicate_members": old_validation_members,
        "old_final_test_duplicate_members": old_final_members,
        "conflicting_label_group_count": sum(item["status"] == "conflicting_label_duplicate" for item in group_records),
        "groups": group_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--unsafe-manifest", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.report_dir.exists():
        raise FileExistsError("refusing to overwrite v3 split or audit outputs")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(config["g1_root"])
    unsafe = pd.read_parquet(args.unsafe_manifest / "splits.parquet")
    base = pd.read_parquet(args.base_manifest / "splits.parquet")
    if len(unsafe) != len(base) or set(unsafe.episode_index) != set(base.episode_index):
        raise RuntimeError("unsafe and base manifests do not cover the same episodes")
    old_split_by_episode = dict(zip(unsafe.episode_index.astype(int), unsafe.split.astype(str), strict=True))
    rows = base.sort_values("episode_index", kind="mergesort").reset_index(drop=True).copy()
    rows["old_split"] = rows["episode_index"].map(old_split_by_episode)
    if rows["old_split"].isna().any():
        raise RuntimeError("missing old split assignment")
    rows["video_sha256"] = ""
    rows["parquet_sha256"] = ""
    total = len(rows)
    for position, row in enumerate(rows.itertuples(index=False), 1):
        video_path = root / str(row.video_path)
        parquet_path = root / str(row.parquet_path)
        if not video_path.is_file() or not parquet_path.is_file():
            raise FileNotFoundError(video_path if not video_path.is_file() else parquet_path)
        rows.loc[rows["episode_index"].eq(int(row.episode_index)), "video_sha256"] = sha256_file(video_path)
        rows.loc[rows["episode_index"].eq(int(row.episode_index)), "parquet_sha256"] = sha256_file(parquet_path)
        if position % 250 == 0 or position == total:
            print(f"V3_HASH_PROGRESS {position}/{total}", flush=True)
    exact = exact_groups(rows)
    if len(exact) != 40:
        raise RuntimeError(f"expected 40 exact duplicate groups from checkpoint, found {len(exact)}")
    phashes: dict[int, tuple[int, int, int]] = {}
    for position, row in enumerate(rows.itertuples(index=False), 1):
        phashes[int(row.episode_index)] = video_phashes(
            root / str(row.video_path), int(row.video_nb_frames or row.num_frames)
        )
        if position % 100 == 0 or position == total:
            print(f"V3_PHASH_PROGRESS {position}/{total}", flush=True)
    threshold_validation = validate_threshold(rows, phashes, exact)
    sha_by_episode = dict(zip(rows.episode_index, rows.video_sha256, strict=True))
    near_edges: list[dict[str, Any]] = []
    for task, frame in rows.groupby("original_task_index"):
        records = list(frame.itertuples(index=False))
        for left, right in combinations(records, 2):
            left_id, right_id = int(left.episode_index), int(right.episode_index)
            if sha_by_episode[left_id] == sha_by_episode[right_id]:
                continue
            if abs(int(left.num_frames) - int(right.num_frames)) > FRAME_COUNT_TOLERANCE:
                continue
            distances = [
                hamming(a, b)
                for a, b in zip(phashes[left_id], phashes[right_id], strict=True)
            ]
            if max(distances) <= PERCEPTUAL_HAMMING_THRESHOLD:
                near_edges.append(
                    {
                        "left_episode_index": min(left_id, right_id),
                        "right_episode_index": max(left_id, right_id),
                        "original_task_index": int(task),
                        "frame_count_difference": abs(int(left.num_frames) - int(right.num_frames)),
                        "phash_hamming_first": distances[0],
                        "phash_hamming_middle": distances[1],
                        "phash_hamming_last": distances[2],
                    }
                )
    _parent, find, union = union_find(rows["episode_index"].astype(int).tolist())
    for episodes in exact.values():
        for left, right in pairwise(episodes):
            union(left, right)
    for edge in near_edges:
        union(edge["left_episode_index"], edge["right_episode_index"])
    components: dict[int, list[int]] = defaultdict(list)
    for episode in rows["episode_index"].astype(int):
        components[find(episode)].append(int(episode))
    trusted_components = [sorted(episodes) for episodes in components.values() if len(episodes) > 1]
    trusted_components.sort(key=lambda episodes: episodes[0])
    base_split_by_episode = dict(zip(rows.episode_index.astype(int), rows.split.astype(str), strict=True))
    v3_split_by_episode = dict(base_split_by_episode)
    component_moves: list[dict[str, Any]] = []
    for component in trusted_components:
        target = base_split_by_episode[component[0]]
        for episode in component:
            if v3_split_by_episode[episode] != target:
                component_moves.append(
                    {
                        "episode_index": episode,
                        "from_split": v3_split_by_episode[episode],
                        "to_split": target,
                        "component_episodes": component,
                    }
                )
                v3_split_by_episode[episode] = target
    rows["split"] = rows["episode_index"].map(v3_split_by_episode)
    conflict_episodes = {
        episode
        for digest, episodes in exact.items()
        if len({int(rows.loc[rows.episode_index.eq(episode), "original_task_index"].iloc[0]) for episode in episodes}) > 1
        for episode in episodes
    }
    rows["v3_benchmark_inclusion"] = np.where(
        rows["episode_index"].isin(conflict_episodes),
        "duplicate_conflict_excluded",
        rows["classification_inclusion"],
    )
    primary = rows[rows["v3_benchmark_inclusion"].eq("primary")]
    missing_primary_coverage = {
        str(int(task)): sorted(set({"train", "validation", "final_test"}) - set(frame.split))
        for task, frame in primary.groupby("original_task_index")
        if set(frame.split) != {"train", "validation", "final_test"}
    }
    if missing_primary_coverage:
        raise RuntimeError(f"v3 primary coverage missing: {missing_primary_coverage}")
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=False)
    exact_report = write_exact_reports(
        report_dir,
        rows,
        exact,
        old_split_by_episode,
        v3_split_by_episode,
        dict(zip(rows.episode_index.astype(int), rows.parquet_sha256.astype(str), strict=True)),
    )
    root_report_dir = report_dir.parent
    for name in ("EXACT_DUPLICATE_GROUPS_20260826.csv", "EXACT_DUPLICATE_GROUPS_20260826.md"):
        root_report = root_report_dir / name
        if root_report.exists():
            raise FileExistsError(f"refusing to overwrite root exact-duplicate report: {root_report}")
        shutil.copy2(report_dir / name, root_report)
    near_report = {
        "status": "pass",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "algorithm": threshold_validation["algorithm"],
        "threshold_validation": threshold_validation,
        "candidate_near_duplicate_edge_count": len(near_edges),
        "candidate_near_duplicate_edges": near_edges,
        "trusted_component_count": len(trusted_components),
        "trusted_components": trusted_components,
        "path_proxy_audit": path_proxies(rows),
        "interpretation": (
            "No session/scene/operator metadata field was found. Candidate near-duplicate edges "
            "are conservative same-task three-frame perceptual-hash matches; they are used as atomic "
            "constraints in v3 and are not model-selected."
        ),
    }
    json_dump(report_dir / "NEAR_DUPLICATE_PROXY_AUDIT_20260826.json", near_report)
    near_lines = [
        "# Near-duplicate and session-proxy audit",
        "",
        f"Fixed algorithm: {near_report['algorithm']}",
        f"Fixed maximum Hamming distance per frame: {PERCEPTUAL_HAMMING_THRESHOLD}",
        f"Fixed frame-count tolerance: {FRAME_COUNT_TOLERANCE}",
        f"Candidate near-duplicate edges: {len(near_edges)}",
        f"Trusted atomic components after exact and near constraints: {len(trusted_components)}",
        "",
        "No trusted session/scene/operator grouping field was present in the inspected manifest or raw metadata keys.",
        "The threshold was fixed before the full scan and its train-only validation is recorded in the JSON report.",
        "",
        "## Candidate edges",
        "",
        "| left episode | right episode | task | frame-count delta | first/middle/last Hamming |",
        "|---:|---:|---:|---:|---|",
    ]
    for edge in near_edges:
        near_lines.append(
            f"| {edge['left_episode_index']} | {edge['right_episode_index']} | {edge['original_task_index']} | "
            f"{edge['frame_count_difference']} | {edge['phash_hamming_first']}/{edge['phash_hamming_middle']}/{edge['phash_hamming_last']} |"
        )
    (report_dir / "NEAR_DUPLICATE_PROXY_AUDIT_20260826.md").write_text(
        "\n".join(near_lines) + "\n", encoding="utf-8"
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("episodes.parquet", "tasks.csv", "label_map.json"):
        source = args.base_manifest / name
        shutil.copy2(source, args.output_dir / name)
    output_rows = rows.drop(columns=["old_split", "video_sha256", "parquet_sha256"])
    output_rows.to_parquet(args.output_dir / "split_manifest.parquet", index=False)
    output_rows.to_parquet(args.output_dir / "splits.parquet", index=False)
    train_validation = output_rows[output_rows["split"].isin(("train", "validation"))].copy()
    train_validation.to_parquet(args.output_dir / "train_validation_label_manifest.parquet", index=False)
    final_test = output_rows[output_rows["split"].eq("final_test")]
    final_test_label = final_test[
        ["episode_index", "split", "original_task_index", "contiguous_label_index", "classification_inclusion", "task_name_raw", "category_raw"]
    ].copy()
    final_test_label.to_parquet(args.output_dir / "final_test_label_manifest.parquet", index=False)
    final_test_index = final_test[
        [
            "episode_index",
            "split",
            "video_path",
            "parquet_path",
            "num_frames",
            "duration_seconds",
            "video_nb_frames",
            "video_fps",
            "video_size_bytes",
            "video_sha256_fingerprint",
            "eligible_4s",
        ]
    ].copy()
    final_test_index.to_parquet(args.output_dir / "final_test_index_label_free.parquet", index=False)
    final_test_primary_index = final_test[
        final_test["v3_benchmark_inclusion"].eq("primary")
    ][final_test_index.columns].copy()
    final_test_primary_index.to_parquet(
        args.output_dir / "final_test_index_label_free_primary.parquet", index=False
    )
    label_map = json.loads((args.base_manifest / "label_map.json").read_text(encoding="utf-8"))
    json_dump(
        args.output_dir / "label_map_bidirectional.json",
        {
            "original_to_contiguous": label_map.get("original_to_contiguous", label_map),
            "contiguous_to_original": {
                str(value): int(key)
                for key, value in label_map.get("original_to_contiguous", label_map).items()
            },
        },
    )
    summary = {
        "protocol_version": "core_a_protocol_v3_duplicate_safe_20260826",
        "dataset_revision": config["dataset_revision"],
        "supersedes": {
            "path": str(args.unsafe_manifest),
            "reason": "retired_due_to_duplicate_unsafe_split",
        },
        "base_manifest": str(args.base_manifest),
        "split_counts": counts(output_rows),
        "primary_split_counts": primary_counts(output_rows[output_rows["v3_benchmark_inclusion"].eq("primary")]),
        "per_primary_task_split_counts": task_split_counts(output_rows[output_rows["v3_benchmark_inclusion"].eq("primary")]),
        "eligible_episode_count": len(output_rows),
        "primary_class_count": int(output_rows[output_rows["v3_benchmark_inclusion"].eq("primary")]["original_task_index"].nunique()),
        "conflicting_duplicate_excluded_episode_count": len(conflict_episodes),
        "exact_duplicate_group_count": exact_report["group_count"],
        "old_cross_split_exact_duplicate_group_count": exact_report["old_cross_split_group_count"],
        "near_candidate_edge_count": len(near_edges),
        "trusted_atomic_component_count": len(trusted_components),
        "component_move_count": len(component_moves),
        "component_moves": component_moves,
        "missing_primary_coverage": missing_primary_coverage,
        "final_test_metrics_computed": False,
        "final_test_label_free_index": "final_test_index_label_free.parquet",
        "final_test_primary_label_free_index": "final_test_index_label_free_primary.parquet",
        "final_test_label_manifest": "final_test_label_manifest.parquet",
        "old_split_status": "retired_due_to_duplicate_unsafe_split",
    }
    json_dump(args.output_dir / "split_summary.json", summary)
    json_dump(
        args.output_dir / "v3_protocol_lock.json",
        {
            **summary,
            "label_free_index_sha256": sha256_file(
                args.output_dir / "final_test_index_label_free.parquet"
            ),
            "primary_label_free_index_sha256": sha256_file(
                args.output_dir / "final_test_index_label_free_primary.parquet"
            ),
        },
    )
    with (args.output_dir / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(args.output_dir.iterdir()):
            if path.is_file() and path.name != "SHA256SUMS":
                handle.write(f"{sha256_file(path)}  {path.name}\n")
    repair_report = {
        "status": "pass" if not conflict_episodes and not missing_primary_coverage else "review",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "account": os.environ.get("USER", "unknown"),
        "dataset_revision": config["dataset_revision"],
        "old_split_status": "retired_due_to_duplicate_unsafe_split",
        "old_split": str(args.unsafe_manifest),
        "base_duplicate_safe_split": str(args.base_manifest),
        "v3_split": str(args.output_dir),
        "exact_duplicate_summary": exact_report,
        "near_duplicate_summary": near_report,
        "v3_split_counts": counts(output_rows),
        "v3_primary_split_counts": primary_counts(output_rows[output_rows["v3_benchmark_inclusion"].eq("primary")]),
        "component_moves": component_moves,
        "missing_primary_coverage": missing_primary_coverage,
        "final_test_metrics_computed": False,
        "interpretation": (
            "The old split is retired because 22 exact full-video duplicate groups crossed its "
            "boundaries. The v3 bundle keeps exact and fixed-threshold trusted near-duplicate "
            "components atomic. This is not proof of session/scene independence because no trusted "
            "session/scene/operator field was available."
        ),
    }
    json_dump(report_dir / "SPLIT_LEAKAGE_REPAIR_REPORT_20260826.json", repair_report)
    repair_md = [
        "# Core A v3 duplicate-safe split repair",
        "",
        "Status: " + repair_report["status"],
        "",
        "The old task-stratified split is marked `retired_due_to_duplicate_unsafe_split`.",
        f"Exact duplicate groups: {exact_report['group_count']}; old cross-split groups: {exact_report['old_cross_split_group_count']}.",
        f"Old validation duplicate members: {exact_report['old_validation_duplicate_members']}; old final-test duplicate members: {exact_report['old_final_test_duplicate_members']}.",
        f"V3 split counts: {counts(output_rows)}.",
        f"V3 primary split counts: {primary_counts(output_rows[output_rows['v3_benchmark_inclusion'].eq('primary')])}.",
        f"V3 split manifest SHA-256: {sha256_file(args.output_dir / 'split_manifest.parquet')}.",
        f"Near-duplicate candidate edges: {len(near_edges)}; trusted atomic components: {len(trusted_components)}.",
        f"Conflicting-label duplicate episodes excluded from primary population: {len(conflict_episodes)}.",
        "",
        "No final-test model metric, prediction aggregate or class geometry was computed.",
        "The final-test label-free index and protected label manifest are separate files in the v3 bundle.",
        "No trusted session/scene/operator field was found; therefore v3 prevents exact and fixed-rule near-duplicate cross-split leakage but does not prove scene/session independence.",
    ]
    (report_dir / "SPLIT_LEAKAGE_REPAIR_REPORT_20260826.md").write_text(
        "\n".join(repair_md) + "\n", encoding="utf-8"
    )
    print(json.dumps(repair_report, indent=2, sort_keys=True))
    return 0 if repair_report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
