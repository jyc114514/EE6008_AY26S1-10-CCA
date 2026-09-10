"""Deterministic, task-stratified split construction for the G1 corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_SEED = 20260825
DEFAULT_MIN_EPISODES_PER_CLASS = 3
SPLIT_NAMES = ("train", "validation", "final_test")


def _stable_key(seed: int, task_index: int, episode_index: int) -> str:
    payload = f"{seed}:{task_index}:{episode_index}".encode()
    return hashlib.sha256(payload).hexdigest()


def _allocate_counts(n: int) -> dict[str, int]:
    """Allocate approximately 70/15/15 while preserving all three splits."""
    if n < 3:
        # These rows are retained for corpus accounting but are not primary
        # classification classes.  Keeping them in the train split avoids
        # inventing a class-balanced test result for an under-sampled task.
        return {"train": n, "validation": 0, "final_test": 0}
    validation = max(1, round(n * 0.15))
    final_test = max(1, round(n * 0.15))
    while validation + final_test >= n:
        if validation >= final_test and validation > 1:
            validation -= 1
        elif final_test > 1:
            final_test -= 1
        else:
            break
    return {
        "train": n - validation - final_test,
        "validation": validation,
        "final_test": final_test,
    }


def build_splits(
    episodes: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    min_episodes_per_class: int = DEFAULT_MIN_EPISODES_PER_CLASS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return an episode split table and an auditable split summary."""
    required = {
        "episode_index",
        "original_task_index",
        "task_name_raw",
        "task_name_normalized",
        "category_raw",
        "category_normalized",
        "video_path",
        "parquet_path",
        "num_frames",
        "duration_seconds",
        "video_fps",
        "action_dim",
        "arm_dim",
        "hand_dim",
        "leg_dim",
        "eligible_4s",
        "validation_status",
    }
    missing = sorted(required - set(episodes.columns))
    if missing:
        raise ValueError(f"episode manifest is missing columns: {missing}")

    valid = episodes.loc[
        (episodes["validation_status"] == "valid")
        & episodes["eligible_4s"].astype(bool)
    ].copy()
    if valid.empty:
        raise ValueError("no valid 4-second episodes available")
    if valid["episode_index"].duplicated().any():
        raise ValueError("duplicate episode_index in input manifest")

    counts = valid.groupby("original_task_index").size().to_dict()
    active_tasks = sorted(counts)
    primary_tasks = sorted(t for t, n in counts.items() if n >= min_episodes_per_class)
    low_sample_tasks = sorted(
        t for t, n in counts.items() if n < min_episodes_per_class
    )
    primary_label_map = {str(task): index for index, task in enumerate(primary_tasks)}

    rows: list[dict[str, Any]] = []
    for task_index in active_tasks:
        task_rows = valid.loc[valid["original_task_index"] == task_index].copy()
        task_rows["_stable_key"] = [
            _stable_key(seed, int(task_index), int(episode_index))
            for episode_index in task_rows["episode_index"]
        ]
        task_rows = task_rows.sort_values(
            ["_stable_key", "episode_index"], kind="mergesort"
        )
        allocation = _allocate_counts(len(task_rows))
        cursor = 0
        for split in SPLIT_NAMES:
            amount = allocation[split]
            selected = task_rows.iloc[cursor : cursor + amount]
            cursor += amount
            for _, row in selected.iterrows():
                record = row.drop(labels=["_stable_key"]).to_dict()
                record.update(
                    {
                        "split": split,
                        "contiguous_label_index": int(
                            primary_label_map.get(str(task_index), -1)
                        ),
                        "classification_inclusion": (
                            "primary"
                            if task_index in primary_tasks
                            else "low_sample_excluded"
                        ),
                        "split_assignment_key": _stable_key(
                            seed, int(task_index), int(row["episode_index"])
                        ),
                    }
                )
                rows.append(record)

    result = (
        pd.DataFrame(rows)
        .sort_values("episode_index", kind="mergesort")
        .reset_index(drop=True)
    )
    expected = set(valid["episode_index"].astype(int))
    actual = set(result["episode_index"].astype(int))
    if expected != actual:
        raise AssertionError(
            f"split coverage mismatch: missing={sorted(expected - actual)[:10]} extra={sorted(actual - expected)[:10]}"
        )
    if result["episode_index"].duplicated().any():
        raise AssertionError("split output contains duplicate episodes")

    primary = result[result["classification_inclusion"] == "primary"]
    coverage: dict[str, dict[str, int]] = {}
    for task_index in primary_tasks:
        task_rows = primary[primary["original_task_index"] == task_index]
        coverage[str(task_index)] = {
            split: int((task_rows["split"] == split).sum()) for split in SPLIT_NAMES
        }
        if any(coverage[str(task_index)][split] < 1 for split in SPLIT_NAMES):
            raise AssertionError(
                f"primary task {task_index} is not present in all splits"
            )

    summary: dict[str, Any] = {
        "split_policy": "task-stratified episode split; no reliable group field was present",
        "seed": seed,
        "target_fractions": {"train": 0.70, "validation": 0.15, "final_test": 0.15},
        "min_episodes_per_primary_class": min_episodes_per_class,
        "input_valid_4s_episode_count": len(valid),
        "output_episode_count": len(result),
        "active_task_count": len(active_tasks),
        "primary_task_count": len(primary_tasks),
        "low_sample_task_count": len(low_sample_tasks),
        "low_sample_tasks": low_sample_tasks,
        "primary_tasks": primary_tasks,
        "split_counts": {
            split: int((result["split"] == split).sum()) for split in SPLIT_NAMES
        },
        "primary_split_counts": {
            split: int((primary["split"] == split).sum()) for split in SPLIT_NAMES
        },
        "per_primary_task_split_counts": coverage,
        "final_test_policy": "label-blind frozen feature extraction and cache validation only; no label metrics or geometry",
        "low_sample_policy": "retained in corpus and split table, assigned to train, excluded from primary classification",
        "label_map": primary_label_map,
    }
    return result, summary


def write_manifest_bundle(
    episodes: pd.DataFrame,
    tasks: pd.DataFrame,
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    min_episodes_per_class: int = DEFAULT_MIN_EPISODES_PER_CLASS,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    split_table, summary = build_splits(
        episodes,
        seed=seed,
        min_episodes_per_class=min_episodes_per_class,
    )
    # Preserve the audited episode inventory in this immutable Core-A bundle.
    episodes.to_parquet(output_dir / "episodes.parquet", index=False)
    tasks.to_csv(output_dir / "tasks.csv", index=False)
    split_table.to_parquet(output_dir / "splits.parquet", index=False)
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    label_rows = tasks.copy()
    label_rows["contiguous_label_index"] = (
        label_rows["original_task_index"]
        .map({int(k): int(v) for k, v in summary["label_map"].items()})
        .fillna(-1)
        .astype(int)
    )
    label_rows["classification_inclusion"] = (
        label_rows["original_task_index"]
        .isin(summary["primary_tasks"])
        .map({True: "primary", False: "inactive_or_low_sample"})
    )
    label_map = {
        str(int(row.original_task_index)): {
            "contiguous_label_index": int(row.contiguous_label_index),
            "classification_inclusion": row.classification_inclusion,
            "task_name_raw": row.task_name_raw,
            "task_name_normalized": row.task_name_normalized,
            "category_raw": row.category_raw,
            "category_normalized": row.category_normalized,
            "valid_4s_episode_count": int(row.valid_4s_episode_count),
        }
        for row in label_rows.itertuples()
    }
    (output_dir / "label_map.json").write_text(
        json.dumps(label_map, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    hashes: list[str] = []
    for path in sorted(output_dir.iterdir()):
        if path.name == "SHA256SUMS" or not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.append(f"{digest}  {path.name}")
    (output_dir / "SHA256SUMS").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    return summary
