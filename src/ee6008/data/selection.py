"""Pre-registered, label-only smoke and pilot task selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def benchmark_primary_mask(split_table: pd.DataFrame) -> pd.Series:
    """Return the benchmark-inclusion mask for a split manifest.

    The duplicate-safe v3 manifest keeps the historical
    ``classification_inclusion`` column for provenance and adds
    ``v3_benchmark_inclusion`` so conflicting-label duplicate groups can be
    excluded without rewriting the raw classification accounting.  Older
    manifests do not have the v3 column and retain the historical behavior.
    """

    column = (
        "v3_benchmark_inclusion"
        if "v3_benchmark_inclusion" in split_table.columns
        else "classification_inclusion"
    )
    return split_table[column].eq("primary")


def _task_key(seed: int, task_index: int) -> str:
    return hashlib.sha256(f"{seed}:{task_index}".encode()).hexdigest()


def select_tasks(
    split_table: pd.DataFrame,
    *,
    seed: int = 20260825,
    task_metadata: pd.DataFrame | None = None,
) -> dict[str, Any]:
    primary = split_table[benchmark_primary_mask(split_table)].copy()
    if primary.empty:
        raise ValueError("no primary tasks available")
    grouped = primary.groupby(
        [
            "original_task_index",
            "task_name_raw",
            "task_name_normalized",
            "category_raw",
            "category_normalized",
        ],
        as_index=False,
    ).agg(valid_4s_episode_count=("episode_index", "count"))
    grouped["selection_key"] = grouped["original_task_index"].map(
        lambda value: _task_key(seed, int(value))
    )
    if task_metadata is not None:
        metadata = task_metadata[
            [
                "original_task_index",
                "task_name_raw",
                "task_name_normalized",
                "category_raw",
                "category_normalized",
            ]
        ].drop_duplicates("original_task_index")
        grouped = grouped.drop(
            columns=[
                "task_name_raw",
                "task_name_normalized",
                "category_raw",
                "category_normalized",
            ]
        ).merge(metadata, on="original_task_index", how="left")
    grouped = grouped.sort_values(
        ["category_normalized", "valid_4s_episode_count", "selection_key"],
        ascending=[True, False, True],
        kind="mergesort",
    )

    smoke_tasks: list[int] = []
    for category in sorted(grouped["category_normalized"].unique()):
        candidate = grouped[grouped["category_normalized"] == category].iloc[0]
        smoke_tasks.append(int(candidate.original_task_index))
        if len(smoke_tasks) == 2:
            break
    if (
        len(smoke_tasks) != 2
        or grouped[grouped["original_task_index"].isin(smoke_tasks)][
            "category_normalized"
        ].nunique()
        != 2
    ):
        raise AssertionError("smoke selection did not produce two distinct categories")

    pilot_tasks: list[int] = []
    for category in sorted(grouped["category_normalized"].unique()):
        candidate = grouped[grouped["category_normalized"] == category].iloc[0]
        pilot_tasks.append(int(candidate.original_task_index))
    remaining = grouped[~grouped["original_task_index"].isin(pilot_tasks)].sort_values(
        ["valid_4s_episode_count", "selection_key"],
        ascending=[False, True],
        kind="mergesort",
    )
    pilot_tasks.extend(
        int(value) for value in remaining["original_task_index"].tolist()
    )
    pilot_tasks = pilot_tasks[:12]
    if len(pilot_tasks) < 12:
        raise AssertionError("fewer than 12 primary tasks available for pilot")

    def records(task_ids: list[int]) -> list[dict[str, Any]]:
        return [
            {
                key: (int(row[key]) if key == "original_task_index" else row[key])
                for key in (
                    "original_task_index",
                    "task_name_raw",
                    "task_name_normalized",
                    "category_raw",
                    "category_normalized",
                    "valid_4s_episode_count",
                )
                for _, row in grouped[grouped["original_task_index"] == task_id]
                .head(1)
                .iterrows()
            }
            for task_id in task_ids
        ]

    selection = {
        "selection_rule": {
            "seed": seed,
            "smoke": "top-count task in the first two lexicographic broad categories; no model results used",
            "pilot": "top-count task from every available broad category, then highest-count tasks until 12; hash tie-break",
            "ordering": "category_normalized, descending valid 4s episode count, SHA-256(seed:task_index)",
        },
        "smoke_tasks": smoke_tasks,
        "smoke": records(smoke_tasks),
        "pilot_tasks": pilot_tasks,
        "pilot": records(pilot_tasks),
    }
    selection["selection_sha256"] = hashlib.sha256(
        json.dumps(selection, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return selection


def write_selection(
    split_path: Path,
    output_path: Path,
    *,
    seed: int = 20260825,
    tasks_path: Path | None = None,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite selection manifest: {output_path}"
        )
    task_metadata = pd.read_csv(tasks_path) if tasks_path is not None else None
    selection = select_tasks(
        pd.read_parquet(split_path), seed=seed, task_metadata=task_metadata
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return selection
