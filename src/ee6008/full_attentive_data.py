"""v3 train/development-validation population and token-cache gates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ee6008.config import CoreAConfig
from ee6008.data.selection import benchmark_primary_mask
from ee6008.full_attentive import validate_token_shape

EXPECTED_PRIMARY_COUNTS = {"train": 2757, "validation": 617}
EXPECTED_CLASS_COUNT = 120
EXPECTED_TOKEN_SHAPE = (9216, 768)
EXPECTED_POOLED_SHAPE = (768,)


@dataclass(frozen=True)
class Population:
    rows: pd.DataFrame
    task_ids: tuple[int, ...]
    local_label_map: dict[int, int]
    population_sha256: str
    split_sha256: str
    label_map_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _population_hash(rows: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    ordered = rows.sort_values("episode_index", kind="mergesort")
    for row in ordered.itertuples(index=False):
        digest.update(
            f"{int(row.episode_index)}\t{row.split}\t{int(row.local_label_index)}\n".encode()
        )
    return digest.hexdigest()


def load_v3_population(config: CoreAConfig, *, full: bool = True) -> Population:
    """Load only train/development-validation rows, never final-test labels."""
    split_path = config.manifest_dir / "splits.parquet"
    label_map_path = config.manifest_dir / "label_map.json"
    table = pd.read_parquet(
        split_path,
        columns=[
            "episode_index",
            "split",
            "v3_benchmark_inclusion",
            "validation_status",
            "original_task_index",
            "contiguous_label_index",
            "video_sha256_fingerprint",
        ],
        filters=[("split", "in", ["train", "validation"])],
    )
    rows = table[
        table["split"].isin(("train", "validation"))
        & benchmark_primary_mask(table)
        & table["validation_status"].eq("valid")
    ].copy()
    if rows["episode_index"].duplicated().any():
        raise ValueError("v3 population has duplicate episode IDs")
    if (
        rows["episode_index"].isna().any()
        or rows["contiguous_label_index"].isna().any()
    ):
        raise ValueError("v3 population has null episode or label values")
    counts = rows.groupby("split").size().to_dict()
    if full and counts != EXPECTED_PRIMARY_COUNTS:
        raise ValueError(f"unexpected v3 primary train/validation counts: {counts}")
    historical_labels = sorted(
        int(value) for value in rows["contiguous_label_index"].unique()
    )
    historical_task_order = [
        int(row.original_task_index)
        for row in rows[["original_task_index", "contiguous_label_index"]]
        .drop_duplicates()
        .sort_values(
            ["contiguous_label_index", "original_task_index"], kind="mergesort"
        )
        .itertuples(index=False)
    ]
    local_label_map = {
        task_id: index for index, task_id in enumerate(historical_task_order)
    }
    if full and (
        len(local_label_map) != EXPECTED_CLASS_COUNT
        or sorted(local_label_map.values()) != list(range(EXPECTED_CLASS_COUNT))
        or len(historical_labels) != EXPECTED_CLASS_COUNT
    ):
        raise ValueError(
            "v3 train/development-validation labels are not exactly 120 classes"
        )
    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    primary_map = {
        int(task_id): int(info["contiguous_label_index"])
        for task_id, info in label_map.items()
        if info.get("classification_inclusion") == "primary"
    }
    if any(task_id not in primary_map for task_id in local_label_map):
        raise ValueError("v3 primary task is absent from label_map.json")
    if any(
        primary_map.get(int(row.original_task_index)) != int(row.contiguous_label_index)
        for row in rows.itertuples()
    ):
        raise ValueError("v3 row labels do not round-trip through label_map.json")
    rows["local_label_index"] = rows["original_task_index"].map(local_label_map).astype(int)
    return Population(
        rows=rows.sort_values("episode_index", kind="mergesort").reset_index(drop=True),
        task_ids=tuple(historical_task_order),
        local_label_map=local_label_map,
        population_sha256=_population_hash(rows),
        split_sha256=sha256_file(split_path),
        label_map_sha256=sha256_file(label_map_path),
    )


def attach_feature_manifest(
    population: Population, feature_run: Path
) -> tuple[pd.DataFrame, str]:
    """Join feature paths by episode only; v3 owns the split assignment."""
    manifest_path = feature_run / "feature_manifest.parquet"
    columns = [
        "episode_index",
        "split",
        "feature_path",
        "feature_sha256",
        "token_shape",
        "pooled_shape",
    ]
    manifest = pd.read_parquet(manifest_path, columns=columns)
    if manifest["episode_index"].duplicated().any():
        raise ValueError(f"feature manifest has duplicate episode IDs: {manifest_path}")
    joined = population.rows.merge(
        manifest.drop(columns=["split"]),
        on="episode_index",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing = joined.loc[joined["_merge"].ne("both"), "episode_index"].tolist()
        raise ValueError(f"feature manifest is missing v3 episodes: {missing[:10]}")
    joined = joined.drop(columns=["_merge"])
    joined["resolved_feature_path"] = joined["feature_path"].map(
        lambda value: (
            str(value)
            if Path(str(value)).is_absolute()
            else str(feature_run / str(value))
        )
    )
    if (
        not joined["resolved_feature_path"]
        .map(lambda value: Path(value).is_file())
        .all()
    ):
        missing = joined.loc[
            ~joined["resolved_feature_path"].map(lambda value: Path(value).is_file()),
            "resolved_feature_path",
        ].tolist()
        raise FileNotFoundError(f"feature files are missing: {missing[:5]}")
    for column, expected in (
        ("token_shape", list(EXPECTED_TOKEN_SHAPE)),
        ("pooled_shape", list(EXPECTED_POOLED_SHAPE)),
    ):
        shapes = joined[column].map(
            lambda value: json.loads(value) if isinstance(value, str) else list(value)
        )
        if not shapes.map(lambda value, expected=expected: value == expected).all():
            raise ValueError(f"feature manifest {column} is not {expected}")
    first = torch.load(
        joined.iloc[0].resolved_feature_path, map_location="cpu", weights_only=True
    )
    validate_token_shape(first.get("tokens"), expected=EXPECTED_TOKEN_SHAPE)
    pooled = first.get("pooled")
    if (
        not isinstance(pooled, torch.Tensor)
        or tuple(pooled.shape) != EXPECTED_POOLED_SHAPE
    ):
        raise ValueError(f"pooled feature shape is not {EXPECTED_POOLED_SHAPE}")
    return joined, sha256_file(manifest_path)


def alignment_report(
    config: CoreAConfig, dynamic_run: Path, static_run: Path
) -> dict[str, Any]:
    population = load_v3_population(config, full=True)
    dynamic, dynamic_hash = attach_feature_manifest(population, dynamic_run)
    static, static_hash = attach_feature_manifest(population, static_run)
    dynamic_ids = dynamic["episode_index"].tolist()
    static_ids = static["episode_index"].tolist()
    if dynamic_ids != static_ids:
        raise ValueError("dynamic/static episode ordering differs")
    for column in (
        "split",
        "original_task_index",
        "contiguous_label_index",
        "local_label_index",
    ):
        if dynamic[column].tolist() != static[column].tolist():
            raise ValueError(f"dynamic/static {column} alignment differs")
    return {
        "population_rows": len(population.rows),
        "split_counts": {
            key: int(value)
            for key, value in population.rows.groupby("split").size().to_dict().items()
        },
        "class_count": len(population.task_ids),
        "population_sha256": population.population_sha256,
        "split_sha256": population.split_sha256,
        "label_map_sha256": population.label_map_sha256,
        "dynamic_manifest_sha256": dynamic_hash,
        "static_manifest_sha256": static_hash,
        "dynamic_static_episode_alignment": True,
        "dynamic_static_split_alignment": True,
        "dynamic_static_label_alignment": True,
        "final_test_selected": False,
    }
