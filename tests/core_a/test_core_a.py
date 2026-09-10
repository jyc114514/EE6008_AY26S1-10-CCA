from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from ee6008.cache import CacheConfigMismatch, FeatureCache
from ee6008.config import load_config
from ee6008.data.clips import (
    ShortEpisodeError,
    preprocess_vjepa,
    timestamp_frame_indices,
)
from ee6008.data.splits import build_splits
from ee6008.metrics import classification_metrics, require_label_safe_split
from ee6008.probe import train_linear_probe


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _episode_rows(task_counts: dict[int, int]) -> pd.DataFrame:
    rows = []
    episode_index = 0
    for task_index, count in task_counts.items():
        for _ in range(count):
            rows.append(
                {
                    "episode_index": episode_index,
                    "original_task_index": task_index,
                    "task_name_raw": f"Basic/task_{task_index}",
                    "task_name_normalized": f"basic/task {task_index}",
                    "category_raw": "Basic",
                    "category_normalized": "basic",
                    "video_path": f"videos/episode_{episode_index:06d}.mp4",
                    "parquet_path": f"data/episode_{episode_index:06d}.parquet",
                    "num_frames": 120,
                    "duration_seconds": 4.0,
                    "video_fps": 30.0,
                    "action_dim": 28,
                    "arm_dim": 14,
                    "hand_dim": 14,
                    "leg_dim": 15,
                    "eligible_4s": True,
                    "validation_status": "valid",
                }
            )
            episode_index += 1
    return pd.DataFrame(rows)


def test_config_parsing_and_frame_contract():
    config = load_config(PROJECT_ROOT / "configs/core_a/core_a_benchmark_v3.yaml")
    assert config.frame_count == 32
    assert len(config.config_hash) == 64


def test_timestamp_frame_sampling_and_short_episode():
    indices, timestamps = timestamp_frame_indices(num_frames=120, source_fps=30.0)
    assert len(indices) == 32
    assert indices[0] == 0
    assert timestamps[-1] == pytest.approx(31 / 8)
    with pytest.raises(ShortEpisodeError):
        timestamp_frame_indices(num_frames=119, source_fps=30.0)


def test_preprocess_shape_and_finiteness():
    frames = np.zeros((32, 480, 640, 3), dtype=np.uint8)
    output = preprocess_vjepa(frames)
    assert tuple(output.shape) == (3, 32, 384, 384)
    assert torch.isfinite(output).all()


def test_split_coverage_disjointness_and_low_sample_policy():
    table, summary = build_splits(_episode_rows({0: 6, 1: 5, 2: 2}))
    assert len(table) == 13
    assert table["episode_index"].is_unique
    assert set(table["split"]) == {"train", "validation", "final_test"}
    assert summary["low_sample_tasks"] == [2]
    primary = table[table["classification_inclusion"] == "primary"]
    for task_index in (0, 1):
        assert set(primary.loc[primary.original_task_index == task_index, "split"]) == {
            "train",
            "validation",
            "final_test",
        }


def test_final_test_access_guard():
    require_label_safe_split("validation")
    with pytest.raises(PermissionError):
        require_label_safe_split("final_test")


def test_feature_cache_resume_and_config_mismatch(tmp_path):
    cache = FeatureCache(tmp_path / "cache", {"contract": "a"})
    status = cache.write(
        episode_index=1,
        pooled=torch.ones(4),
        tokens=torch.ones(2, 4),
        metadata={"split": "train"},
    )
    assert status == "created"
    assert cache.has_valid(1)
    assert (
        cache.write(
            episode_index=1,
            pooled=torch.zeros(4),
            tokens=torch.zeros(2, 4),
            metadata={"split": "train"},
        )
        == "existing"
    )
    mismatched = FeatureCache(tmp_path / "cache", {"contract": "b"})
    with pytest.raises(CacheConfigMismatch):
        mismatched.write(
            episode_index=1,
            pooled=torch.zeros(4),
            tokens=torch.zeros(2, 4),
            metadata={"split": "train"},
        )


def test_metric_correctness():
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [4.0, 0.0], [0.0, 4.0]])
    labels = torch.tensor([0, 1, 1, 1])
    metrics = classification_metrics(logits, labels, num_classes=2)
    assert metrics["top1"] == pytest.approx(0.75)
    assert metrics["top5"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_mean_class_recall_at_5_is_not_scalar_top5():
    logits = torch.zeros((3, 6))
    logits[0, 0] = 10.0
    logits[1, 0] = 10.0
    logits[2, :5] = torch.arange(5.0, 0.0, -1.0)
    labels = torch.tensor([0, 0, 5])
    metrics = classification_metrics(logits, labels, num_classes=6)
    assert metrics["top5"] == pytest.approx(2 / 3)
    assert metrics["mean_class_recall_at_5"] == pytest.approx(0.5)


def test_synthetic_probe_overfit():
    torch.manual_seed(0)
    train_features = torch.cat([torch.ones(6, 4), -torch.ones(6, 4)])
    train_labels = torch.tensor([0] * 6 + [1] * 6)
    validation_features = torch.cat([torch.ones(2, 4), -torch.ones(2, 4)])
    validation_labels = torch.tensor([0, 0, 1, 1])
    result = train_linear_probe(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        num_classes=2,
        epochs=60,
    )
    assert result.metrics["top1"] == pytest.approx(1.0)
