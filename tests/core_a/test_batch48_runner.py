from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from run_batch48_ablation_matrix_v3 import build_loss, load_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs/core_a/batch48"


def load_public_config(path: Path, tmp_path: Path):
    """Keep config-lock tests independent of excluded feature caches."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    feature_run = tmp_path / path.stem / "feature_run"
    feature_run.mkdir(parents=True)
    raw["feature_run"] = str(feature_run)
    temporary = tmp_path / path.name
    temporary.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_experiment(temporary)


def test_batch48_configs_lock_physical_batch_and_single_factors(tmp_path: Path) -> None:
    expected = {
        "standard_dynamic.yaml": ("standard", "dynamic", 4, 0.0, "cross_entropy", 2, 3),
        "standard_static.yaml": ("standard", "static_repeated_first_frame", 4, 0.0, "cross_entropy", 2, 3),
        "depth2_dynamic.yaml": ("depth2", "dynamic", 2, 0.0, "cross_entropy", 3, 1),
        "depth2_static.yaml": ("depth2", "static_repeated_first_frame", 2, 0.0, "cross_entropy", 3, 1),
        "label_smoothing_dynamic.yaml": ("label_smoothing", "dynamic", 4, 0.1, "label_smoothing_cross_entropy", 3, 1),
        "label_smoothing_static.yaml": ("label_smoothing", "static_repeated_first_frame", 4, 0.1, "label_smoothing_cross_entropy", 3, 1),
        "class_balanced_dynamic.yaml": ("class_balanced", "dynamic", 4, 0.0, "class_balanced_cross_entropy", 3, 1),
        "class_balanced_static.yaml": ("class_balanced", "static_repeated_first_frame", 4, 0.0, "class_balanced_cross_entropy", 3, 1),
    }
    assert set(expected) == {path.name for path in CONFIG_DIR.glob("*.yaml")}
    for name, (ablation, variant, depth, smoothing, loss_name, gpu, seed_count) in expected.items():
        experiment = load_public_config(CONFIG_DIR / name, tmp_path)
        assert (experiment.ablation, experiment.variant, experiment.probe_depth) == (ablation, variant, depth)
        assert experiment.label_smoothing == pytest.approx(smoothing)
        assert experiment.loss_name == loss_name
        assert experiment.physical_gpu == gpu
        assert experiment.batch_size == 48
        assert experiment.gradient_accumulation_steps == 1
        assert len(experiment.seeds) == seed_count
        assert experiment.core.allow_final_test_label_metrics is False


def test_batch48_losses_are_single_factor_and_class_weights_are_train_only(tmp_path: Path) -> None:
    standard = load_public_config(CONFIG_DIR / "standard_dynamic.yaml", tmp_path)
    smoothing = load_public_config(CONFIG_DIR / "label_smoothing_dynamic.yaml", tmp_path)
    balanced = load_public_config(CONFIG_DIR / "class_balanced_dynamic.yaml", tmp_path)
    standard_loss = build_loss(standard, torch.device("cpu"))
    smoothing_loss = build_loss(smoothing, torch.device("cpu"))
    balanced_loss = build_loss(balanced, torch.device("cpu"))
    assert standard_loss.label_smoothing == pytest.approx(0.0)
    assert standard_loss.weight is None
    assert smoothing_loss.label_smoothing == pytest.approx(0.1)
    assert smoothing_loss.weight is None
    assert balanced_loss.label_smoothing == pytest.approx(0.0)
    assert balanced_loss.weight is not None
    assert balanced_loss.weight.numel() == 120
    assert float(balanced_loss.weight.mean()) == pytest.approx(1.0, abs=1e-6)
    with balanced.class_weight_path.open(encoding="utf-8") as handle:
        weights = json.load(handle)
    assert weights["source"] == "v3 train rows only"
    assert weights["train_count"] == 2757
    assert weights["class_count"] == 120


def test_batch48_configs_keep_final_test_denied(tmp_path: Path) -> None:
    for path in CONFIG_DIR.glob("*.yaml"):
        experiment = load_public_config(path, tmp_path)
        assert experiment.raw["final_test_allowed"] is False
        assert experiment.core.allow_final_test_label_metrics is False
