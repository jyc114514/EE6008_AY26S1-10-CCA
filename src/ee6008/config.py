"""Configuration loading and safety checks for Core A."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CoreAConfig:
    project_root: Path
    g1_root: Path
    manifest_dir: Path
    checkpoint: Path
    vjepa_source_root: Path
    feature_root: Path
    source_commit: str
    dataset_revision: str
    observation_seconds: float = 4.0
    sample_fps: int = 8
    source_fps: float = 30.0
    input_resolution: int = 384
    tubelet_size: int = 2
    model_name: str = "vjepa2_1_vit_base_384"
    checkpoint_key: str = "ema_encoder"
    pooled_dtype: str = "float32"
    token_dtype: str = "float16"
    allow_final_test_label_metrics: bool = False

    @property
    def frame_count(self) -> int:
        return round(self.observation_seconds * self.sample_fps)

    @property
    def config_hash(self) -> str:
        payload = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.__dict__.items()
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def validate(self) -> None:
        if self.observation_seconds != 4.0:
            raise ValueError("Core A primary contract requires observation_seconds=4.0")
        if self.sample_fps != 8:
            raise ValueError("Core A primary contract requires sample_fps=8")
        if self.frame_count != 32:
            raise ValueError("Core A primary contract requires 32 sampled frames")
        if self.input_resolution != 384:
            raise ValueError("V-JEPA 2.1-B primary contract requires 384px input")
        if self.tubelet_size != 2:
            raise ValueError("locked V-JEPA 2.1-B contract requires tubelet_size=2")
        if self.model_name != "vjepa2_1_vit_base_384":
            raise ValueError("only the locked V-JEPA 2.1 ViT-B/384 encoder is enabled")
        if self.checkpoint_key != "ema_encoder":
            raise ValueError("the locked V-JEPA 2.1-B checkpoint key is ema_encoder")
        if self.pooled_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"unsupported pooled dtype: {self.pooled_dtype}")
        if self.token_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError(f"unsupported token dtype: {self.token_dtype}")

    def to_dict(self) -> dict[str, Any]:
        result = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in self.__dict__.items()
        }
        result["frame_count"] = self.frame_count
        result["config_hash"] = self.config_hash
        return result


def expand_path(value: str | Path) -> Path:
    """Expand portable environment-variable paths used by public configs."""
    return Path(os.path.expandvars(str(value))).expanduser()


def load_config(path: Path) -> CoreAConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"config must be a YAML mapping: {path}")
    required = {
        "project_root",
        "g1_root",
        "manifest_dir",
        "checkpoint",
        "vjepa_source_root",
        "feature_root",
        "source_commit",
        "dataset_revision",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"config missing required keys: {missing}")
    config = CoreAConfig(
        project_root=expand_path(raw["project_root"]),
        g1_root=expand_path(raw["g1_root"]),
        manifest_dir=expand_path(raw["manifest_dir"]),
        checkpoint=expand_path(raw["checkpoint"]),
        vjepa_source_root=expand_path(raw["vjepa_source_root"]),
        feature_root=expand_path(raw["feature_root"]),
        source_commit=str(raw["source_commit"]),
        dataset_revision=str(raw["dataset_revision"]),
        observation_seconds=float(raw.get("observation_seconds", 4.0)),
        sample_fps=int(raw.get("sample_fps", 8)),
        source_fps=float(raw.get("source_fps", 30.0)),
        input_resolution=int(raw.get("input_resolution", 384)),
        tubelet_size=int(raw.get("tubelet_size", 2)),
        model_name=str(raw.get("model_name", "vjepa2_1_vit_base_384")),
        checkpoint_key=str(raw.get("checkpoint_key", "ema_encoder")),
        pooled_dtype=str(raw.get("pooled_dtype", "float32")),
        token_dtype=str(raw.get("token_dtype", "float16")),
        allow_final_test_label_metrics=bool(
            raw.get("allow_final_test_label_metrics", False)
        ),
    )
    config.validate()
    return config
