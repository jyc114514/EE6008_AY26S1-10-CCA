"""Atomic, resumable per-episode feature cache."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


class CacheConfigMismatch(RuntimeError):
    """Raised when an existing cache entry was made under another contract."""


def cache_config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class FeatureCache:
    def __init__(self, root: Path, config: dict[str, Any]):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.config_hash = cache_config_hash(config)
        (self.root / "cache_config.json").write_text(
            json.dumps(
                {**config, "config_hash": self.config_hash}, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        ) if not (self.root / "cache_config.json").exists() else None

    def path_for(self, episode_index: int) -> Path:
        return self.root / f"episode_{episode_index:06d}.pt"

    def has_valid(self, episode_index: int) -> bool:
        path = self.path_for(episode_index)
        if not path.exists():
            return False
        try:
            record = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, EOFError, RuntimeError, ValueError):
            return False
        return record.get("config_hash") == self.config_hash and all(
            key in record for key in ("episode_index", "pooled", "tokens")
        )

    def write(
        self,
        *,
        episode_index: int,
        pooled: torch.Tensor,
        tokens: torch.Tensor,
        metadata: dict[str, Any],
    ) -> str:
        destination = self.path_for(episode_index)
        if destination.exists():
            try:
                existing = torch.load(
                    destination, map_location="cpu", weights_only=True
                )
            except (OSError, EOFError, RuntimeError, ValueError) as exc:
                raise CacheConfigMismatch(
                    f"cannot read existing cache entry {destination}"
                ) from exc
            if existing.get("config_hash") != self.config_hash:
                raise CacheConfigMismatch(
                    f"config mismatch for existing cache entry {destination}"
                )
            return "existing"
        record = {
            "config_hash": self.config_hash,
            "episode_index": int(episode_index),
            "pooled": pooled.detach().cpu(),
            "tokens": tokens.detach().cpu(),
            "metadata": metadata,
        }
        partial = Path(f"{destination}.part")
        if partial.exists():
            partial.unlink()
        with partial.open("wb") as handle:
            torch.save(record, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists():
            raise FileExistsError(
                f"destination appeared while writing cache: {destination}"
            )
        os.replace(partial, destination)
        return "created"

    def load(self, episode_index: int) -> dict[str, Any]:
        path = self.path_for(episode_index)
        record = torch.load(path, map_location="cpu", weights_only=True)
        if record.get("config_hash") != self.config_hash:
            raise CacheConfigMismatch(f"config mismatch for {path}")
        return record
