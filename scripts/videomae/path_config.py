"""Resolve a caller-selected VideoMAE scratch area without reading data."""

from __future__ import annotations

import os
import re
from pathlib import Path

_ENV_NAME = "EE6008_VIDEOMAE_SCRATCH_NAME"
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def scratch_name() -> str:
    """Return the configured single directory name or fail before a path is used."""
    name = os.environ.get(_ENV_NAME)
    if name is None or not _SAFE_NAME.fullmatch(name):
        raise ValueError(
            f"Set {_ENV_NAME} to one ASCII directory name "
            "(letters/digits followed by letters, digits, '_' or '-')."
        )
    return name


def scratch_root(project: Path) -> Path:
    """Resolve project/tmp/<name>, rejecting the tmp root and symlink escapes."""
    project_root = Path(project).resolve()
    tmp_root = (project_root / "tmp").resolve()
    scratch = (tmp_root / scratch_name()).resolve()
    if scratch == tmp_root or not scratch.is_relative_to(tmp_root):
        raise ValueError(f"Configured {_ENV_NAME} must resolve below project/tmp.")
    return scratch
