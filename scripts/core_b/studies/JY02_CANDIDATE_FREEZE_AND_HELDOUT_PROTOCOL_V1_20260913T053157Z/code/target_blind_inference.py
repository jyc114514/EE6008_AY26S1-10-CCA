"""Target-blind inference helper used only by the engineering dry-run.

The public function deliberately has no target/label argument.  It loads a
label-free input bundle, calls a supplied model, and writes a prediction
bundle.  Target analysis lives in ``frozen_prediction_analysis.py`` and is a
separate, explicitly authorized phase.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import numpy as np


_FORBIDDEN_INPUT_MARKERS = (
    "final_test_label_manifest",
    "protected_label_manifest",
    "final_test_targets",
    "final_targets",
    "target_values",
)


def _assert_label_free_input_path(path: Path) -> None:
    lowered = str(path).lower()
    if any(marker in lowered for marker in _FORBIDDEN_INPUT_MARKERS):
        raise PermissionError(f"target-bearing path is not an inference input: {path}")


def run_target_blind_inference(
    input_path: str | Path,
    model: Callable[[np.ndarray], np.ndarray],
    output_path: str | Path,
    *,
    input_key: str = "features",
    expected_shape: tuple[int, int] = (30, 28),
) -> dict[str, object]:
    """Run a model without opening a target/label file."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    _assert_label_free_input_path(input_path)
    with np.load(input_path, allow_pickle=False) as bundle:
        if input_key not in bundle or "episode_index" not in bundle:
            raise ValueError("input bundle must contain features and episode_index")
        features = np.asarray(bundle[input_key])
        episode_index = np.asarray(bundle["episode_index"])
    predictions = np.asarray(model(features))
    if predictions.shape != (len(episode_index), *expected_shape):
        raise ValueError(f"prediction shape {predictions.shape} does not match contract")
    if not np.isfinite(predictions).all():
        raise ValueError("prediction bundle contains non-finite values")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, episode_index=episode_index, predictions=predictions)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return {
        "status": "PASS",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "N": int(len(episode_index)),
        "shape": list(predictions.shape),
        "target_file_opened": False,
        "prediction_sha256": digest,
    }
