from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from frozen_prediction_analysis import compute_mae_from_frozen_predictions
from target_blind_inference import run_target_blind_inference


def test_target_blind_inference_writes_contract_shape_without_target(tmp_path: Path):
    inputs = tmp_path / "synthetic_inputs.npz"
    predictions = tmp_path / "synthetic_predictions.npz"
    np.savez(inputs, episode_index=np.arange(3), features=np.ones((3, 4), dtype=np.float32))

    result = run_target_blind_inference(
        inputs,
        lambda x: np.zeros((len(x), 30, 28), dtype=np.float32),
        predictions,
    )
    assert result["target_file_opened"] is False
    assert result["shape"] == [3, 30, 28]
    assert predictions.exists()


def test_analysis_requires_frozen_predictions_and_explicit_phase(tmp_path: Path):
    prediction_path = tmp_path / "predictions.npz"
    target_path = tmp_path / "synthetic_targets.npz"
    manifest_path = tmp_path / "prediction_manifest.json"
    ids = np.arange(2)
    pred = np.zeros((2, 30, 28), dtype=np.float32)
    target = np.ones((2, 30, 28), dtype=np.float32)
    np.savez(prediction_path, episode_index=ids, predictions=pred)
    np.savez(target_path, episode_index=ids, targets=target)
    manifest_path.write_text(
        json.dumps({
            "frozen": True,
            "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        }),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError):
        compute_mae_from_frozen_predictions(prediction_path, manifest_path, target_path)
    assert compute_mae_from_frozen_predictions(
        prediction_path, manifest_path, target_path, allow_target_access=True
    ) == pytest.approx(1.0)
