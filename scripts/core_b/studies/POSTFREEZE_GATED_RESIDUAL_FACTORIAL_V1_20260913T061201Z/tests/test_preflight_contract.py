from __future__ import annotations
import os

import hashlib
import json
from pathlib import Path

import numpy as np

from train_gated_cell import CELL_ORDER, CELL_SPECS, INPUT_PATH, NORMALIZER_PATH, THRESHOLD_PATH


EXP_ROOT = Path(__file__).resolve().parents[1]
FREEZE_ROOT = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/JY02_CANDIDATE_FREEZE_AND_HELDOUT_PROTOCOL_V1_20260913T053157Z'))


def test_real_development_input_schema_and_counts():
    with np.load(INPUT_PATH, allow_pickle=False) as archive:
        assert {"episode_index", "original_task_index", "state_action_history", "dynamic_pooled", "targets_standardized", "persistence_standardized"}.issubset(archive.files)
        assert archive["state_action_history"].shape == (3355, 568)
        assert archive["dynamic_pooled"].shape == (3355, 768)
        assert archive["targets_standardized"].shape == (3355, 30, 28)
        assert archive["persistence_standardized"].shape == (3355, 30, 28)
        assert int((archive["split_code"] == 0).sum()) == 2741
        assert int((archive["split_code"] == 1).sum()) == 614


def test_train_only_normalizer_and_motion_threshold_contract():
    normalizer = json.loads(NORMALIZER_PATH.read_text())["1.0s"]
    threshold = json.loads(THRESHOLD_PATH.read_text())
    assert normalizer["source_split"] == "train only"
    assert normalizer["final_test_targets_accessed"] == 0
    assert normalizer["target_shape"] == [30, 28]
    assert threshold["source_split"] == "split_code=0"
    assert threshold["final_test_access"] == 0


def test_matrix_is_complete_and_seed_is_locked_before_results():
    matrix = json.loads((EXP_ROOT / "MATRIX.json").read_text())
    seed = json.loads((EXP_ROOT / "SEED_REGISTRY.json").read_text())
    assert matrix["status"] == "LOCKED"
    assert len(matrix["rows"]) == 8
    assert set(row["cell_id"] for row in matrix["rows"]) == set(CELL_ORDER)
    assert seed["status"] == "LOCKED_BEFORE_RESULTS"
    assert seed["selected_seed"] == 20260924
    assert seed["must_not_change_after_results"] is True


def test_no_final_or_frozen_protocol_mutation_is_requested_by_training_code():
    source = (EXP_ROOT / "code/train_gated_cell.py").read_text()
    assert "FINAL_TEST_EXPLICITLY_AUTHORIZED" not in source
    assert '\"final_test_access\": 0' in source
    assert "raw_video" in source
    assert "encoder_or_lora_modified" in source


def test_stage2_and_freeze_checksum_manifests_are_present():
    stage2 = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/JY02_RESIDUAL_TEMPORAL_DECODER_V1_20260912T095449Z/runs/stage2_authorized_20260912T151855Z/qa/CHECKSUMS.sha256'))
    freeze = FREEZE_ROOT / "qa/CHECKSUMS.sha256"
    assert stage2.is_file() and freeze.is_file()
    assert len([line for line in freeze.read_text().splitlines() if line.strip()]) == 85
    assert len([line for line in stage2.read_text().splitlines() if line.strip()]) == 284
    assert len(hashlib.sha256(freeze.read_bytes()).hexdigest()) == 64
