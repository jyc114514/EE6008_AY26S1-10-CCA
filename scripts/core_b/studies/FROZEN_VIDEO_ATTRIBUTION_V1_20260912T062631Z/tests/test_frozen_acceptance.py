from __future__ import annotations
import os

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_checkpoint_hashes_match_recovery() -> None:
    rows = list(csv.DictReader((ROOT / "FROZEN_CHECKPOINT_REGISTRY.csv").open()))
    assert {row["identity_status"] for row in rows} == {"verified_614_full_max_abs_0"}
    assert len(rows) == 2


def test_8row_identity_replay() -> None:
    for model in ("B3-V-D-F", "B3-FUS-D-F"):
        assert json.loads((ROOT / "predictions" / model / "STAGE0_IDENTITY.json").read_text())["max_abs"] == 0.0


def test_prediction_shape_30x28() -> None:
    for path in ROOT.glob("predictions/**/predictions.npz"):
        with np.load(path, allow_pickle=False) as archive:
            assert archive["predictions_standardized"].shape[1:] == (30, 28)


def test_arm_hand_mapping() -> None:
    assert json.loads((ROOT / "FROZEN_ATTR_PROTOCOL_LOCKED.json").read_text())["source_population"]["target_shape"] == [614, 30, 28]
    assert "arm14=14:28" in (ROOT / "FROZEN_ATTR_PROTOCOL_LOCKED.md").read_text()


def test_damage_effect_sign() -> None:
    rows = list(csv.DictReader((ROOT / "PAIRED_EFFECTS.csv").open()))
    assert all("damage" in row for row in rows)
    assert next(float(row["damage"]) for row in rows if row["cell"] == "FF-R1" and row["metric"] == "arm_full") > 0


def test_bootstrap_episode_pairing() -> None:
    assert np.load(ROOT / "statistics/paired_bootstrap_indices_N614.npy").shape == (10000, 614)
    assert np.load(ROOT / "statistics/paired_bootstrap_indices_N589.npy").shape == (10000, 589)


def test_failure_and_success_terminal_markers() -> None:
    assert (ROOT / "predictions/B3-FUS-D-F/RUN_FAILED.json").exists()
    assert (ROOT / "predictions/B3-V-D-F/RUN_COMPLETE.json").exists()
    assert (ROOT / "predictions/B3-FUS-D-F_FEATURE_RETRY_20260912T070723Z/RUN_COMPLETE.json").exists()


def test_optimizer_steps_zero() -> None:
    for path in ROOT.glob("predictions/**/RUN_CONFIG_LOCKED.json"):
        assert json.loads(path.read_text())["optimizer_steps"] == 0


def test_final_test_zero() -> None:
    for path in ROOT.glob("predictions/**/RUN_CONFIG_LOCKED.json"):
        assert json.loads(path.read_text())["final_test_accessed"] == 0


def test_gpu_pool_1_2_3_and_gpu0_rejected() -> None:
    watcher = (Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/reports/core_b/ROUTE_C_VIDEO_ATTRIBUTION_CONTROLS_V1_20260910T154453Z/code/task_local_gpu_watcher.py'))).read_text()
    shared = Path(os.path.join(os.environ.get("EE6008_PROJECT_ROOT", "."), 'repo/ee6008-humanoid-vjepa/tools/gpu_lease_watcher.py')).read_text()
    assert "ALLOWED = (1, 2, 3)" in watcher and "ALLOWED_GPUS = (2, 3)" in shared
    assert "if gpu not in ALLOWED" in watcher and "if not isinstance(gpu, int) or gpu not in ALLOWED_GPUS" in shared


def test_lease_release() -> None:
    for path in ROOT.glob("logs/*watcher.json"):
        payload = json.loads(path.read_text())
        assert payload["lease_released"] is True


def test_output_links_resolve_to_distinct_files() -> None:
    assert len({p.resolve() for p in ROOT.glob("figures/png/*.png")}) == 12


def test_pairing_no_fixed_points() -> None:
    rows = list(csv.DictReader((ROOT / "PAIRING_MANIFEST.csv").open()))
    assert all(row["same_task_fixed_point"] == "False" for row in rows if row["same_task_eligible"] == "True")
    assert all(row["cross_task_same_task"] == "False" for row in rows)


def test_zero_changes_only_video_feature() -> None:
    audit = json.loads((ROOT / "predictions/B3-FUS-D-F_FEATURE_RETRY_20260912T070723Z/FF-R1/INTERVENTION_AUDIT.json").read_text())
    assert audit["condition"] == "R1" and audit["level"] == "normalized_feature"
    assert audit["recipient_state_target_id_unchanged"] is True
