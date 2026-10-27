import numpy as np
import pytest

from oracle_audit import (
    advantage_arrays,
    bootstrap_mean,
    composite_full28,
    oracle_select,
    validate_prediction_contract,
)


def test_advantage_sign_positive_negative_and_tie():
    target = np.zeros((1, 2, 2), dtype=np.float64)
    persistence = np.ones_like(target)
    neural = np.array([[[0.25, 0.25], [1.0, 1.0]]], dtype=np.float64)
    _p_error, _n_error, advantage = advantage_arrays(persistence, neural, target)
    assert np.all(advantage[:, 0] > 0)
    assert np.all(advantage[:, 1] == 0)
    tie_p = np.zeros_like(target)
    tie_n = np.zeros_like(target)
    _a, _b, tie_adv = advantage_arrays(tie_p, tie_n, target)
    assert np.all(tie_adv == 0)


def test_oracle_granularities_preserve_shapes_and_tie_persistence():
    target = np.zeros((2, 30, 2), dtype=np.float64)
    persistence = np.ones_like(target)
    neural = np.ones_like(target)
    neural[0, :3] = 0.0
    neural[0, 3:] = 2.0
    experts = {"PERSISTENCE": persistence, "N_RAW": neural}
    for granularity in ("GLOBAL", "EPISODE", "WINDOW5", "STEP_GROUP", "JOINT_STEP"):
        output, choice, scores = oracle_select(experts, target, granularity)
        assert output.shape == target.shape
        assert choice.shape == (2,) if granularity == "GLOBAL" else choice.shape[0] == 2
        assert np.asarray(scores["PERSISTENCE"]).ndim == 0 if granularity == "GLOBAL" else scores["PERSISTENCE"].shape[0] == 2
    output, choice, _scores = oracle_select({"PERSISTENCE": persistence, "N_RAW": persistence.copy()}, target, "STEP_GROUP")
    assert np.array_equal(output, persistence)
    assert np.all(choice == 0)


def test_window5_and_full28_use_canonical_hand_then_arm_order():
    hand = np.full((1, 30, 2), 2.0)
    arm = np.full((1, 30, 2), 3.0)
    target = np.zeros((1, 30, 4), dtype=np.float64)
    result = composite_full28(hand, arm, target)
    assert result.shape == target.shape
    assert np.all(result[:, :, :2] == 2.0)
    assert np.all(result[:, :, 2:] == 3.0)


def test_bootstrap_is_episode_level_and_finite():
    values = np.arange(8, dtype=np.float64)
    result = bootstrap_mean(values, np.random.default_rng(3), replicates=200)
    assert result["N_episode"] == 8
    assert result["replicates"] == 200
    assert np.isfinite([result["effect"], result["ci_low"], result["ci_high"]]).all()


def test_prediction_contract_fails_closed_on_nan_or_shape():
    good = np.zeros((3, 30, 14), dtype=np.float64)
    validate_prediction_contract(good, "good")
    with pytest.raises(ValueError, match="shape"):
        validate_prediction_contract(np.zeros((3, 29, 14)), "bad-shape")
    bad = good.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_prediction_contract(bad, "bad-finite")
