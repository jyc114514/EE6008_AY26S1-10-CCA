from __future__ import annotations

import pytest

from final_eval_contract import (
    CONFIGS,
    EXPECTED_SHAPE,
    SEEDS,
    effect,
    expected_learned_pairs,
    final_test_explicitly_authorized,
    next_state,
    validate_model_registry,
)


def _registry():
    return [
        {
            "model_type": "learned",
            "config_id": config,
            "seed": seed,
            "expected_shape": EXPECTED_SHAPE,
            "selection_rule": "fixed checkpoint_best from each locked seed",
        }
        for config in CONFIGS
        for seed in SEEDS
    ] + [{"model_type": "persistence", "config_id": "PERSISTENCE", "seed": 0}]


def test_registry_has_exactly_four_configs_by_three_seeds():
    validate_model_registry(_registry())
    assert len(expected_learned_pairs()) == 12


def test_effect_sign_is_control_minus_candidate():
    assert effect(0.20, 0.15) == pytest.approx(0.05)
    assert effect(0.15, 0.20) == pytest.approx(-0.05)


def test_authorization_defaults_false():
    assert final_test_explicitly_authorized({}) is False
    assert final_test_explicitly_authorized({"FINAL_TEST_EXPLICITLY_AUTHORIZED": "false"}) is False


def test_state_machine_is_linear_and_fail_closed():
    state = "UNAUTHORIZED"
    for event, expected in [
        ("lock_protocol", "PROTOCOL_LOCKED"),
        ("human_approve", "HUMAN_APPROVED"),
        ("complete_input_inference", "INPUT_INFERENCE_COMPLETE"),
        ("unlock_targets_once", "TARGET_UNLOCK_ONCE"),
        ("complete_analysis", "ANALYSIS_COMPLETE"),
        ("seal_outputs", "SEALED"),
    ]:
        state = next_state(state, event)
        assert state == expected
    with pytest.raises(PermissionError):
        next_state("PROTOCOL_LOCKED", "unlock_targets_once")
