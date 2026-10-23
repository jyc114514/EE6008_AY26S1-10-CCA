"""Small, dependency-light contracts for the one-time JY-02 evaluation.

This module is intentionally not an evaluation runner.  It validates the
pre-registered model set, effect sign, and authorization state before a future
runner is allowed to do any work.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, Mapping

CONFIGS = ("JY02-S-A-D", "JY02-F-A-D", "JY02-S-R-D", "JY02-F-R-D")
SEEDS = (20260921, 20260922, 20260923)
EXPECTED_SHAPE = (30, 28)
PRIMARY_METRIC = "arm14_standardized_full_horizon_mae"
EFFECT_DEFINITION = "control_mae_minus_candidate_mae"
STATES = (
    "UNAUTHORIZED",
    "PROTOCOL_LOCKED",
    "HUMAN_APPROVED",
    "INPUT_INFERENCE_COMPLETE",
    "TARGET_UNLOCK_ONCE",
    "ANALYSIS_COMPLETE",
    "SEALED",
)

_TRANSITIONS = {
    ("UNAUTHORIZED", "lock_protocol"): "PROTOCOL_LOCKED",
    ("PROTOCOL_LOCKED", "human_approve"): "HUMAN_APPROVED",
    ("HUMAN_APPROVED", "complete_input_inference"): "INPUT_INFERENCE_COMPLETE",
    ("INPUT_INFERENCE_COMPLETE", "unlock_targets_once"): "TARGET_UNLOCK_ONCE",
    ("TARGET_UNLOCK_ONCE", "complete_analysis"): "ANALYSIS_COMPLETE",
    ("ANALYSIS_COMPLETE", "seal_outputs"): "SEALED",
}


def final_test_explicitly_authorized(environment: Mapping[str, str] | None = None) -> bool:
    """Return the explicit authorization flag; absence is fail-closed."""

    values = os.environ if environment is None else environment
    return values.get("FINAL_TEST_EXPLICITLY_AUTHORIZED", "false").strip().lower() == "true"


def require_final_authorization(environment: Mapping[str, str] | None = None) -> None:
    if not final_test_explicitly_authorized(environment):
        raise PermissionError("FINAL_TEST_EXPLICITLY_AUTHORIZED is not true")


def expected_learned_pairs() -> set[tuple[str, int]]:
    return {(config, seed) for config in CONFIGS for seed in SEEDS}


def validate_model_registry(rows: Iterable[Mapping[str, object]]) -> None:
    """Reject missing/extra seeds and any implicit best-seed or ensemble choice."""

    rows = list(rows)
    learned = [row for row in rows if row.get("model_type") == "learned"]
    pairs = {(str(row.get("config_id")), int(row.get("seed"))) for row in learned}
    if pairs != expected_learned_pairs() or len(learned) != 12:
        raise ValueError(f"expected exactly 4 configs x 3 seeds, got {len(learned)} rows")
    if sum(row.get("model_type") == "persistence" for row in rows) != 1:
        raise ValueError("registry must contain exactly one fixed Persistence control")
    for row in learned:
        selection = str(row.get("selection_rule", "")).lower()
        if "best seed" in selection or "ensemble" in selection:
            raise ValueError("best-seed selection and unregistered ensembles are forbidden")
        if tuple(row.get("expected_shape", ())) != EXPECTED_SHAPE:
            raise ValueError("unexpected prediction shape")


def effect(control_mae: float, candidate_mae: float) -> float:
    """Positive values mean the candidate has lower MAE."""

    return float(control_mae) - float(candidate_mae)


def next_state(state: str, event: str) -> str:
    if state not in STATES:
        raise ValueError(f"unknown state: {state}")
    try:
        return _TRANSITIONS[(state, event)]
    except KeyError as exc:
        raise PermissionError(f"invalid state transition: {state} + {event}") from exc


def validate_sha256(value: object) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value)))
