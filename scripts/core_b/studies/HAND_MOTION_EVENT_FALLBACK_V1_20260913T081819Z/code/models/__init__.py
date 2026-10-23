"""Models for the locked Hand event-gate exploratory matrix."""

from .event_gated_residual import (
    HandEventGatedResidualModel,
    base_parameter_count,
    build_model,
    event_parameter_count,
    split_hidden_for_parameter_match,
)

__all__ = [
    "HandEventGatedResidualModel",
    "base_parameter_count",
    "build_model",
    "event_parameter_count",
    "split_hidden_for_parameter_match",
]
