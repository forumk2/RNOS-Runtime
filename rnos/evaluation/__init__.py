"""Evaluation helpers for adversarial review of RNOS claims."""

from .domain_legitimacy import (
    EXAMPLE_SCENARIOS,
    aggregate_legitimacy,
    evaluate_claim,
    list_example_scenarios,
)

__all__ = [
    "EXAMPLE_SCENARIOS",
    "evaluate_claim",
    "aggregate_legitimacy",
    "list_example_scenarios",
]
