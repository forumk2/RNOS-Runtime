"""Trust model for execution confidence."""

from __future__ import annotations

from collections.abc import Sequence

from .types import ActionRecord


def calculate_trust(history: Sequence[ActionRecord], entropy: float) -> float:
    """Return a confidence score between 0.0 and 1.0."""

    if not history:
        baseline = 0.65
    else:
        successes = sum(1 for item in history[-10:] if item.success is True)
        failures = sum(1 for item in history[-10:] if item.success is False)
        total = max(successes + failures, 1)
        baseline = max(0.3, successes / total)

    entropy_penalty = min(entropy / 12.0, 0.7)
    trust = max(0.0, min(1.0, baseline - entropy_penalty + 0.2))
    return round(trust, 3)
