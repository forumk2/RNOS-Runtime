"""Failure classification helpers shared by RNOS recovery flows."""

from __future__ import annotations

from typing import Any


HIGH_RISK_THRESHOLD = 8.5
DRIFT_THRESHOLD = 4.5


def classify_failure(context: dict[str, Any]) -> str:
    """Classify the current instability signal into a recovery policy bucket."""

    tool_risk = float(context.get("tool_risk", 0.0))
    validation_failures = int(context.get("validation_failures", 0))
    entropy = float(context.get("entropy", 0.0))
    drift_score = float(context.get("drift_score", 0.0))

    if tool_risk > HIGH_RISK_THRESHOLD or bool(context.get("destructive_action", False)):
        return "fatal_risk"
    if bool(context.get("malformed_output", False)):
        return "malformed_output"
    if drift_score > DRIFT_THRESHOLD:
        return "drift"
    if validation_failures > 0 and entropy < 7.0:
        return "recoverable_validation"
    return "unknown"
