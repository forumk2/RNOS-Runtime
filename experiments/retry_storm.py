"""Experiment that models repeated transient failures."""

from __future__ import annotations

from rnos.runtime import RNOSRuntime
from rnos.types import ActionRecord


def run() -> list[tuple[int, str, float, float]]:
    runtime = RNOSRuntime()
    observations: list[tuple[int, str, float, float]] = []
    for retry_count in range(6):
        action = ActionRecord(tool_name="unstable_api", retry_count=retry_count)
        assessment = runtime.evaluate(action)
        runtime.record_outcome(action, success=False)
        observations.append((retry_count, assessment.decision.value, assessment.entropy, assessment.trust))
        if assessment.decision.value == "refuse":
            break
    return observations
