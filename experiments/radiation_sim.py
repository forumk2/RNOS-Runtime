"""Experiment that combines depth, retries, and failures."""

from __future__ import annotations

from rnos.runtime import RNOSRuntime
from rnos.types import ActionRecord


def run() -> list[tuple[int, str, float, float]]:
    runtime = RNOSRuntime()
    observations: list[tuple[int, str, float, float]] = []
    for step in range(6):
        action = ActionRecord(
            tool_name="sensor_array",
            depth=step,
            retry_count=step // 2,
            metadata={"noise": step * 0.2},
        )
        assessment = runtime.evaluate(action)
        runtime.record_outcome(action, success=step < 2)
        observations.append((step, assessment.decision.value, assessment.entropy, assessment.trust))
        if assessment.decision.value == "refuse":
            break
    return observations
