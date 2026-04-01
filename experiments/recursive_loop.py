"""Experiment that simulates repeated tool recursion."""

from __future__ import annotations

from rnos.runtime import RNOSRuntime
from rnos.types import ActionRecord


def run() -> list[tuple[int, str, float, float]]:
    runtime = RNOSRuntime()
    observations: list[tuple[int, str, float, float]] = []
    for depth in range(6):
        action = ActionRecord(tool_name="recursive_probe", depth=depth)
        assessment = runtime.evaluate(action)
        runtime.record_outcome(action, success=assessment.decision.value != "refuse")
        observations.append((depth, assessment.decision.value, assessment.entropy, assessment.trust))
        if assessment.decision.value == "refuse":
            break
    return observations
