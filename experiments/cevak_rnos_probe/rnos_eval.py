"""Simplified RNOS evaluator for the probe experiment.

Inputs: final-step retry_count, branching_factor, cumulative_cost.
Threshold logic only — no full entropy math.
Evaluated at the final step of each trace.
"""

from __future__ import annotations

from dataclasses import dataclass

from .scenario_generator import Step, Trace

# ---------------------------------------------------------------------------
# Thresholds (configurable)
# Calibrated so STABLE_CORRECT comfortably sits below LOW,
# and RETRY_STORM exceeds HIGH well before the final step.
# ---------------------------------------------------------------------------
LOW_RETRY: int = 3
HIGH_RETRY: int = 10

LOW_BRANCH: int = 3
HIGH_BRANCH: int = 8

LOW_COST: float = 80.0
HIGH_COST: float = 200.0


@dataclass(slots=True)
class RnosResult:
    regime: str     # STABLE | DEGRADING | COLLAPSING
    action: str     # CONTINUE | CONTAIN | REFUSE
    retry: int
    branch: int
    cost: float


def evaluate(trace: Trace) -> RnosResult:
    """Evaluate a trace using only the final step's execution-layer fields."""
    final: Step = trace.steps[-1]
    retry = final.retry_count
    branch = final.branching_factor
    cost = final.cumulative_cost

    collapsing = (
        retry > HIGH_RETRY
        or branch > HIGH_BRANCH
        or cost > HIGH_COST
    )
    degrading = (
        retry > LOW_RETRY
        or branch > LOW_BRANCH
        or cost > LOW_COST
    )

    if collapsing:
        return RnosResult("COLLAPSING", "REFUSE", retry, branch, cost)
    if degrading:
        return RnosResult("DEGRADING", "CONTAIN", retry, branch, cost)
    return RnosResult("STABLE", "CONTINUE", retry, branch, cost)
