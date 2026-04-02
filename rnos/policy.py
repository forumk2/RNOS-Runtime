"""Runtime policy logic for allow/degrade/refuse decisions."""

from __future__ import annotations

from .types import PolicyDecision, RuntimeAssessment


def evaluate_policy(entropy: float, trust: float) -> RuntimeAssessment:
    """Convert entropy and trust into a runtime decision."""

    reasons: list[str] = []
    constraints: dict[str, int | bool] = {}

    if entropy >= 6.0 or trust <= 0.2:
        reasons.append("runtime_unstable")
        return RuntimeAssessment(
            entropy=entropy,
            trust=trust,
            decision=PolicyDecision.REFUSE,
            reasons=reasons,
        )

    if entropy >= 3.0 or trust <= 0.45:
        reasons.append("caution_window")
        constraints["max_additional_steps"] = 1
        constraints["allow_side_effects"] = False
        return RuntimeAssessment(
            entropy=entropy,
            trust=trust,
            decision=PolicyDecision.DEGRADE,
            reasons=reasons,
            constraints=constraints,
        )

    return RuntimeAssessment(
        entropy=entropy,
        trust=trust,
        decision=PolicyDecision.ALLOW,
        reasons=["healthy_execution"],
    )
