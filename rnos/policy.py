"""Runtime policy logic for allow/degrade/refuse decisions."""

from __future__ import annotations

from dataclasses import dataclass

from .types import PolicyDecision, RuntimeAssessment


@dataclass(slots=True)
class PolicyConfig:
    """Threshold configuration for the RNOS policy engine.

    All fields have defaults matching the original hardcoded values so that
    existing callers that omit a config continue to behave identically.
    """

    refuse_entropy: float = 6.0
    refuse_trust: float = 0.2
    degrade_entropy: float = 3.0
    degrade_trust: float = 0.45


def evaluate_policy(
    entropy: float,
    trust: float,
    config: PolicyConfig | None = None,
) -> RuntimeAssessment:
    """Convert entropy and trust into a runtime decision.

    Args:
        entropy: Current instability score from :func:`calculate_entropy`.
        trust: Current confidence score from :func:`calculate_trust`.
        config: Optional threshold overrides. Uses hardcoded defaults when None.
    """

    cfg = config if config is not None else PolicyConfig()
    reasons: list[str] = []
    constraints: dict[str, int | bool] = {}

    if entropy >= cfg.refuse_entropy or trust <= cfg.refuse_trust:
        reasons.append("runtime_unstable")
        return RuntimeAssessment(
            entropy=entropy,
            trust=trust,
            decision=PolicyDecision.REFUSE,
            reasons=reasons,
        )

    if entropy >= cfg.degrade_entropy or trust <= cfg.degrade_trust:
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
