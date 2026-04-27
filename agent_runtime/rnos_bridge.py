"""Bridge between Agent Gate signals and the existing RNOS policy engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from rnos.policy import PolicyConfig, evaluate_policy
from rnos.types import PolicyDecision


GateAction = Literal["ALLOW", "DEGRADE", "REFUSE"]


@dataclass(frozen=True)
class RNOSContext:
    entropy: float
    retry_count: int
    drift_score: float
    tool_risk: float
    validation_failures: int
    destructive_action: bool = False
    risk_escalation: bool = False


@dataclass(frozen=True)
class GateDecision:
    action: GateAction
    entropy: float
    trust: float
    reasons: tuple[str, ...] = field(default_factory=tuple)
    constraints: dict[str, object] = field(default_factory=dict)


class RNOSBridge:
    """Adapts agent-loop control signals to RNOS ALLOW/DEGRADE/REFUSE."""

    def __init__(
        self,
        refuse_threshold: float = 7.0,
        degrade_threshold: float = 4.5,
    ) -> None:
        self.policy_config = PolicyConfig(
            refuse_entropy=refuse_threshold,
            degrade_entropy=degrade_threshold,
            refuse_trust=-1.0,
            degrade_trust=-1.0,
        )

    def evaluate(self, context: RNOSContext) -> GateDecision:
        entropy = self._compose_entropy(context)
        trust = round(max(0.0, min(1.0, 1.0 - (entropy / 10.0))), 3)

        if context.destructive_action or context.tool_risk >= 9.0:
            reason = self._explain(context, entropy, "REFUSE")
            return GateDecision(
                action="REFUSE",
                entropy=max(entropy, 9.0),
                trust=min(trust, 0.1),
                reasons=(reason,),
                constraints={"execute": False},
            )

        assessment = evaluate_policy(entropy, trust, self.policy_config)
        action = self._map_action(assessment.decision)
        return GateDecision(
            action=action,
            entropy=round(assessment.entropy, 3),
            trust=round(assessment.trust, 3),
            reasons=(self._explain(context, entropy, action),),
            constraints=dict(assessment.constraints),
        )

    def _compose_entropy(self, context: RNOSContext) -> float:
        score = max(context.entropy, 0.0)
        score += min(context.retry_count * 0.9, 2.7)
        score += min(context.validation_failures * 1.3, 3.9)
        score += context.tool_risk * 0.55
        score += context.drift_score * 0.35

        if context.retry_count >= 2 and context.validation_failures >= 2:
            score += 1.7
        if context.risk_escalation:
            score += 0.8

        return round(max(0.0, min(10.0, score)), 3)

    @staticmethod
    def _map_action(decision: PolicyDecision) -> GateAction:
        if decision is PolicyDecision.REFUSE:
            return "REFUSE"
        if decision is PolicyDecision.DEGRADE:
            return "DEGRADE"
        return "ALLOW"

    def _explain(self, context: RNOSContext, entropy: float, action: GateAction) -> str:
        if context.destructive_action or context.tool_risk >= 9.0:
            return "high tool risk + large blast radius"
        if action == "REFUSE" and context.retry_count >= 2 and context.validation_failures >= 2:
            return "drift increasing with repeated failures"
        if action == "REFUSE" and entropy >= self.policy_config.refuse_entropy:
            return "entropy threshold exceeded"
        if action == "DEGRADE" and context.drift_score >= 4.5:
            return "drift increasing"
        if action == "DEGRADE" and context.tool_risk >= 6.0:
            return "tool risk elevated"
        if action == "DEGRADE":
            return "caution window reached"
        return "healthy execution"
