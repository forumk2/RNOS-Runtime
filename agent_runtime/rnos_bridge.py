"""Bridge between Agent Gate signals and the existing RNOS policy engine."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Literal

from rnos.policy import PolicyConfig, evaluate_policy
from rnos.types import PolicyDecision

from .tuning.classifier import classify_failure

GateAction = Literal["ALLOW", "DEGRADE", "RECOVER", "REFUSE"]


@dataclass(frozen=True)
class RNOSContext:
    entropy: float
    retry_count: int
    drift_score: float
    tool_risk: float
    validation_failures: int
    destructive_action: bool = False
    risk_escalation: bool = False
    retry_limit: int = 2
    malformed_output: bool = False
    previous_failures: int | None = None
    previous_entropy: float | None = None


@dataclass(frozen=True)
class GateDecision:
    action: GateAction
    entropy: float
    trust: float
    reasons: tuple[str, ...] = field(default_factory=tuple)
    constraints: dict[str, object] = field(default_factory=dict)
    failure_type: str = "unknown"
    improvement: bool | None = None


class RNOSBridge:
    """Adapts agent-loop control signals to RNOS ALLOW/DEGRADE/RECOVER/REFUSE."""

    def __init__(
        self,
        refuse_threshold: float | None = None,
        degrade_threshold: float | None = None,
    ) -> None:
        refuse = refuse_threshold if refuse_threshold is not None else _env_float("RNOS_ENTROPY_THRESHOLD", 7.0)
        degrade = degrade_threshold if degrade_threshold is not None else _env_float("RNOS_DEGRADE_THRESHOLD", 4.5)
        self.policy_config = PolicyConfig(
            refuse_entropy=refuse,
            degrade_entropy=degrade,
            refuse_trust=-1.0,
            degrade_trust=-1.0,
        )

    def evaluate(self, context: RNOSContext) -> GateDecision:
        entropy = self._compose_entropy(context)
        trust = round(max(0.0, min(1.0, 1.0 - (entropy / 10.0))), 3)
        failure_type = classify_failure(
            {
                "entropy": context.entropy,
                "retry_count": context.retry_count,
                "drift_score": context.drift_score,
                "tool_risk": context.tool_risk,
                "validation_failures": context.validation_failures,
                "destructive_action": context.destructive_action,
                "malformed_output": context.malformed_output,
            }
        )
        improvement = self._compute_improvement(context, entropy)

        if failure_type == "fatal_risk" or context.tool_risk >= 9.0:
            reason = self._explain(context, entropy, "REFUSE", failure_type)
            return GateDecision(
                action="REFUSE",
                entropy=max(entropy, 9.0),
                trust=min(trust, 0.1),
                reasons=(reason,),
                constraints={"execute": False},
                failure_type=failure_type,
                improvement=improvement,
            )

        has_failure_signal = (
            context.validation_failures > 0
            or context.retry_count > 0
            or context.malformed_output
        )
        recoverable = failure_type in {"recoverable_validation", "malformed_output", "unknown"}
        if has_failure_signal and recoverable:
            if (
                failure_type == "unknown"
                and improvement is False
                and context.retry_count > 1
                and context.previous_failures is not None
            ):
                if context.validation_failures > context.previous_failures:
                    return GateDecision(
                        action="REFUSE",
                        entropy=max(entropy, self.policy_config.refuse_entropy),
                        trust=min(trust, 0.2),
                        reasons=("retry worsened outcome",),
                        constraints={"execute": False},
                        failure_type=failure_type,
                        improvement=improvement,
                    )
            if context.retry_count < context.retry_limit:
                return GateDecision(
                    action="RECOVER",
                    entropy=entropy,
                    trust=trust,
                    reasons=(self._explain(context, entropy, "RECOVER", failure_type),),
                    constraints={
                        "single_file_only": True,
                        "allow_side_effects": False,
                        "max_files_modified": 1,
                        "max_lines_changed": 30,
                        "remove_risky_actions": True,
                    },
                    failure_type=failure_type,
                    improvement=improvement,
                )
            if improvement is False or context.retry_count >= context.retry_limit:
                return GateDecision(
                    action="REFUSE",
                    entropy=max(entropy, self.policy_config.refuse_entropy),
                    trust=min(trust, 0.2),
                    reasons=("no improvement after retries",),
                    constraints={"execute": False},
                    failure_type=failure_type,
                    improvement=improvement,
                )

        assessment = evaluate_policy(entropy, trust, self.policy_config)
        action = self._map_action(assessment.decision)
        return GateDecision(
            action=action,
            entropy=round(assessment.entropy, 3),
            trust=round(assessment.trust, 3),
            reasons=(self._explain(context, entropy, action, failure_type),),
            constraints=dict(assessment.constraints),
            failure_type=failure_type,
            improvement=improvement,
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

    def _explain(
        self,
        context: RNOSContext,
        entropy: float,
        action: GateAction,
        failure_type: str = "unknown",
    ) -> str:
        if action == "RECOVER":
            if failure_type == "malformed_output":
                return "malformed output recoverable with strict format retry"
            if failure_type == "recoverable_validation":
                return "recoverable failure, attempting constrained retry"
            return "retry allowed for fixable failure"
        if failure_type == "fatal_risk" or context.destructive_action or context.tool_risk >= 9.0:
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

    @staticmethod
    def _compute_improvement(context: RNOSContext, entropy: float) -> bool | None:
        if context.previous_failures is None and context.previous_entropy is None:
            return None
        if context.previous_failures is not None:
            if context.validation_failures < context.previous_failures:
                return True
            if context.validation_failures > context.previous_failures:
                return False
        if context.previous_entropy is not None:
            if entropy < context.previous_entropy:
                return True
            if entropy > context.previous_entropy:
                return False
        return False


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
