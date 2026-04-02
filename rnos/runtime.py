"""Main RNOS evaluation loop."""

from __future__ import annotations

from .entropy import calculate_entropy
from .logger import get_logger, write_trace
from .policy import evaluate_policy
from .trust import calculate_trust
from .types import ActionRecord, RuntimeAssessment


class RNOSRuntime:
    """Evaluates whether an agent action should continue executing."""

    def __init__(self) -> None:
        self.history: list[ActionRecord] = []
        self.logger = get_logger("rnos.runtime")

    def evaluate(self, action: ActionRecord) -> RuntimeAssessment:
        """Assess an action before execution."""

        entropy = calculate_entropy(self.history, action)
        trust = calculate_trust(self.history, entropy)
        assessment = evaluate_policy(entropy, trust)

        self.logger.info(
            "tool=%s entropy=%.2f trust=%.2f decision=%s",
            action.tool_name,
            assessment.entropy,
            assessment.trust,
            assessment.decision.value,
        )
        write_trace(
            {
                "stage": "assessment",
                "tool": action.tool_name,
                "payload": action.payload,
                "depth": action.depth,
                "retry_count": action.retry_count,
                "metadata": action.metadata,
                "entropy": assessment.entropy,
                "trust": assessment.trust,
                "decision": assessment.decision.value,
                "reasons": assessment.reasons,
                "constraints": assessment.constraints,
            }
        )
        return assessment

    def record_outcome(self, action: ActionRecord, *, success: bool) -> None:
        """Store the outcome of an executed action."""

        action.success = success
        self.history.append(action)
        write_trace(
            {
                "stage": "outcome",
                "tool": action.tool_name,
                "payload": action.payload,
                "depth": action.depth,
                "retry_count": action.retry_count,
                "metadata": action.metadata,
                "success": success,
            }
        )
