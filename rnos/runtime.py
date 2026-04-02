"""Main RNOS evaluation loop."""

from __future__ import annotations

from pathlib import Path

from .entropy import calculate_entropy
from .logger import get_logger, write_trace
from .policy import PolicyConfig, evaluate_policy
from .trust import calculate_trust
from .types import ActionRecord, RuntimeAssessment

_DEFAULT_TRACE_PATH = Path("logs/rnos_trace.jsonl")


class RNOSRuntime:
    """Evaluates whether an agent action should continue executing."""

    def __init__(
        self,
        trace_path: str | Path | None = None,
        policy_config: PolicyConfig | None = None,
    ) -> None:
        """Initialise the runtime.

        Args:
            trace_path: Absolute or relative path to the JSONL trace file.
                Defaults to ``logs/rnos_trace.jsonl`` relative to CWD.
            policy_config: Optional threshold overrides forwarded to
                :func:`evaluate_policy`.
        """
        self.history: list[ActionRecord] = []
        self.logger = get_logger("rnos.runtime")
        self._trace_path: Path = Path(trace_path) if trace_path is not None else _DEFAULT_TRACE_PATH
        self._policy_config = policy_config

    def evaluate(self, action: ActionRecord) -> RuntimeAssessment:
        """Assess an action before execution."""

        entropy = calculate_entropy(self.history, action)
        trust = calculate_trust(self.history, entropy)
        assessment = evaluate_policy(entropy, trust, self._policy_config)

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
            },
            path=self._trace_path,
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
            },
            path=self._trace_path,
        )
