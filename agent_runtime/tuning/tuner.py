"""Deterministic adaptive tuning for RNOS recovery runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .metrics import RecoveryMetrics
from .profiles import TuningProfile


@dataclass(frozen=True)
class TuningDecision:
    profile: TuningProfile
    adjustments: dict[str, dict[str, float | int]] = field(default_factory=dict)
    reason: str = "no tuning adjustment"

    @property
    def changed(self) -> bool:
        return bool(self.adjustments)


class RNOSTuner:
    """Bounded, explainable RNOS threshold tuner."""

    def __init__(self, profile: TuningProfile | None = None) -> None:
        self.profile = profile or TuningProfile()
        self.history: list[dict[str, Any]] = []

    def adjust(self, context: dict[str, float | int | bool], metrics: RecoveryMetrics) -> TuningDecision:
        entropy = float(context.get("entropy", 0.0))
        drift = float(context.get("drift_score", 0.0))
        risk = float(context.get("tool_risk", 0.0))
        failures = int(context.get("validation_failures", 0))
        retry_count = int(context.get("retry_count", 0))
        recoverable = bool(context.get("recoverable", True))

        if risk >= self.profile.tool_risk_threshold:
            decision = self._apply(
                self.profile.adjusted(tool_risk_delta=-0.5, entropy_delta=-0.4),
                "Reduce exposure to destructive actions",
            )
        elif recoverable and 0 < failures <= 2:
            decision = self._apply(
                self.profile.adjusted(entropy_delta=0.8, retry_delta=1),
                "Recoverable failure detected, allowing retry",
            )
        elif retry_count >= self.profile.retry_limit or failures >= self.profile.retry_limit + 1:
            decision = self._apply(
                self.profile.adjusted(entropy_delta=-0.6, retry_delta=-1),
                "Prevent retry storm",
            )
        elif drift >= self.profile.drift_threshold:
            decision = self._apply(
                self.profile.adjusted(drift_delta=-0.4),
                "Tighten drift threshold and redirect to target",
            )
        else:
            decision = TuningDecision(profile=self.profile)

        self.history.append(
            {
                "context": dict(context),
                "metrics": {
                    "recovery_attempts": metrics.recovery_attempts,
                    "successful_recoveries": metrics.successful_recoveries,
                    "refusals": metrics.refusals,
                    "degradations": metrics.degradations,
                },
                "adjustments": decision.adjustments,
                "reason": decision.reason,
            }
        )
        return decision

    def _apply(self, updated: TuningProfile, reason: str) -> TuningDecision:
        adjustments = _diff_profiles(self.profile, updated)
        self.profile = updated
        return TuningDecision(profile=updated, adjustments=adjustments, reason=reason)


def _diff_profiles(old: TuningProfile, new: TuningProfile) -> dict[str, dict[str, float | int]]:
    changes: dict[str, dict[str, float | int]] = {}
    for field_name in ("entropy_threshold", "drift_threshold", "retry_limit", "tool_risk_threshold"):
        before = getattr(old, field_name)
        after = getattr(new, field_name)
        if before != after:
            changes[field_name] = {"old": before, "new": after}
    return changes
