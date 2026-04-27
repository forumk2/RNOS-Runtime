"""Deterministic adaptive tuning for RNOS recovery runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .metrics import RecoveryMetrics
from .profiles import TuningProfile


@dataclass(frozen=True)
class TuningDecision:
    profile: TuningProfile
    adjustments: dict[str, dict[str, float | int | bool]] = field(default_factory=dict)
    reason: str = "no tuning adjustment"

    @property
    def changed(self) -> bool:
        return bool(self.adjustments)


class RNOSTuner:
    """Bounded, explainable RNOS threshold tuner."""

    def __init__(self, profile: TuningProfile | None = None) -> None:
        self.profile = profile or TuningProfile()
        self.history: list[dict[str, Any]] = []

    def adjust(self, context: dict[str, float | int | bool | str], metrics: RecoveryMetrics) -> TuningDecision:
        drift = float(context.get("drift_score", 0.0))
        failures = int(context.get("validation_failures", 0))
        retry_count = int(context.get("retry_count", 0))
        failure_type = str(context.get("failure_type", "unknown"))

        if failure_type == "recoverable_validation":
            decision = self._apply(
                self.profile.adjusted(retry_delta=1, entropy_delta=0.5),
                "Recoverable validation failure",
            )
        elif failure_type == "drift":
            decision = self._apply(
                self.profile.adjusted(drift_delta=-0.5),
                "Drift detected, tightening drift threshold",
            )
        elif failure_type == "malformed_output":
            decision = self._apply(
                self.profile.adjusted(retry_delta=1, enforce_strict_format=True),
                "Malformed output, enforcing strict format retry",
            )
        elif failure_type == "fatal_risk":
            decision = self._apply(
                self.profile.adjusted(tool_risk_delta=-1.0),
                "Fatal risk detected, reducing tool risk threshold",
            )
        elif retry_count >= self.profile.retry_limit or failures >= self.profile.retry_limit + 1:
            decision = self._apply(
                self.profile.adjusted(entropy_delta=-0.6, retry_delta=-1),
                "Prevent retry storm",
            )
        elif drift >= self.profile.drift_threshold:
            decision = self._apply(
                self.profile.adjusted(drift_delta=-0.5),
                "Drift detected, tightening drift threshold",
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


def _diff_profiles(old: TuningProfile, new: TuningProfile) -> dict[str, dict[str, float | int | bool]]:
    changes: dict[str, dict[str, float | int | bool]] = {}
    for field_name in (
        "entropy_threshold",
        "drift_threshold",
        "retry_limit",
        "tool_risk_threshold",
        "enforce_strict_format",
    ):
        before = getattr(old, field_name)
        after = getattr(new, field_name)
        if before != after:
            changes[field_name] = {"old": before, "new": after}
    return changes
