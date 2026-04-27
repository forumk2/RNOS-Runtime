"""Tunable RNOS threshold profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
import os


@dataclass(frozen=True)
class TuningProfile:
    entropy_threshold: float = field(default_factory=lambda: _env_float("RNOS_ENTROPY_THRESHOLD", 7.0))
    drift_threshold: float = field(default_factory=lambda: _env_float("RNOS_DRIFT_THRESHOLD", 4.5))
    retry_limit: int = field(default_factory=lambda: _env_int("RNOS_RETRY_LIMIT", 2))
    tool_risk_threshold: float = field(default_factory=lambda: _env_float("RNOS_TOOL_RISK_THRESHOLD", 9.0))
    enforce_strict_format: bool = False

    min_entropy_threshold: float = 5.5
    max_entropy_threshold: float = 8.5
    min_drift_threshold: float = 3.0
    max_drift_threshold: float = 6.5
    min_retry_limit: int = 1
    max_retry_limit: int = 4
    min_tool_risk_threshold: float = 7.0
    max_tool_risk_threshold: float = 9.5

    def adjusted(
        self,
        *,
        entropy_delta: float = 0.0,
        drift_delta: float = 0.0,
        retry_delta: int = 0,
        tool_risk_delta: float = 0.0,
        enforce_strict_format: bool | None = None,
    ) -> "TuningProfile":
        return TuningProfile(
            entropy_threshold=_clamp(
                self.entropy_threshold + entropy_delta,
                self.min_entropy_threshold,
                self.max_entropy_threshold,
            ),
            drift_threshold=_clamp(
                self.drift_threshold + drift_delta,
                self.min_drift_threshold,
                self.max_drift_threshold,
            ),
            retry_limit=int(
                _clamp(
                    self.retry_limit + retry_delta,
                    self.min_retry_limit,
                    self.max_retry_limit,
                )
            ),
            tool_risk_threshold=_clamp(
                self.tool_risk_threshold + tool_risk_delta,
                self.min_tool_risk_threshold,
                self.max_tool_risk_threshold,
            ),
            enforce_strict_format=(
                self.enforce_strict_format
                if enforce_strict_format is None
                else enforce_strict_format
            ),
        )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
