"""Hybrid controller composing RNOS and a circuit breaker.

Implements a "safety-first" merge strategy: the more-severe decision from
either sub-system wins.  This gives the hybrid at least the protection of
the better-performing controller on any given failure geometry.

Decision severity mapping
-------------------------
    ALLOW / closed          → 0
    DEGRADE / half_open     → 1
    REFUSE / open / perm    → 2

The merged decision is whichever sub-system produces the highest severity.
``hybrid_trigger_source`` indicates which sub-system drove the outcome:
"rnos", "cb", or "both" when severity is tied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .runtime import RNOSRuntime
from .types import ActionRecord, RuntimeAssessment


# ---------------------------------------------------------------------------
# Severity table (shared between RNOS decision values and CB reason strings)
# ---------------------------------------------------------------------------

_SEVERITY: dict[str, int] = {
    # RNOS PolicyDecision.value strings
    "allow": 0,
    "degrade": 1,
    "refuse": 2,
    # CircuitBreaker / AdaptiveCircuitBreaker reason strings
    "closed": 0,
    "half_open_probe": 1,
    "open_blocked": 2,
    "permanently_open": 2,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class HybridDecision:
    """Merged result from a single hybrid control evaluation."""

    decision: str           # "ALLOW", "DEGRADE", or "REFUSE"
    rnos_decision: str      # Raw RNOS decision (upper-cased), e.g. "DEGRADE"
    rnos_entropy: float
    rnos_trust: float
    cb_state: str           # Circuit breaker state string
    cb_reason: str          # CB reason from should_execute()
    cb_failure_rate: float  # Sliding-window failure rate (0.0 if not available)
    trigger_source: str     # "rnos", "cb", or "both"
    rnos_assessment: RuntimeAssessment  # Full RNOS assessment for downstream use


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class HybridController:
    """Composes RNOSRuntime and any circuit-breaker-compatible object.

    The circuit breaker must expose:
        tick() -> None
        should_execute() -> tuple[bool, str]   (allowed, reason)
        record_result(*, success: bool) -> None
        state -> str
        stats -> dict

    Both ``baselines.circuit_breaker.CircuitBreaker`` and
    ``baselines.adaptive_circuit_breaker.AdaptiveCircuitBreaker`` satisfy
    this interface.
    """

    def __init__(
        self,
        rnos_runtime: RNOSRuntime,
        circuit_breaker: Any,
    ) -> None:
        self.rnos = rnos_runtime
        self.cb = circuit_breaker

    # ------------------------------------------------------------------
    # Core interface (mirrors the CB tick/evaluate/record idiom)
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Advance the circuit breaker by one step."""
        self.cb.tick()

    def evaluate(self, action: ActionRecord) -> HybridDecision:
        """Evaluate *action* under both RNOS and CB; return merged decision.

        Both sub-systems are always queried.  The merged decision is the
        maximum-severity outcome between the two.
        """
        assessment = self.rnos.evaluate(action)
        _allowed, cb_reason = self.cb.should_execute()
        cb_stats = self.cb.stats
        return self._merge(assessment, cb_reason, cb_stats)

    def record_outcome(self, action: ActionRecord, *, success: bool) -> None:
        """Record the tool outcome to both RNOS and the circuit breaker."""
        self.rnos.record_outcome(action, success=success)
        self.cb.record_result(success=success)

    # ------------------------------------------------------------------
    # Merge logic
    # ------------------------------------------------------------------

    def _merge(
        self,
        assessment: RuntimeAssessment,
        cb_reason: str,
        cb_stats: dict[str, Any],
    ) -> HybridDecision:
        """Safety-first merge: max severity wins."""
        rnos_str = assessment.decision.value      # "allow" | "degrade" | "refuse"
        rnos_sev = _SEVERITY.get(rnos_str, 0)
        cb_sev = _SEVERITY.get(cb_reason, 0)
        max_sev = max(rnos_sev, cb_sev)

        if max_sev >= 2:
            decision = "REFUSE"
        elif max_sev == 1:
            decision = "DEGRADE"
        else:
            decision = "ALLOW"

        if rnos_sev > cb_sev:
            trigger_source = "rnos"
        elif cb_sev > rnos_sev:
            trigger_source = "cb"
        else:
            trigger_source = "both"

        # AdaptiveCircuitBreaker exposes "failure_rate"; basic CB does not.
        cb_failure_rate = float(cb_stats.get("failure_rate", 0.0))

        return HybridDecision(
            decision=decision,
            rnos_decision=rnos_str.upper(),
            rnos_entropy=assessment.entropy,
            rnos_trust=assessment.trust,
            cb_state=self.cb.state,
            cb_reason=cb_reason,
            cb_failure_rate=cb_failure_rate,
            trigger_source=trigger_source,
            rnos_assessment=assessment,
        )
