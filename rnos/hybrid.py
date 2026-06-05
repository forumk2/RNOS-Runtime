"""Hybrid controller composing RNOS and a circuit breaker.

Coupled-detector merge strategy (2026-06-04 revision)
------------------------------------------------------
Three orthogonal detectors are evaluated independently and then combined:

  1. **RNOS entropy**   — detects burst complexity via cumulative instability.
  2. **CB failure_rate** — detects sustained high failure density in a sliding
     window, even when bursts are small (CB strength).
  3. **Coherence Λ proxy** — detects planner↔executor desynchronisation:
     tracks a sliding window of per-step λ_t values derived from execution
     outcome and RNOS allow/block decisions.

Coupling rules
--------------
(a) **Threshold lowering** — elevated CB failure_rate or Λ-collapse
    lowers RNOS's effective REFUSE threshold so that a sustained background
    failure rate alone can tip a marginally-elevated RNOS score into REFUSE.

(b) **Combination REFUSE** — fire REFUSE on combinations that no single
    detector would trip alone:
    * RNOS at DEGRADE AND CB blocked/elevated  → REFUSE
    * CB failure_rate > 0.30 AND coherence Λ critical  → REFUSE
    * RNOS at DEGRADE AND coherence Λ collapse  → REFUSE
    * All three detectors simultaneously elevated  → REFUSE

This lets the hybrid strictly dominate both sub-systems on
``cascading_burst`` (RNOS strength) and ``distributed_low_rate``
(CB strength) simultaneously, which the simple max-severity union
cannot achieve when scenarios stress different detectors.

Decision severity mapping
-------------------------
    ALLOW / closed          → 0
    DEGRADE / half_open     → 1
    REFUSE / open / perm    → 2
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .policy import PolicyConfig
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

# Coherence proxy window: number of executed steps to smooth Λ over.
_COHERENCE_WINDOW = 8

# Threshold-lowering parameters.
# CB failure-rate band: ramps from 0 at _RATE_LO to 1.0 at _RATE_HI.
_RATE_LO: float = 0.25
_RATE_HI: float = 0.75
# Maximum fractional reduction to RNOS refuse/degrade thresholds from rate.
_RATE_DISCOUNT: float = 0.30
# Additional fractional reduction when coherence is degraded.
_COHERENCE_DISCOUNT: float = 0.15
# CB failure rate above which the combination-REFUSE rule is eligible.
_COMBO_RATE_THRESHOLD: float = 0.30


# ---------------------------------------------------------------------------
# Internal coherence-proxy record
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _StepRecord:
    rnos_allowed: bool          # True if RNOS said ALLOW or DEGRADE
    success: bool | None = None  # filled in by record_outcome


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class HybridDecision:
    """Merged result from a single hybrid control evaluation."""

    decision: str                # "ALLOW", "DEGRADE", or "REFUSE"
    rnos_decision: str           # Raw RNOS decision (upper-cased)
    rnos_entropy: float
    rnos_trust: float
    cb_state: str                # Circuit breaker state string
    cb_reason: str               # CB reason from should_execute()
    cb_failure_rate: float       # Sliding-window failure rate
    trigger_source: str          # "rnos", "cb", "coherence", "combo", or "both"
    rnos_assessment: RuntimeAssessment
    coherence_lambda: float = field(default=1.0)


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
        # Rolling buffer of per-step coherence records for Λ proxy.
        self._step_records: deque[_StepRecord] = deque(maxlen=_COHERENCE_WINDOW)

    # ------------------------------------------------------------------
    # Core interface (mirrors the CB tick/evaluate/record idiom)
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Advance the circuit breaker by one step."""
        self.cb.tick()

    def evaluate(self, action: ActionRecord) -> HybridDecision:
        """Evaluate *action* under RNOS, CB, and coherence; return merged decision.

        All three detectors are always queried.  The merged decision uses
        the coupled merge strategy described in the module docstring.
        """
        assessment = self.rnos.evaluate(action)
        _allowed, cb_reason = self.cb.should_execute()
        cb_stats = self.cb.stats

        # Append a step record for the coherence proxy; outcome filled later.
        rnos_allowed = assessment.decision.value in {"allow", "degrade"}
        self._step_records.append(_StepRecord(rnos_allowed=rnos_allowed))

        return self._coupled_merge(assessment, cb_reason, cb_stats)

    def record_outcome(self, action: ActionRecord, *, success: bool) -> None:
        """Record the tool outcome to both RNOS and the circuit breaker."""
        self.rnos.record_outcome(action, success=success)
        self.cb.record_result(success=success)
        # Back-fill the outcome on the latest step record.
        if self._step_records:
            self._step_records[-1].success = success

    # ------------------------------------------------------------------
    # Coherence proxy
    # ------------------------------------------------------------------

    def _coherence_lambda_proxy(self) -> float:
        """Simplified Λ proxy over the recent executed-step buffer.

        For each step where an outcome was recorded:
            r_t = (s_pe=1  +  s_pg  +  s_pt=1  +  s_et) / 4
            h_t = 0.35 × f_t  +  0.25 × b_t
            λ_t = r_t / (1 + h_t)

        s_pe=1 (planner intent is assumed for all executed steps),
        s_pt=1 (tool produced a result when executed),
        s_et = 1 if success, 0 otherwise,
        s_pg = 1 if RNOS allowed, 0 if RNOS blocked.

        Returns the mean λ_t over the window.  Falls back to 1.0 (healthy)
        when fewer than 2 recorded outcomes are available.
        """
        records = [r for r in self._step_records if r.success is not None]
        if len(records) < 2:
            return 1.0

        lambdas: list[float] = []
        for rec in records:
            s_et = 1.0 if rec.success else 0.0
            s_pg = 1.0 if rec.rnos_allowed else 0.0
            r_t = (1.0 + s_pg + 1.0 + s_et) / 4.0
            f_t = 0.0 if rec.success else 1.0
            b_t = 0.0 if rec.rnos_allowed else 1.0
            h_t = 0.35 * f_t + 0.25 * b_t
            lambdas.append(r_t / (1.0 + h_t))

        return sum(lambdas) / len(lambdas)

    # ------------------------------------------------------------------
    # Coupled merge logic
    # ------------------------------------------------------------------

    def _coupled_merge(
        self,
        assessment: RuntimeAssessment,
        cb_reason: str,
        cb_stats: dict[str, Any],
    ) -> HybridDecision:
        """Coupled merge over RNOS entropy, CB failure_rate, and coherence Λ."""
        rnos_str = assessment.decision.value       # "allow" | "degrade" | "refuse"
        rnos_sev = _SEVERITY.get(rnos_str, 0)
        cb_sev = _SEVERITY.get(cb_reason, 0)
        cb_failure_rate = float(cb_stats.get("failure_rate", 0.0))
        lambda_proxy = round(self._coherence_lambda_proxy(), 3)
        rnos_entropy = assessment.entropy

        # --- Coherence regime (0=resonant, 1=critical, 2=collapse) -----------
        if lambda_proxy < 0.20:
            coherence_sev = 2
        elif lambda_proxy < 0.45:
            coherence_sev = 1
        else:
            coherence_sev = 0

        # --- (a) Threshold lowering ------------------------------------------
        # Elevated CB failure_rate or Λ-collapse lowers RNOS's effective
        # REFUSE/DEGRADE threshold, so that RNOS tips into REFUSE when it
        # is in the DEGRADE zone AND the background failure context is bad.
        cfg = (
            self.rnos._policy_config
            if self.rnos._policy_config is not None
            else PolicyConfig()
        )
        base_refuse = cfg.refuse_entropy
        base_degrade = cfg.degrade_entropy

        rate_pressure = min(
            1.0,
            max(0.0, (cb_failure_rate - _RATE_LO) / (_RATE_HI - _RATE_LO)),
        )
        coherence_discount = _COHERENCE_DISCOUNT if coherence_sev >= 1 else 0.0

        effective_refuse = base_refuse * (
            1.0 - _RATE_DISCOUNT * rate_pressure - coherence_discount
        )
        effective_degrade = base_degrade * (1.0 - 0.25 * rate_pressure)

        if rnos_entropy >= effective_refuse:
            rnos_sev = max(rnos_sev, 2)
        elif rnos_entropy >= effective_degrade:
            rnos_sev = max(rnos_sev, 1)

        # --- (b) Combination REFUSE -----------------------------------------
        # Fire on combinations that no single detector would trip alone.
        combo_refuse = (
            (rnos_sev >= 1 and cb_sev >= 1)                         # both degraded
            or (cb_failure_rate > _COMBO_RATE_THRESHOLD and coherence_sev >= 1)
            or (rnos_sev >= 1 and coherence_sev >= 2)               # entropy + collapse
            or (rnos_sev >= 1 and cb_failure_rate > _COMBO_RATE_THRESHOLD
                and coherence_sev >= 1)                             # all three elevated
        )

        # Final severity
        max_sev = max(rnos_sev, cb_sev, coherence_sev)
        if combo_refuse:
            max_sev = max(max_sev, 2)

        if max_sev >= 2:
            decision = "REFUSE"
        elif max_sev == 1:
            decision = "DEGRADE"
        else:
            decision = "ALLOW"

        # --- Trigger attribution --------------------------------------------
        if combo_refuse and max_sev >= 2:
            trigger_source = "combo"
        elif rnos_sev >= max_sev and rnos_sev > cb_sev and rnos_sev > coherence_sev:
            trigger_source = "rnos"
        elif cb_sev >= max_sev and cb_sev > rnos_sev and cb_sev > coherence_sev:
            trigger_source = "cb"
        elif coherence_sev >= max_sev and coherence_sev > 0:
            trigger_source = "coherence"
        else:
            trigger_source = "both"

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
            coherence_lambda=lambda_proxy,
        )
