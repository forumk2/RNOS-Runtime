"""Controllers for DB control experiment.

Four controllers operate over QueryState:

1. RNOSDBController    — entropy-based pre-execution gating (structural complexity)
2. SlidingWindowCBController — sliding-window failure-rate circuit breaker
3. HybridDBController  — safety-first merge of RNOS + CB (dual-axis)
4. TriModalDBController — safety-first merge of RNOS + CB + Persistence (tri-axis)

Design mirrors the RNOS + AdaptiveCircuitBreaker hybrid from experiment_5_hybrid,
adapted to a database-query domain without pulling in the full RNOS stack.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

from experiments.db_control.query_model import Decision, QueryState


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

_SEVERITY: dict[str, int] = {
    "allow": 0,
    "degrade": 1,
    "refuse": 2,
}


def _to_severity(decision: Decision) -> int:
    return _SEVERITY[decision.value]


def _from_severity(sev: int) -> Decision:
    return [Decision.ALLOW, Decision.DEGRADE, Decision.REFUSE][sev]


# ---------------------------------------------------------------------------
# 1. RNOSDBController
# ---------------------------------------------------------------------------

@dataclass
class RNOSDBAssessment:
    decision: Decision
    entropy: float
    join_depth_score: float
    cost_score: float
    cumulative_cost_score: float


class RNOSDBController:
    """Entropy-based pre-execution gate for database queries.

    Entropy components
    ------------------
    join_depth_score     = min(join_depth * 1.0, 5.0)
        Penalises deep join trees linearly; caps at 5.0.
    cost_score           = min(log2(max(estimated_cost, 1.0)) * 0.5, 4.0)
        Log-scale penalises exponentially growing costs; caps at 4.0.
    cumulative_cost_score = min(cumulative_cost / 100.0, 2.0)
        Non-resetting; accumulates monotonically; caps at 2.0.

    Total entropy cap ~11.0. Calibrated thresholds:
        DEGRADE = 8.0  (structural complexity is high but may be recoverable)
        REFUSE  = 10.0 (structural complexity indicates certain overload)
    """

    def __init__(
        self,
        degrade_threshold: float = 8.0,
        refuse_threshold: float = 10.0,
    ) -> None:
        self.degrade_threshold = degrade_threshold
        self.refuse_threshold = refuse_threshold
        self._cumulative_cost: float = 0.0

    def evaluate(self, state: QueryState) -> RNOSDBAssessment:
        """Compute entropy and return decision for this query step."""
        join_depth_score = min(state.join_depth * 1.0, 5.0)
        cost_score = min(math.log2(max(state.estimated_cost, 1.0)) * 0.5, 4.0)
        cumulative_cost_score = min(self._cumulative_cost / 100.0, 2.0)

        entropy = join_depth_score + cost_score + cumulative_cost_score

        if entropy >= self.refuse_threshold:
            decision = Decision.REFUSE
        elif entropy >= self.degrade_threshold:
            decision = Decision.DEGRADE
        else:
            decision = Decision.ALLOW

        return RNOSDBAssessment(
            decision=decision,
            entropy=entropy,
            join_depth_score=join_depth_score,
            cost_score=cost_score,
            cumulative_cost_score=cumulative_cost_score,
        )

    def record_outcome(self, state: QueryState) -> None:
        """Update cumulative cost after execution."""
        self._cumulative_cost += state.estimated_cost

    def reset(self) -> None:
        self._cumulative_cost = 0.0


# ---------------------------------------------------------------------------
# 2. SlidingWindowCBController
# ---------------------------------------------------------------------------

@dataclass
class CBAssessment:
    decision: Decision
    state: str           # "closed" | "open"
    failure_rate: float
    window_size: int


class SlidingWindowCBController:
    """Sliding-window failure-rate circuit breaker.

    Trips (REFUSE) when the failure rate in the last `window_size` executions
    exceeds `threshold`. Stays closed (ALLOW) otherwise.

    Unlike AdaptiveCircuitBreaker, this controller has no backoff or half-open
    probe — it is intentionally simple to illustrate the CB mechanism in isolation.
    The window must be full before any trip can fire (requires >= window_size execs).
    """

    def __init__(self, window_size: int = 5, threshold: float = 0.6) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self._window: deque[bool] = deque(maxlen=window_size)
        self._tripped = False

    @property
    def failure_rate(self) -> float:
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    def evaluate(self) -> CBAssessment:
        """Return REFUSE if window is full and failure rate exceeds threshold."""
        if self._tripped:
            return CBAssessment(
                decision=Decision.REFUSE,
                state="open",
                failure_rate=self.failure_rate,
                window_size=len(self._window),
            )

        rate = self.failure_rate
        if len(self._window) >= self.window_size and rate > self.threshold:
            self._tripped = True
            return CBAssessment(
                decision=Decision.REFUSE,
                state="open",
                failure_rate=rate,
                window_size=len(self._window),
            )

        return CBAssessment(
            decision=Decision.ALLOW,
            state="closed",
            failure_rate=rate,
            window_size=len(self._window),
        )

    def record_outcome(self, success: bool) -> None:
        """Record execution result into the sliding window."""
        self._window.append(not success)  # True = failure

    def reset(self) -> None:
        self._window.clear()
        self._tripped = False


# ---------------------------------------------------------------------------
# 3. HybridDBController
# ---------------------------------------------------------------------------

@dataclass
class HybridAssessment:
    decision: Decision
    trigger_source: str          # "rnos" | "cb" | "both" | "none"
    rnos_decision: Decision
    rnos_entropy: float
    cb_decision: Decision
    cb_failure_rate: float
    cb_state: str


class HybridDBController:
    """Safety-first merge of RNOSDBController and SlidingWindowCBController.

    Merge rule: max severity wins.
        severity("allow")=0, severity("degrade")=1, severity("refuse")=2

    trigger_source attribution:
        "rnos"  — RNOS raised severity, CB did not
        "cb"    — CB raised severity, RNOS did not
        "both"  — both raised severity equally
        "none"  — both ALLOW (no intervention)

    By construction, hybrid can never perform worse than the better sub-system.
    """

    def __init__(
        self,
        rnos: Optional[RNOSDBController] = None,
        cb: Optional[SlidingWindowCBController] = None,
    ) -> None:
        self.rnos = rnos or RNOSDBController()
        self.cb = cb or SlidingWindowCBController()

    def evaluate(self, state: QueryState) -> HybridAssessment:
        rnos_assessment = self.rnos.evaluate(state)
        cb_assessment = self.cb.evaluate()

        rnos_sev = _to_severity(rnos_assessment.decision)
        cb_sev = _to_severity(cb_assessment.decision)

        merged_sev = max(rnos_sev, cb_sev)
        merged_decision = _from_severity(merged_sev)

        if rnos_sev == 0 and cb_sev == 0:
            trigger_source = "none"
        elif rnos_sev > cb_sev:
            trigger_source = "rnos"
        elif cb_sev > rnos_sev:
            trigger_source = "cb"
        else:
            trigger_source = "both"

        return HybridAssessment(
            decision=merged_decision,
            trigger_source=trigger_source,
            rnos_decision=rnos_assessment.decision,
            rnos_entropy=rnos_assessment.entropy,
            cb_decision=cb_assessment.decision,
            cb_failure_rate=cb_assessment.failure_rate,
            cb_state=cb_assessment.state,
        )

    def record_outcome(self, state: QueryState, *, success: bool) -> None:
        self.rnos.record_outcome(state)
        self.cb.record_outcome(success)

    def reset(self) -> None:
        self.rnos.reset()
        self.cb.reset()


# ---------------------------------------------------------------------------
# 4. TriModalDBController
# ---------------------------------------------------------------------------

from experiments.common.persistence import PersistenceAssessment, PersistenceController  # noqa: E402


@dataclass
class TriModalAssessment:
    decision: Decision
    trigger_source: str        # "rnos" | "cb" | "persistence" | "none"
    rnos_decision: Decision
    rnos_entropy: float
    cb_decision: Decision
    cb_failure_rate: float
    cb_state: str
    persist_decision: str      # "allow" | "degrade" | "refuse"
    persist_score: float
    persist_failure_rate: float
    persist_above_floor: float
    persist_window_fill: int


class TriModalDBController:
    """Safety-first merge of RNOS + CB + Persistence for DB queries.

    Extends HybridDBController with a third control axis: the PersistenceController.
    Merge rule: max(severity(rnos), severity(cb), severity(persistence)) wins.

    trigger_source attribution: the highest-severity source. If multiple sources
    tie at the winning severity, they are joined with "+" (e.g. "rnos+cb").
    "none" when all three return ALLOW.

    Persistence requires a full long window (default 10 steps) before it can
    raise any alert — ensuring it cannot fire before RNOS or CB on fast-diverging
    scenarios. This guarantees no regression on existing scenarios.
    """

    _SEVERITY: dict[str, int] = {"allow": 0, "degrade": 1, "refuse": 2}

    def __init__(
        self,
        rnos: Optional[RNOSDBController] = None,
        cb: Optional[SlidingWindowCBController] = None,
        persistence: Optional[PersistenceController] = None,
    ) -> None:
        self.rnos = rnos or RNOSDBController()
        self.cb = cb or SlidingWindowCBController()
        self.persistence = persistence or PersistenceController()
        self._last_rnos_entropy: float = 0.0

    def evaluate(self, state: QueryState) -> TriModalAssessment:
        rnos_assessment = self.rnos.evaluate(state)
        cb_assessment = self.cb.evaluate()
        persist_assessment = self.persistence.evaluate()

        self._last_rnos_entropy = rnos_assessment.entropy

        rnos_sev = _to_severity(rnos_assessment.decision)
        cb_sev = _to_severity(cb_assessment.decision)
        persist_sev = self._SEVERITY[persist_assessment.decision]

        merged_sev = max(rnos_sev, cb_sev, persist_sev)
        merged_decision = _from_severity(merged_sev)

        # Attribute trigger source
        if merged_sev == 0:
            trigger_source = "none"
        else:
            winners = []
            if rnos_sev == merged_sev:
                winners.append("rnos")
            if cb_sev == merged_sev:
                winners.append("cb")
            if persist_sev == merged_sev:
                winners.append("persistence")
            trigger_source = "+".join(winners)

        return TriModalAssessment(
            decision=merged_decision,
            trigger_source=trigger_source,
            rnos_decision=rnos_assessment.decision,
            rnos_entropy=rnos_assessment.entropy,
            cb_decision=cb_assessment.decision,
            cb_failure_rate=cb_assessment.failure_rate,
            cb_state=cb_assessment.state,
            persist_decision=persist_assessment.decision,
            persist_score=persist_assessment.score,
            persist_failure_rate=persist_assessment.rolling_failure_rate,
            persist_above_floor=persist_assessment.time_above_entropy_floor,
            persist_window_fill=persist_assessment.window_fill,
        )

    def record_outcome(self, state: QueryState, *, success: bool) -> None:
        self.rnos.record_outcome(state)
        self.cb.record_outcome(success)
        self.persistence.update(success, self._last_rnos_entropy)

    def reset(self) -> None:
        self.rnos.reset()
        self.cb.reset()
        self.persistence.reset()
        self._last_rnos_entropy = 0.0
