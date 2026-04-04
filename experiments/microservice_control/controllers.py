"""Controllers for microservice control experiment.

Four controllers operate over RequestState:

1. RNOSMSController        — entropy-based pre-execution gating (structural complexity)
2. SlidingWindowCBController — sliding-window failure-rate circuit breaker
3. HybridMSController      — safety-first merge of RNOS + CB (dual-axis)
4. TriModalMSController    — safety-first merge of RNOS + CB + Persistence (tri-axis)

Design mirrors db_control and ci_control experiments, adapted to the
distributed microservice domain. The three control axes capture orthogonal
failure geometries:

    RNOS        — request graph expansion (fanout, depth, cumulative volume)
    CB          — failure density (short sliding window, fast response)
    Persistence — latency drift (long window, slow-burn degradation)

The Persistence controller is reused unchanged from experiments/common/persistence.py.
Its "rnos_entropy" input is replaced with latency_trend (ms/step), and its
entropy_floor is set to 10.0 so that steps with trend > 10 ms/step are counted
as "above floor". This allows it to detect sustained latency drift without any
structural explosion or failure spike.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

from experiments.microservice_control.service_model import Decision, RequestState


# ---------------------------------------------------------------------------
# Severity helpers (shared across all controllers)
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
# 1. RNOSMSController
# ---------------------------------------------------------------------------

@dataclass
class RNOSMSAssessment:
    decision: Decision
    entropy: float
    fanout_score: float
    depth_score: float
    requests_score: float


class RNOSMSController:
    """Entropy-based pre-execution gate for microservice request cycles.

    Entropy components (all read from RequestState — no internal state)
    --------------------
    fanout_score   = min(log2(max(fanout, 1)) * 1.2, 5.0)
        Log-scale penalises exponential fanout growth; caps at 5.0.
        fanout=1→0.0, fanout=2→1.2, fanout=4→2.4, fanout=8→3.6,
        fanout=16→4.8, fanout=32→5.0 (cap).
    depth_score    = min(depth * 0.6, 4.0)
        Linear penalty for call-chain depth; caps at 4.0 (depth ≥ 7).
    requests_score = min(log2(max(total_requests, 1)) * 0.5, 2.0)
        Log-scale penalises cumulative request volume; caps at 2.0.

    Total entropy cap ~11.0. Calibrated thresholds:
        DEGRADE = 8.0  (structural pressure is high, may still recover)
        REFUSE  = 10.0 (structural overload, halt request processing)

    Signal geometry for fanout_cascade (fanout doubles, depth += 1 each step):
        step 4: entropy ≈ 7.95  → ALLOW  (just below DEGRADE)
        step 5: entropy ≈ 9.80  → DEGRADE
        step 6: entropy ≈ 10.6  → REFUSE
    """

    def __init__(
        self,
        degrade_threshold: float = 8.0,
        refuse_threshold: float = 10.0,
    ) -> None:
        self.degrade_threshold = degrade_threshold
        self.refuse_threshold = refuse_threshold

    def evaluate(self, state: RequestState) -> RNOSMSAssessment:
        fanout_score = min(math.log2(max(state.fanout, 1)) * 1.2, 5.0)
        depth_score = min(state.depth * 0.6, 4.0)
        requests_score = min(math.log2(max(state.total_requests, 1)) * 0.5, 2.0)

        entropy = fanout_score + depth_score + requests_score

        if entropy >= self.refuse_threshold:
            decision = Decision.REFUSE
        elif entropy >= self.degrade_threshold:
            decision = Decision.DEGRADE
        else:
            decision = Decision.ALLOW

        return RNOSMSAssessment(
            decision=decision,
            entropy=entropy,
            fanout_score=fanout_score,
            depth_score=depth_score,
            requests_score=requests_score,
        )


# ---------------------------------------------------------------------------
# 2. SlidingWindowCBController
# ---------------------------------------------------------------------------

@dataclass
class CBAssessment:
    decision: Decision
    state: str          # "closed" | "open"
    failure_rate: float
    window_fill: int


class SlidingWindowCBController:
    """Sliding-window failure-rate circuit breaker for request outcomes.

    Trips (REFUSE) when the failure rate in the last `window_size` request
    cycles exceeds `threshold`. Window must be full before any trip can fire
    (requires >= window_size recorded outcomes).

    Same design as db_control and ci_control — intentionally simple to
    illustrate the CB mechanism in isolation.

    Signal geometry for retry_storm (F,F,F,S repeating):
        After step 5 window = [F,F,F,S,F] → failure_rate = 4/5 = 0.80 > 0.60
        CB REFUSE fires at step 6 (before execution of step 6).
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
                window_fill=len(self._window),
            )

        rate = self.failure_rate
        if len(self._window) >= self.window_size and rate > self.threshold:
            self._tripped = True
            return CBAssessment(
                decision=Decision.REFUSE,
                state="open",
                failure_rate=rate,
                window_fill=len(self._window),
            )

        return CBAssessment(
            decision=Decision.ALLOW,
            state="closed",
            failure_rate=rate,
            window_fill=len(self._window),
        )

    def record_outcome(self, success: bool) -> None:
        self._window.append(not success)  # True = failure

    def reset(self) -> None:
        self._window.clear()
        self._tripped = False


# ---------------------------------------------------------------------------
# 3. HybridMSController
# ---------------------------------------------------------------------------

@dataclass
class HybridAssessment:
    decision: Decision
    trigger_source: str       # "rnos" | "cb" | "both" | "none"
    rnos_decision: Decision
    rnos_entropy: float
    cb_decision: Decision
    cb_failure_rate: float
    cb_state: str


class HybridMSController:
    """Safety-first merge of RNOSMSController and SlidingWindowCBController.

    Merge rule: max severity wins.
        severity("allow")=0, severity("degrade")=1, severity("refuse")=2

    trigger_source attribution:
        "rnos"  — RNOS raised severity, CB did not
        "cb"    — CB raised severity, RNOS did not
        "both"  — both raised severity equally (and above ALLOW)
        "none"  — both ALLOW

    By construction, hybrid can never perform worse than the better sub-system.
    """

    def __init__(
        self,
        rnos: Optional[RNOSMSController] = None,
        cb: Optional[SlidingWindowCBController] = None,
    ) -> None:
        self.rnos = rnos or RNOSMSController()
        self.cb = cb or SlidingWindowCBController()

    def evaluate(self, state: RequestState) -> HybridAssessment:
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

    def record_outcome(self, success: bool) -> None:
        self.cb.record_outcome(success)

    def reset(self) -> None:
        self.rnos = RNOSMSController(
            self.rnos.degrade_threshold, self.rnos.refuse_threshold
        )
        self.cb.reset()


# ---------------------------------------------------------------------------
# 4. TriModalMSController
# ---------------------------------------------------------------------------

from experiments.common.persistence import PersistenceController  # noqa: E402


@dataclass
class TriModalAssessment:
    decision: Decision
    trigger_source: str        # "rnos" | "cb" | "persistence" | "none" | combinations
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


class TriModalMSController:
    """Safety-first merge of RNOS + CB + Persistence for microservice requests.

    Extends HybridMSController with a third control axis: PersistenceController.
    Merge rule: max(severity(rnos), severity(cb), severity(persistence)) wins.

    Persistence axis adaptation
    ---------------------------
    The domain-agnostic PersistenceController is reused unchanged. Its second
    update() signal (nominally "rnos_entropy") is replaced with latency_trend
    (ms/step). The entropy_floor is set to 10.0 so that any step with a latency
    increase > 10 ms/step is counted as "above floor" — detecting sustained
    latency drift without requiring structural explosion or high failure density.

    Persistence requires a full long window (default 10 steps) before raising
    any alert — guaranteeing no regression on fast-diverging scenarios where
    RNOS or CB are the correct primary signal.

    Signal geometry for latency_drift (alternating F/S, latency +20ms/step):
        After 10 steps: failure_rate=0.50, time_above_floor=0.90
        score = 0.7*0.50 + 0.3*0.90 = 0.35 + 0.27 = 0.62 → REFUSE at step 11
    """

    _SEVERITY: dict[str, int] = {"allow": 0, "degrade": 1, "refuse": 2}

    def __init__(
        self,
        rnos: Optional[RNOSMSController] = None,
        cb: Optional[SlidingWindowCBController] = None,
        persistence: Optional[PersistenceController] = None,
    ) -> None:
        self.rnos = rnos or RNOSMSController()
        self.cb = cb or SlidingWindowCBController()
        self.persistence = persistence or PersistenceController()
        self._last_latency_trend: float = 0.0

    def evaluate(self, state: RequestState) -> TriModalAssessment:
        rnos_assessment = self.rnos.evaluate(state)
        cb_assessment = self.cb.evaluate()
        persist_assessment = self.persistence.evaluate()

        # Stash latency_trend for record_outcome (called after evaluate)
        self._last_latency_trend = state.latency_trend

        rnos_sev = _to_severity(rnos_assessment.decision)
        cb_sev = _to_severity(cb_assessment.decision)
        persist_sev = self._SEVERITY[persist_assessment.decision]

        merged_sev = max(rnos_sev, cb_sev, persist_sev)
        merged_decision = _from_severity(merged_sev)

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

    def record_outcome(self, success: bool) -> None:
        self.cb.record_outcome(success)
        # Pass latency_trend as the "entropy" signal to Persistence.
        # entropy_floor=10.0 means steps with trend > 10 ms/step are "above floor".
        self.persistence.update(success, self._last_latency_trend)

    def reset(self) -> None:
        self.rnos = RNOSMSController(
            self.rnos.degrade_threshold, self.rnos.refuse_threshold
        )
        self.cb.reset()
        self.persistence.reset()
        self._last_latency_trend = 0.0
