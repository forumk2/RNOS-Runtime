"""Controllers for the job scheduler control experiment.

Four controllers operate over SchedulerState:

1. RNOSSchedulerController     — entropy-based pre-execution gating (structural)
2. SlidingWindowCBController   — sliding-window failure-rate circuit breaker
3. HybridSchedulerController   — safety-first merge of RNOS + CB (dual-axis)
4. TriModalSchedulerController — safety-first merge of RNOS + CB + Persistence

The three control axes capture orthogonal failure geometries in a batch
job scheduling system:

    RNOS        — job graph expansion (active_jobs, total spawned, depth)
    CB          — failure density (short sliding window, fast response)
    Persistence — queue backlog drift (long window, slow-burn saturation)

Persistence is reused unchanged from experiments/common/persistence.py.
Its second update() signal ("rnos_entropy") is replaced with wait_time_trend
(units/step). entropy_floor is set to 2.0 so that any cycle where the queue
wait time increases by more than 2 units is counted as "above floor", enabling
detection of sustained queue saturation without structural explosion or elevated
failure density.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

from experiments.scheduler_control.scheduler_model import Decision, SchedulerState


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
# 1. RNOSSchedulerController
# ---------------------------------------------------------------------------

@dataclass
class RNOSSchedulerAssessment:
    decision: Decision
    entropy: float
    active_score: float
    spawned_score: float
    depth_score: float


class RNOSSchedulerController:
    """Entropy-based pre-execution gate for job scheduling cycles.

    Entropy components (all read from SchedulerState — no internal state)
    --------------------
    active_score  = min(log(active_jobs + 1) * 1.2, 5.0)
        Log-scale penalises exponential active-job growth; caps at 5.0.
        active=1→0.83, active=4→1.93, active=8→2.64, active=16→3.40,
        active=32→4.20, active=64→5.00 (cap).
    spawned_score = min(log(total_jobs_spawned + 1) * 0.5, 2.0)
        Log-scale penalises cumulative job volume; caps at 2.0.
    depth_score   = min(dependency_depth * 0.6, 4.0)
        Linear penalty for dependency graph depth; caps at 4.0 (depth ≥ 7).

    Total entropy cap ~11.0. Calibrated thresholds:
        DEGRADE = 8.0  (structural pressure high, may still recover)
        REFUSE  = 10.0 (job graph overloaded, halt dispatching)

    Signal geometry for dependency_explosion (active doubles, depth += 1/step):
        step 4: entropy ≈ 7.67  → ALLOW  (below DEGRADE)
        step 5: entropy ≈ 8.13  → DEGRADE
        step 6: entropy ≈ 9.80  → DEGRADE
        step 7: entropy ≈ 11.00 → REFUSE
    """

    def __init__(
        self,
        degrade_threshold: float = 8.0,
        refuse_threshold: float = 10.0,
    ) -> None:
        self.degrade_threshold = degrade_threshold
        self.refuse_threshold = refuse_threshold

    def evaluate(self, state: SchedulerState) -> RNOSSchedulerAssessment:
        active_score = min(math.log(state.active_jobs + 1) * 1.2, 5.0)
        spawned_score = min(math.log(state.total_jobs_spawned + 1) * 0.5, 2.0)
        depth_score = min(state.dependency_depth * 0.6, 4.0)

        entropy = active_score + spawned_score + depth_score

        if entropy >= self.refuse_threshold:
            decision = Decision.REFUSE
        elif entropy >= self.degrade_threshold:
            decision = Decision.DEGRADE
        else:
            decision = Decision.ALLOW

        return RNOSSchedulerAssessment(
            decision=decision,
            entropy=entropy,
            active_score=active_score,
            spawned_score=spawned_score,
            depth_score=depth_score,
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
    """Sliding-window failure-rate circuit breaker for scheduling outcomes.

    Trips (REFUSE) when the failure rate in the last `window_size` cycles
    exceeds `threshold`. Window must be full before any trip can fire.

    Signal geometry for failing_jobs_storm (F,F,F,S repeating):
        After step 5: window = [F,F,F,S,F] → rate = 4/5 = 0.80 > 0.60
        CB REFUSE fires at step 6 evaluation.
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
# 3. HybridSchedulerController
# ---------------------------------------------------------------------------

@dataclass
class HybridSchedulerAssessment:
    decision: Decision
    trigger_source: str       # "rnos" | "cb" | "both" | "none"
    rnos_decision: Decision
    rnos_entropy: float
    cb_decision: Decision
    cb_failure_rate: float
    cb_state: str


class HybridSchedulerController:
    """Safety-first merge of RNOSSchedulerController and SlidingWindowCBController.

    Merge rule: max severity wins.
        severity("allow")=0, severity("degrade")=1, severity("refuse")=2

    By construction, hybrid can never perform worse than the better sub-system.
    """

    def __init__(
        self,
        rnos: Optional[RNOSSchedulerController] = None,
        cb: Optional[SlidingWindowCBController] = None,
    ) -> None:
        self.rnos = rnos or RNOSSchedulerController()
        self.cb = cb or SlidingWindowCBController()

    def evaluate(self, state: SchedulerState) -> HybridSchedulerAssessment:
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

        return HybridSchedulerAssessment(
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
        self.rnos = RNOSSchedulerController(
            self.rnos.degrade_threshold, self.rnos.refuse_threshold
        )
        self.cb.reset()


# ---------------------------------------------------------------------------
# 4. TriModalSchedulerController
# ---------------------------------------------------------------------------

from experiments.common.persistence import PersistenceController  # noqa: E402


@dataclass
class TriModalSchedulerAssessment:
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


class TriModalSchedulerController:
    """Safety-first merge of RNOS + CB + Persistence for job scheduling.

    Extends HybridSchedulerController with a third control axis:
    PersistenceController. Merge rule: max severity across all three wins.

    Persistence axis adaptation
    ---------------------------
    The domain-agnostic PersistenceController is reused unchanged. Its
    second update() signal ("rnos_entropy") is replaced with wait_time_trend
    (units/step). The entropy_floor is set to 2.0 so that any cycle where
    the queue wait time increases by more than 2 units is counted as "above
    floor" — detecting sustained queue saturation without requiring structural
    explosion or elevated failure density.

    Persistence requires a full long window (default 10 steps) before raising
    any alert, guaranteeing no regression on fast-diverging scenarios where
    RNOS or CB are the correct primary signal.

    Signal geometry for queue_backlog_drift (alternating F/S, wait +3/step):
        After 10 steps: failure_rate=0.50, time_above_floor=0.90
        score = 0.7*0.50 + 0.3*0.90 = 0.35 + 0.27 = 0.62 → REFUSE at step 11
    """

    _SEVERITY: dict[str, int] = {"allow": 0, "degrade": 1, "refuse": 2}

    def __init__(
        self,
        rnos: Optional[RNOSSchedulerController] = None,
        cb: Optional[SlidingWindowCBController] = None,
        persistence: Optional[PersistenceController] = None,
    ) -> None:
        self.rnos = rnos or RNOSSchedulerController()
        self.cb = cb or SlidingWindowCBController()
        self.persistence = persistence or PersistenceController()
        self._last_wait_time_trend: float = 0.0

    def evaluate(self, state: SchedulerState) -> TriModalSchedulerAssessment:
        rnos_assessment = self.rnos.evaluate(state)
        cb_assessment = self.cb.evaluate()
        persist_assessment = self.persistence.evaluate()

        # Stash wait_time_trend for record_outcome (called after evaluate)
        self._last_wait_time_trend = state.wait_time_trend

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

        return TriModalSchedulerAssessment(
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
        # Pass wait_time_trend as the "entropy" signal to Persistence.
        # entropy_floor=2.0 means steps with trend > 2 units/step are "above floor".
        self.persistence.update(success, self._last_wait_time_trend)

    def reset(self) -> None:
        self.rnos = RNOSSchedulerController(
            self.rnos.degrade_threshold, self.rnos.refuse_threshold
        )
        self.cb.reset()
        self.persistence.reset()
        self._last_wait_time_trend = 0.0
