"""Controllers for CI control experiment.

Four controllers operate over PipelineState:

1. RNOSCIController        — entropy-based pre-execution gating (structural complexity)
2. SlidingWindowCBController — sliding-window failure-rate circuit breaker
3. HybridCIController      — safety-first merge of RNOS + CB (dual-axis)
4. TriModalCIController    — safety-first merge of RNOS + CB + Persistence (tri-axis)

The RNOS controller is stateless: it reads all signal fields directly from
PipelineState, which carries pre-computed structural context (active_jobs,
total_jobs_spawned, retry_count). The CB controller is stateful, maintaining
a sliding window of execution outcomes.

Design mirrors the db_control experiment, adapted to CI pipeline domain.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

from experiments.ci_control.pipeline_model import Decision, PipelineState


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
# 1. RNOSCIController
# ---------------------------------------------------------------------------

@dataclass
class RNOSCIAssessment:
    decision: Decision
    entropy: float
    active_jobs_score: float
    spawned_score: float
    retry_score: float


class RNOSCIController:
    """Entropy-based pre-execution gate for CI pipeline ticks.

    Entropy components (all read from PipelineState — no internal state)
    --------------------
    active_jobs_score  = min(active_jobs * 0.4, 5.0)
        Penalises parallel fanout linearly; caps at 5.0 (representing
        extreme parallelism beyond which additional jobs add constant risk).
    spawned_score      = min(log2(max(total_jobs_spawned, 1)) * 0.5, 3.0)
        Log-scale penalises cumulative pipeline expansion; caps at 3.0.
    retry_score        = min(retry_count * 0.8, 3.0)
        Penalises retry accumulation linearly; caps at 3.0.

    Total entropy cap ~11.0. Calibrated thresholds:
        DEGRADE = 8.0  (high structural pressure, may still recover)
        REFUSE  = 10.0 (structural overload, halt pipeline)
    """

    def __init__(
        self,
        degrade_threshold: float = 8.0,
        refuse_threshold: float = 10.0,
    ) -> None:
        self.degrade_threshold = degrade_threshold
        self.refuse_threshold = refuse_threshold

    def evaluate(self, state: PipelineState) -> RNOSCIAssessment:
        active_jobs_score = min(state.active_jobs * 0.4, 5.0)
        spawned_score = min(math.log2(max(state.total_jobs_spawned, 1)) * 0.5, 3.0)
        retry_score = min(state.retry_count * 0.8, 3.0)

        entropy = active_jobs_score + spawned_score + retry_score

        if entropy >= self.refuse_threshold:
            decision = Decision.REFUSE
        elif entropy >= self.degrade_threshold:
            decision = Decision.DEGRADE
        else:
            decision = Decision.ALLOW

        return RNOSCIAssessment(
            decision=decision,
            entropy=entropy,
            active_jobs_score=active_jobs_score,
            spawned_score=spawned_score,
            retry_score=retry_score,
        )


# ---------------------------------------------------------------------------
# 2. SlidingWindowCBController
# ---------------------------------------------------------------------------

@dataclass
class CBAssessment:
    decision: Decision
    state: str          # "closed" | "open"
    failure_rate: float
    window_fill: int    # number of entries in window so far


class SlidingWindowCBController:
    """Sliding-window failure-rate circuit breaker for CI job outcomes.

    Trips (REFUSE) when the failure rate in the last `window_size` executions
    exceeds `threshold`. Window must be full before any trip can fire.

    Same design as db_control.SlidingWindowCBController — intentionally simple
    to illustrate the CB mechanism in isolation.
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
# 3. HybridCIController
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


class HybridCIController:
    """Safety-first merge of RNOSCIController and SlidingWindowCBController.

    Merge rule: max severity wins.
        severity("allow")=0, severity("degrade")=1, severity("refuse")=2

    trigger_source attribution:
        "rnos"  — RNOS raised severity, CB did not
        "cb"    — CB raised severity, RNOS did not
        "both"  — both raised severity equally (and above ALLOW)
        "none"  — both ALLOW

    By construction, hybrid can never perform worse than the best sub-system.
    """

    def __init__(
        self,
        rnos: Optional[RNOSCIController] = None,
        cb: Optional[SlidingWindowCBController] = None,
    ) -> None:
        self.rnos = rnos or RNOSCIController()
        self.cb = cb or SlidingWindowCBController()

    def evaluate(self, state: PipelineState) -> HybridAssessment:
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
        self.rnos = RNOSCIController(
            self.rnos.degrade_threshold, self.rnos.refuse_threshold
        )
        self.cb.reset()


# ---------------------------------------------------------------------------
# 4. TriModalCIController
# ---------------------------------------------------------------------------

from experiments.common.persistence import PersistenceAssessment, PersistenceController  # noqa: E402


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


class TriModalCIController:
    """Safety-first merge of RNOS + CB + Persistence for CI pipelines.

    Extends HybridCIController with a third control axis: PersistenceController.
    Merge rule: max(severity(rnos), severity(cb), severity(persistence)) wins.

    Persistence uses a long window (default 10 steps) that must be full before
    alerting — guaranteeing no regression on fast-diverging existing scenarios.
    """

    _SEVERITY: dict[str, int] = {"allow": 0, "degrade": 1, "refuse": 2}

    def __init__(
        self,
        rnos: Optional[RNOSCIController] = None,
        cb: Optional[SlidingWindowCBController] = None,
        persistence: Optional[PersistenceController] = None,
    ) -> None:
        self.rnos = rnos or RNOSCIController()
        self.cb = cb or SlidingWindowCBController()
        self.persistence = persistence or PersistenceController()
        self._last_rnos_entropy: float = 0.0

    def evaluate(self, state: PipelineState) -> TriModalAssessment:
        rnos_assessment = self.rnos.evaluate(state)
        cb_assessment = self.cb.evaluate()
        persist_assessment = self.persistence.evaluate()

        self._last_rnos_entropy = rnos_assessment.entropy

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

    def record_outcome(self, state: PipelineState, *, success: bool) -> None:
        self.cb.record_outcome(success)
        self.persistence.update(success, self._last_rnos_entropy)

    def reset(self) -> None:
        self.rnos = RNOSCIController(
            self.rnos.degrade_threshold, self.rnos.refuse_threshold
        )
        self.cb.reset()
        self.persistence.reset()
        self._last_rnos_entropy = 0.0
