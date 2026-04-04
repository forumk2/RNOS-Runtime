"""Persistence / Drift controller — third control axis.

Captures long-term, low-intensity instability that:
  * does not trigger RNOS  (no structural explosion)
  * does not trigger CB    (failure density below short-window threshold)
  * persists over time     (sustained signal across a long observation window)

This is a domain-agnostic controller. It receives two signals per step:
  - success: bool         (outcome of the execution step)
  - rnos_entropy: float   (entropy computed by the domain's RNOS controller)

Both signals are fed via update() AFTER the step executes. evaluate() is
called BEFORE the step executes, using the window accumulated so far.
The window must be full (window_size steps observed) before any decision
other than ALLOW can be returned. This prevents early false positives.

Persistence Score
-----------------
score = 0.7 * rolling_failure_rate + 0.3 * time_above_entropy_floor

  rolling_failure_rate    : fraction of window steps that failed
  time_above_entropy_floor: fraction of window steps where rnos_entropy
                            exceeded entropy_floor (default 3.0)

Thresholds (score-based, applied only when window is full):
  DEGRADE : score >= 0.30  (persistent low-grade instability)
  REFUSE  : score >= 0.50  (sustained instability requiring intervention)

Signal geometry
---------------
For a 50% alternating failure pattern (F,S,F,S...):
  rolling_failure_rate    = 0.50
  time_above_entropy_floor ~ 0.7-1.0 (domain-dependent)
  score                   ~ 0.56-0.65 → REFUSE at step 11

For pure structural failure (0% failures, entropy rising):
  rolling_failure_rate    = 0.00
  time_above_entropy_floor ~ 1.0 (after a few steps)
  score                   = 0.30  →  DEGRADE, never REFUSE

For 67% burst failure (F,F,S pattern):
  rolling_failure_rate    = 0.70
  time_above_entropy_floor ~ 0.8-1.0
  score                   ~ 0.73-0.79 → REFUSE at step 11, but
  CB (window=5) already trips at step 6 in the hybrid — CB wins.

This ensures that persistence triggers on slow-burn instability that both
RNOS and CB miss, without firing before the faster controller on scenarios
where RNOS or CB are the correct primary signal.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class PersistenceAssessment:
    decision: str           # "allow" | "degrade" | "refuse"
    score: float
    rolling_failure_rate: float
    time_above_entropy_floor: float
    window_fill: int        # how many steps are currently in the window


class PersistenceController:
    """Domain-agnostic persistence / drift detector.

    Parameters
    ----------
    window_size : int
        Number of steps in the long observation window. Default 10
        (2× the default CB short window of 5).
    entropy_floor : float
        Entropy value above which a step is counted as "above floor".
        Default 3.0 (calibrated for both db_control and ci_control domains).
    degrade_threshold : float
        Minimum persistence score to return DEGRADE. Default 0.30.
    refuse_threshold : float
        Minimum persistence score to return REFUSE. Default 0.50.
    """

    def __init__(
        self,
        window_size: int = 10,
        entropy_floor: float = 3.0,
        degrade_threshold: float = 0.30,
        refuse_threshold: float = 0.50,
    ) -> None:
        self.window_size = window_size
        self.entropy_floor = entropy_floor
        self.degrade_threshold = degrade_threshold
        self.refuse_threshold = refuse_threshold
        self._failure_window: deque[bool] = deque(maxlen=window_size)
        self._above_floor_window: deque[bool] = deque(maxlen=window_size)

    @property
    def rolling_failure_rate(self) -> float:
        if not self._failure_window:
            return 0.0
        return sum(self._failure_window) / len(self._failure_window)

    @property
    def time_above_entropy_floor(self) -> float:
        if not self._above_floor_window:
            return 0.0
        return sum(self._above_floor_window) / len(self._above_floor_window)

    def update(self, success: bool, rnos_entropy: float) -> None:
        """Record the outcome of the just-executed step.

        Call this AFTER the step executes (and after evaluate() for that step).
        """
        self._failure_window.append(not success)
        self._above_floor_window.append(rnos_entropy > self.entropy_floor)

    def evaluate(self) -> PersistenceAssessment:
        """Return a persistence assessment based on the current window.

        Returns ALLOW if the window is not yet full — prevents early false
        positives from partial window statistics.
        """
        fill = len(self._failure_window)
        if fill < self.window_size:
            return PersistenceAssessment(
                decision="allow",
                score=0.0,
                rolling_failure_rate=self.rolling_failure_rate,
                time_above_entropy_floor=self.time_above_entropy_floor,
                window_fill=fill,
            )

        failure_rate = self.rolling_failure_rate
        above_floor = self.time_above_entropy_floor
        score = 0.7 * failure_rate + 0.3 * above_floor

        if score >= self.refuse_threshold:
            decision = "refuse"
        elif score >= self.degrade_threshold:
            decision = "degrade"
        else:
            decision = "allow"

        return PersistenceAssessment(
            decision=decision,
            score=score,
            rolling_failure_rate=failure_rate,
            time_above_entropy_floor=above_floor,
            window_fill=fill,
        )

    def reset(self) -> None:
        self._failure_window.clear()
        self._above_floor_window.clear()
