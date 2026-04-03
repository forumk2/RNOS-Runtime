"""Sliding-window adaptive circuit breaker for Experiment 2.

This breaker is intentionally more sophisticated than
``baselines/circuit_breaker.py`` and represents a serious algorithmic
baseline — what a competent engineer would actually deploy.

Key improvements over the basic consecutive-failure breaker:

* **Sliding failure window** — trips on failure *rate* over the last
  ``window_size`` executed steps, not raw consecutive count.  This is
  less sensitive to individual unlucky bursts.

* **Adaptive threshold** — tightens (becomes more sensitive) under
  sustained stress and loosens (becomes more tolerant) when a recovery
  probe succeeds.  Mirrors how a human operator would tune the knob.

* **Exponential backoff** — same cooldown-doubling schedule as the basic
  breaker so recovery probes back off gracefully under continued failure.

* **Half-open probe** — exactly one execution is allowed per cooldown
  cycle to test whether the downstream system has recovered.

The critical design choice for Experiment 2: the failure-rate check uses
strict ``>`` (not ``>=``).  This means a window of [T, T, F, F, F] gives
rate = 0.60 which is NOT > 0.60, so the breaker stays closed.  Scenarios
with exactly ``window_size - 2`` failures out of ``window_size`` steps
(like a 3-failure rough patch with a 5-step window) will therefore pass
through without tripping.
"""

from __future__ import annotations

from collections import deque


class AdaptiveCircuitBreaker:
    """Sliding-window adaptive circuit breaker with exponential backoff.

    States
    ------
    CLOSED:            Normal operation.  Window is monitored; trips when
                       failure rate exceeds the current threshold.
    OPEN:              Blocked.  Exponential cooldown counts down per tick.
    HALF_OPEN:         One probe execution allowed.  Success → CLOSED;
                       failure → OPEN with doubled cooldown.
    PERMANENTLY_OPEN:  Hard stop after ``max_total_blocked`` blocked steps.

    Adaptive threshold
    ------------------
    When the breaker trips: threshold decreases by ``adaptation_step``
    (clamped to ``min_failure_rate``), making it more sensitive.

    When a half-open probe succeeds: threshold increases by
    ``adaptation_step`` (clamped to ``initial_failure_rate``), making it
    more tolerant — trusting that the system has genuinely recovered.
    """

    def __init__(
        self,
        window_size: int = 5,
        initial_failure_rate: float = 0.60,
        min_failure_rate: float = 0.40,
        adaptation_step: float = 0.05,
        initial_cooldown_steps: int = 2,
        max_cooldown_steps: int = 10,
        max_total_blocked: int = 20,
    ) -> None:
        self._window_size = window_size
        self._initial_failure_rate = initial_failure_rate
        self._min_failure_rate = min_failure_rate
        self._adaptation_step = adaptation_step
        self._initial_cooldown = initial_cooldown_steps
        self._max_cooldown = max_cooldown_steps
        self._max_total_blocked = max_total_blocked

        self._state: str = "closed"
        self._window: deque[bool] = deque(maxlen=window_size)
        self._failure_rate_threshold: float = initial_failure_rate
        self._total_blocked: int = 0
        self._cooldown_remaining: int = 0
        self._current_cooldown: int = initial_cooldown_steps

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def should_execute(self) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` for the current step.

        Call this BEFORE executing.  Reason strings:
        - ``"closed"``            → normal, execute
        - ``"half_open_probe"``   → recovery probe, execute
        - ``"open_blocked"``      → backoff active, do NOT execute
        - ``"permanently_open"``  → hard stop, do NOT execute
        """
        if self._state == "permanently_open":
            return False, "permanently_open"

        if self._state == "open":
            self._total_blocked += 1
            if self._total_blocked >= self._max_total_blocked:
                self._state = "permanently_open"
                return False, "permanently_open"
            return False, "open_blocked"

        if self._state == "half_open":
            return True, "half_open_probe"

        # CLOSED: evaluate sliding window when it is full.
        if len(self._window) >= self._window_size:
            failures = sum(1 for r in self._window if not r)
            rate = failures / self._window_size
            if rate > self._failure_rate_threshold:
                # Trip — tighten threshold, enter OPEN.
                self._failure_rate_threshold = max(
                    self._min_failure_rate,
                    self._failure_rate_threshold - self._adaptation_step,
                )
                self._state = "open"
                self._current_cooldown = self._initial_cooldown
                self._cooldown_remaining = self._current_cooldown
                self._total_blocked += 1
                if self._total_blocked >= self._max_total_blocked:
                    self._state = "permanently_open"
                    return False, "permanently_open"
                return False, "open_blocked"

        return True, "closed"

    def record_result(self, *, success: bool) -> None:
        """Record outcome of an executed step.

        Call this AFTER execution.  Updates the sliding window and handles
        the HALF_OPEN → CLOSED / OPEN transition.
        """
        if self._state == "closed":
            self._window.append(success)

        elif self._state == "half_open":
            self._window.append(success)
            if success:
                # Probe succeeded: recover, loosen threshold.
                self._state = "closed"
                self._current_cooldown = self._initial_cooldown
                self._cooldown_remaining = 0
                self._failure_rate_threshold = min(
                    self._initial_failure_rate,
                    self._failure_rate_threshold + self._adaptation_step,
                )
            else:
                # Probe failed: re-open with doubled cooldown.
                self._current_cooldown = min(
                    self._current_cooldown * 2, self._max_cooldown
                )
                self._state = "open"
                self._cooldown_remaining = self._current_cooldown

    def tick(self) -> None:
        """Advance one step.  Call every loop iteration regardless of execution.

        Decrements the cooldown when OPEN; transitions to HALF_OPEN when
        the cooldown reaches zero.
        """
        if self._state == "open":
            self._cooldown_remaining -= 1
            if self._cooldown_remaining <= 0:
                self._cooldown_remaining = 0
                self._state = "half_open"

    # ------------------------------------------------------------------
    # Observable state
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        """Current state string."""
        return self._state

    @property
    def stats(self) -> dict:
        """Step-log metrics for the current state."""
        failures = sum(1 for r in self._window if not r)
        return {
            "failure_rate": round(failures / self._window_size, 3) if self._window else 0.0,
            "failures_in_window": failures,
            "window_fill": len(self._window),
            "current_threshold": round(self._failure_rate_threshold, 3),
            "total_blocked": self._total_blocked,
            "cooldown_remaining": self._cooldown_remaining,
            "current_cooldown_limit": self._current_cooldown,
        }
