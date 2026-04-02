"""Exponential backoff circuit breaker for tool execution control.

This module is intentionally independent of the RNOS runtime — it implements
the standard three-state circuit breaker pattern used in production systems
(AWS SDK, gRPC, Kubernetes readiness probes) as an algorithmic baseline for
comparison against RNOS entropy/trust gating.
"""

from __future__ import annotations


class CircuitBreaker:
    """Exponential backoff circuit breaker for tool execution control.

    States:
        CLOSED:           Normal operation. Requests pass through; consecutive
                          failures are tracked.
        OPEN:             Tripped. Requests are blocked; cooldown timer counts
                          down step by step.
        HALF_OPEN:        Probe state. Exactly one request is allowed through
                          to test whether the downstream system has recovered.
        PERMANENTLY_OPEN: Hard stop. ``max_total_blocked`` blocked requests
                          have accumulated; no further requests are ever allowed.

    Backoff schedule (each failed probe doubles the cooldown, capped at
    ``max_cooldown_steps``):
        initial → initial*2 → initial*4 → … → max_cooldown_steps
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        initial_cooldown_steps: int = 1,
        max_cooldown_steps: int = 8,
        max_total_blocked: int = 10,
    ) -> None:
        """Initialise the circuit breaker.

        Args:
            failure_threshold: Consecutive failures in CLOSED state before
                tripping to OPEN. Default: 3.
            initial_cooldown_steps: Steps to wait in OPEN state before allowing
                the first HALF_OPEN probe. Doubles on each failed probe.
                Default: 1.
            max_cooldown_steps: Upper bound on the exponential cooldown growth.
                Default: 8.
            max_total_blocked: Total blocked requests (across all OPEN windows)
                before the breaker enters PERMANENTLY_OPEN. Default: 10.
        """
        self._failure_threshold = failure_threshold
        self._initial_cooldown_steps = initial_cooldown_steps
        self._max_cooldown_steps = max_cooldown_steps
        self._max_total_blocked = max_total_blocked

        self._state: str = "closed"
        self._consecutive_failures: int = 0
        self._total_blocked: int = 0
        self._cooldown_remaining: int = 0
        self._current_cooldown_limit: int = initial_cooldown_steps

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def should_execute(self) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` for the current step.

        Call this BEFORE executing the tool. The ``reason`` strings:
        - ``"closed"``          → normal operation, execute
        - ``"half_open_probe"`` → probe attempt, execute
        - ``"open_blocked"``    → blocked by cooldown, do NOT execute
        - ``"permanently_open"``→ max blocked reached, do NOT execute
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

        # closed
        return True, "closed"

    def record_result(self, *, success: bool) -> None:
        """Record the outcome of an executed tool call.

        Call this AFTER executing the tool. State transitions:
        - CLOSED  + success → reset consecutive failure counter
        - CLOSED  + failure → increment counter; trip to OPEN when threshold hit
        - HALF_OPEN + success → transition to CLOSED, reset cooldown
        - HALF_OPEN + failure → transition to OPEN, double cooldown (capped)
        """
        if self._state == "closed":
            if success:
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._failure_threshold:
                    self._state = "open"
                    self._current_cooldown_limit = self._initial_cooldown_steps
                    self._cooldown_remaining = self._current_cooldown_limit

        elif self._state == "half_open":
            if success:
                self._state = "closed"
                self._consecutive_failures = 0
                self._current_cooldown_limit = self._initial_cooldown_steps
                self._cooldown_remaining = 0
            else:
                self._consecutive_failures += 1
                self._current_cooldown_limit = min(
                    self._current_cooldown_limit * 2, self._max_cooldown_steps
                )
                self._state = "open"
                self._cooldown_remaining = self._current_cooldown_limit

    def tick(self) -> None:
        """Advance one step. Call every loop iteration regardless of execution.

        Decrements the cooldown counter when OPEN. When the cooldown reaches 0
        the breaker transitions to HALF_OPEN so the next ``should_execute``
        call returns a probe slot.
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
        """Current state: ``'closed'``, ``'open'``, ``'half_open'``, or ``'permanently_open'``."""
        return self._state

    @property
    def stats(self) -> dict:
        """Return current breaker stats suitable for step-log lines.

        Keys:
            consecutive_failures: Failures in the current CLOSED window.
            total_blocked: Blocked requests accumulated across all OPEN windows.
            cooldown_remaining: Steps left before the next HALF_OPEN probe.
            current_cooldown_limit: Current cooldown ceiling (grows exponentially).
        """
        return {
            "consecutive_failures": self._consecutive_failures,
            "total_blocked": self._total_blocked,
            "cooldown_remaining": self._cooldown_remaining,
            "current_cooldown_limit": self._current_cooldown_limit,
        }
