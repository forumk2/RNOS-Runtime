"""Scenario definitions for microservice control experiment.

Three failure geometries designed to stress-test different control primitives:

Scenario A — fanout_cascade (RNOS strength)
--------------------------------------------
Fanout doubles each step, depth increments by 1, all requests succeed.
Structural entropy grows rapidly via the log-scale fanout term:

    fanout_score = min(log2(fanout) * 1.2, 5.0)
    depth_score  = min(depth * 0.6, 4.0)
    requests_score = min(log2(total_requests) * 0.5, 2.0)

step 4: entropy ≈ 7.95  → ALLOW  (just below DEGRADE threshold 8.0)
step 5: entropy ≈ 9.80  → DEGRADE
step 6: entropy ≈ 10.60 → REFUSE

Latency is constant at 50 ms (latency_trend = 0). No failures.
CB never trips (0 failures in window).
Persistence never fires (0 failures, latency_trend = 0 < floor 10).
Hybrid = RNOS.

Scenario B — retry_storm (CB strength)
----------------------------------------
Fanout and depth are stable (fanout=3, depth=2). Failures follow the
pattern F, F, F, S repeating (75% failure rate). On failure steps,
total_requests grows by fanout * 2 (original request + one retry).

RNOS entropy stays ≤ 5.1 throughout (below DEGRADE 8.0) because
fanout and depth are constant; only requests_score grows slowly.

CB sliding window (size=5, threshold=0.6, strict >):
    After step 5: window = [F,F,F,S,F] → rate = 4/5 = 0.80 > 0.60
    CB REFUSE fires at step 6 evaluation.

Persistence window (size=10) cannot fill before CB fires at step 6.
Hybrid = CB.

Scenario C — latency_drift (Persistence strength)
---------------------------------------------------
Fanout and depth are stable (fanout=3, depth=2). Failure pattern is
alternating F, S (50% failure rate). Total_requests grows by fanout each
step (no retry amplification). Latency increases by 20 ms per step:

    latency_ms = 100 + 20 * (step - 1)
    latency_trend = +20 ms/step for step > 1, 0 for step 1

RNOS entropy stays ≤ 5.1 throughout (fanout, depth stable).

CB sliding window (size=5, threshold=0.6, strict >):
    Alternating F,S fills window as [F,S,F,S,F] → rate = 3/5 = 0.60
    NOT strictly > 0.60 → CB stays closed throughout.

Persistence (window=10, entropy_floor=10.0, latency_trend as signal):
    After 10 steps (evaluated at step 11):
        failure_rate        = 5/10 = 0.50
        time_above_floor    = 9/10 = 0.90  (steps 2–10 have trend=20 > 10)
        score = 0.7*0.50 + 0.3*0.90 = 0.62 → REFUSE

Hybrid = Persistence.
"""

from __future__ import annotations

from experiments.microservice_control.service_model import RequestState


# ---------------------------------------------------------------------------
# Scenario A: fanout_cascade
# ---------------------------------------------------------------------------

def make_fanout_cascade(max_steps: int = 20) -> list[RequestState]:
    """Generate RequestState sequence for fanout_cascade.

    Step N (1-indexed):
        fanout          = 2 ** (N - 1)   (1, 2, 4, 8, 16, ...)
        depth           = N
        total_requests  = cumulative sum of fanout (non-resetting)
        failures_last_n = 0
        latency_ms      = 50.0  (stable, modest)
        latency_trend   = 0.0
        success         = True  (structural failure only, no runtime failures)
    """
    states: list[RequestState] = []
    total_requests = 0
    for n in range(1, max_steps + 1):
        fanout = 2 ** (n - 1)
        total_requests += fanout
        states.append(RequestState(
            step=n,
            fanout=fanout,
            depth=n,
            total_requests=total_requests,
            failures_last_n=0,
            latency_ms=50.0,
            latency_trend=0.0,
            success=True,
        ))
    return states


# ---------------------------------------------------------------------------
# Scenario B: retry_storm
# ---------------------------------------------------------------------------

def make_retry_storm(max_steps: int = 20) -> list[RequestState]:
    """Generate RequestState sequence for retry_storm.

    Repeating pattern: F, F, F, S (failure, failure, failure, success).
        fanout          = 3   (stable)
        depth           = 2   (stable)
        total_requests  += fanout * 2 on failure (request + 1 retry)
                         += fanout     on success
        latency_ms      = 200.0 on failure, 80.0 on success
        latency_trend   = latency_ms[N] - latency_ms[N-1]  (0 for step 1)
        failures_last_n = rolling count of failures in previous 5 steps
    """
    pattern = [False, False, False, True]  # F, F, F, S
    fanout = 3
    depth = 2
    total_requests = 0
    states: list[RequestState] = []
    prev_latency: float = 80.0  # baseline for trend computation

    # Rolling failure window (last 5 outcomes) for failures_last_n display
    recent: list[bool] = []

    for n in range(1, max_steps + 1):
        success = pattern[(n - 1) % 4]
        total_requests += fanout * (1 if success else 2)
        latency = 80.0 if success else 200.0
        trend = latency - prev_latency if n > 1 else 0.0
        failures_last_n = recent[-5:].count(False) if recent else 0

        states.append(RequestState(
            step=n,
            fanout=fanout,
            depth=depth,
            total_requests=total_requests,
            failures_last_n=failures_last_n,
            latency_ms=latency,
            latency_trend=trend,
            success=success,
        ))
        recent.append(success)
        prev_latency = latency

    return states


# ---------------------------------------------------------------------------
# Scenario C: latency_drift
# ---------------------------------------------------------------------------

def make_latency_drift(max_steps: int = 20) -> list[RequestState]:
    """Generate RequestState sequence for latency_drift.

    Failure pattern: F, S alternating (50% failure rate).
        fanout          = 3   (stable)
        depth           = 2   (stable)
        total_requests  += fanout each step (no retry on failure)
        latency_ms      = 100 + 20 * (N - 1)  (increases 20 ms/step)
        latency_trend   = +20.0 for N > 1, 0.0 for N == 1
        failures_last_n = rolling count of failures in previous 5 steps

    Why CB misses this (window=5, threshold=0.6, strict >):
        Alternating F,S fills 5-window as [F,S,F,S,F] → rate = 3/5 = 0.60
        Not strictly > 0.60 → CB stays ALLOW throughout.

    Why Persistence catches this (window=10, entropy_floor=10.0):
        At step 11 (first full window, steps 1–10):
            failure_rate     = 5/10 = 0.50
            time_above_floor = 9/10 = 0.90  (steps 2–10: trend=20 > 10)
            score = 0.7*0.50 + 0.3*0.90 = 0.62 → REFUSE
    """
    fanout = 3
    depth = 2
    total_requests = 0
    states: list[RequestState] = []
    recent: list[bool] = []

    for n in range(1, max_steps + 1):
        success = (n % 2 == 0)           # False on odd (F), True on even (S)
        total_requests += fanout
        latency = 100.0 + 20.0 * (n - 1)
        trend = 20.0 if n > 1 else 0.0
        failures_last_n = recent[-5:].count(False) if recent else 0

        states.append(RequestState(
            step=n,
            fanout=fanout,
            depth=depth,
            total_requests=total_requests,
            failures_last_n=failures_last_n,
            latency_ms=latency,
            latency_trend=trend,
            success=success,
        ))
        recent.append(success)

    return states
