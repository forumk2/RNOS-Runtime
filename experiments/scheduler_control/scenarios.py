"""Scenario definitions for the job scheduler control experiment.

Three failure geometries designed to stress-test different control primitives:

Scenario A — dependency_explosion (RNOS strength)
--------------------------------------------------
Active jobs double each cycle, dependency depth increments by 1, all jobs
succeed. Structural entropy grows rapidly:

    active_score  = min(log(active_jobs + 1) * 1.2, 5.0)
    spawned_score = min(log(total_jobs_spawned + 1) * 0.5, 2.0)
    depth_score   = min(dependency_depth * 0.6, 4.0)

step 4: entropy ≈ 7.67  → ALLOW  (below DEGRADE threshold 8.0)
step 5: entropy ≈ 8.13  → DEGRADE
step 6: entropy ≈ 9.80  → DEGRADE
step 7: entropy ≈ 11.00 → REFUSE

Queue wait time is stable at 4.0 (wait_time_trend = 0). No failures.
CB never trips (0 failures in window).
Persistence never fires (0 failures, wait_time_trend = 0 < floor 2.0).
Hybrid = RNOS.

Scenario B — failing_jobs_storm (CB strength)
----------------------------------------------
Job structure is stable (active_jobs=5, dependency_depth=2). Failures
follow the pattern F, F, F, S repeating (75% failure rate). On failure
cycles, total_jobs_spawned grows by active_jobs * 2 (original + one retry).

RNOS entropy stays ≤ 5.0 throughout (active_jobs and depth are constant;
only spawned_score grows slowly via log).

CB sliding window (size=5, threshold=0.6, strict >):
    After step 5: window = [F,F,F,S,F] → rate = 4/5 = 0.80 > 0.60
    CB REFUSE fires at step 6 evaluation.

Persistence window (size=10) cannot fill before CB fires at step 6.
Hybrid = CB.

Scenario C — queue_backlog_drift (Persistence strength)
--------------------------------------------------------
Job structure is stable (active_jobs=3, dependency_depth=2). Failure
pattern is alternating F, S (50% failure rate). Queue wait time increases
by 3 units per cycle:

    queue_wait_time = 5.0 + 3.0 * (step - 1)   → 5, 8, 11, 14, ...
    wait_time_trend = +3.0 for step > 1, 0.0 for step 1

RNOS entropy stays ≤ 4.5 throughout (active_jobs and depth are stable).

CB sliding window (size=5, threshold=0.6, strict >):
    Alternating F,S fills 5-window as [F,S,F,S,F] → rate = 3/5 = 0.60
    NOT strictly > 0.60 → CB stays closed throughout.

Persistence (window=10, entropy_floor=2.0, wait_time_trend as signal):
    After 10 steps (evaluated at step 11):
        failure_rate     = 5/10 = 0.50  (steps 1,3,5,7,9 are failures)
        time_above_floor = 9/10 = 0.90  (steps 2–10: trend=3.0 > 2.0)
        score = 0.7*0.50 + 0.3*0.90 = 0.35 + 0.27 = 0.62 → REFUSE

Hybrid = Persistence.
"""

from __future__ import annotations

from experiments.scheduler_control.scheduler_model import SchedulerState


# ---------------------------------------------------------------------------
# Scenario A: dependency_explosion
# ---------------------------------------------------------------------------

def make_dependency_explosion(max_steps: int = 20) -> list[SchedulerState]:
    """Generate SchedulerState sequence for dependency_explosion.

    Step N (1-indexed):
        active_jobs         = 2 ** (N - 1)   (1, 2, 4, 8, 16, ...)
        dependency_depth    = N
        total_jobs_spawned  = cumulative sum of active_jobs
        queued_jobs         = 5   (stable, moderate backlog)
        failures_last_n     = 0
        queue_wait_time     = 4.0  (stable, no drift)
        wait_time_trend     = 0.0
        success             = True (structural failure only, no runtime failures)
    """
    states: list[SchedulerState] = []
    total_jobs_spawned = 0

    for n in range(1, max_steps + 1):
        active_jobs = 2 ** (n - 1)
        total_jobs_spawned += active_jobs
        states.append(SchedulerState(
            step=n,
            active_jobs=active_jobs,
            queued_jobs=5,
            total_jobs_spawned=total_jobs_spawned,
            dependency_depth=n,
            failures_last_n=0,
            queue_wait_time=4.0,
            wait_time_trend=0.0,
            success=True,
        ))

    return states


# ---------------------------------------------------------------------------
# Scenario B: failing_jobs_storm
# ---------------------------------------------------------------------------

def make_failing_jobs_storm(max_steps: int = 20) -> list[SchedulerState]:
    """Generate SchedulerState sequence for failing_jobs_storm.

    Repeating pattern: F, F, F, S (failure, failure, failure, success).
        active_jobs         = 5   (stable)
        dependency_depth    = 2   (stable)
        total_jobs_spawned  += active_jobs * 2 on failure (job + 1 retry)
                             += active_jobs     on success
        queued_jobs         = 8 on failure (backpressure), 4 on success
        queue_wait_time     = 20.0 on failure, 8.0 on success (fluctuating)
        wait_time_trend     = queue_wait_time[N] - queue_wait_time[N-1]
        failures_last_n     = rolling count of failures in previous 5 cycles
        success             = per pattern
    """
    pattern = [False, False, False, True]  # F, F, F, S
    active_jobs = 5
    dependency_depth = 2
    total_jobs_spawned = 0
    states: list[SchedulerState] = []
    prev_wait: float = 8.0
    recent: list[bool] = []

    for n in range(1, max_steps + 1):
        success = pattern[(n - 1) % 4]
        total_jobs_spawned += active_jobs * (1 if success else 2)
        wait_time = 8.0 if success else 20.0
        trend = wait_time - prev_wait if n > 1 else 0.0
        queued_jobs = 4 if success else 8
        failures_last_n = recent[-5:].count(False) if recent else 0

        states.append(SchedulerState(
            step=n,
            active_jobs=active_jobs,
            queued_jobs=queued_jobs,
            total_jobs_spawned=total_jobs_spawned,
            dependency_depth=dependency_depth,
            failures_last_n=failures_last_n,
            queue_wait_time=wait_time,
            wait_time_trend=trend,
            success=success,
        ))
        recent.append(success)
        prev_wait = wait_time

    return states


# ---------------------------------------------------------------------------
# Scenario C: queue_backlog_drift
# ---------------------------------------------------------------------------

def make_queue_backlog_drift(max_steps: int = 20) -> list[SchedulerState]:
    """Generate SchedulerState sequence for queue_backlog_drift.

    Failure pattern: F, S alternating (50% failure rate).
        active_jobs         = 3   (stable)
        dependency_depth    = 2   (stable)
        total_jobs_spawned  += 3 each step (no retry amplification)
        queued_jobs         = 5 + 3*(N-1)  (slowly accumulating backlog)
        queue_wait_time     = 5.0 + 3.0*(N-1)  (5, 8, 11, 14, ...)
        wait_time_trend     = +3.0 for N > 1, 0.0 for N == 1
        failures_last_n     = rolling count of failures in previous 5 cycles
        success             = False on odd steps, True on even steps

    Why CB misses this (window=5, threshold=0.6, strict >):
        Alternating F,S fills 5-window as [F,S,F,S,F] → rate = 3/5 = 0.60
        Not strictly > 0.60 → CB stays ALLOW throughout.

    Why Persistence catches this (window=10, entropy_floor=2.0):
        At step 11 (first full window, steps 1–10):
            failure_rate     = 5/10 = 0.50  (steps 1,3,5,7,9 fail)
            time_above_floor = 9/10 = 0.90  (steps 2–10: trend=3.0 > 2.0)
            score = 0.7*0.50 + 0.3*0.90 = 0.62 → REFUSE
    """
    active_jobs = 3
    dependency_depth = 2
    total_jobs_spawned = 0
    states: list[SchedulerState] = []
    recent: list[bool] = []

    for n in range(1, max_steps + 1):
        success = (n % 2 == 0)           # False on odd (F), True on even (S)
        total_jobs_spawned += active_jobs
        queued_jobs = 5 + 3 * (n - 1)
        wait_time = 5.0 + 3.0 * (n - 1)
        trend = 3.0 if n > 1 else 0.0
        failures_last_n = recent[-5:].count(False) if recent else 0

        states.append(SchedulerState(
            step=n,
            active_jobs=active_jobs,
            queued_jobs=queued_jobs,
            total_jobs_spawned=total_jobs_spawned,
            dependency_depth=dependency_depth,
            failures_last_n=failures_last_n,
            queue_wait_time=wait_time,
            wait_time_trend=trend,
            success=success,
        ))
        recent.append(success)

    return states
