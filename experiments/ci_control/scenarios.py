"""Scenario definitions for CI control experiment.

Three failure geometries designed to stress-test different control primitives:

Scenario A — pipeline_fanout_explosion (RNOS strength)
-------------------------------------------------------
Each step the pipeline doubles its active jobs (fanout) and increments
retry_count as overwhelmed workers begin re-executing stalled jobs.
All jobs succeed — there is no failure signal for the CB.

Step N:
    active_jobs      = 2^(N-1)           (exponential fanout)
    total_jobs_spawned = 2^N - 1          (cumulative sum of all active)
    retry_count      = max(0, N - 2)      (retries start accumulating at step 3)
    success          = True

RNOS entropy rises with combined fanout + cumulative expansion + retry pressure:
    Step 5: entropy 9.87 → DEGRADE
    Step 6: entropy 10.98 → REFUSE  (6 executions)
CB never trips (zero failures in window, 20 executions).
Hybrid matches RNOS (6 executions, trigger="rnos").

Scenario B — flaky_test_storm (CB strength)
--------------------------------------------
Pipeline structure is stable (fixed active_jobs=3, spawning 3 jobs/step).
Jobs follow a repeating F-F-S failure pattern (67% failure rate).
retry_count accumulates slowly (incremented on each failure step).

Step N:
    active_jobs      = 3                  (stable)
    total_jobs_spawned = 3 * N            (linear growth)
    retry_count      = cumulative failures so far
    success          = F if (N-1)%3 in {0,1} else T

CB window=[F,F,S,F,F] at step 6 evaluation → failure_rate=0.8 > 0.6 → REFUSE (6 exec).
RNOS entropy stays below DEGRADE throughout (max ~7.2 at step 20).
Hybrid matches CB (6 executions, trigger="cb").
"""

from __future__ import annotations

from experiments.ci_control.pipeline_model import PipelineState


# ---------------------------------------------------------------------------
# Scenario A: pipeline_fanout_explosion
# ---------------------------------------------------------------------------

def make_pipeline_fanout_explosion(max_steps: int = 20) -> list[PipelineState]:
    """Exponential job fanout with accumulating retries, all successes.

    Step N:
        active_jobs       = 2^(N-1)
        total_jobs_spawned = 2^N - 1   (cumulative sum: 1+2+4+…+2^(N-1))
        retry_count       = max(0, N-2)
        success           = True
    """
    states: list[PipelineState] = []
    for n in range(1, max_steps + 1):
        active = 2 ** (n - 1)
        spawned = (2 ** n) - 1
        retries = max(0, n - 2)
        states.append(PipelineState(
            step=n,
            active_jobs=active,
            total_jobs_spawned=spawned,
            retry_count=retries,
            success=True,
        ))
    return states


# ---------------------------------------------------------------------------
# Scenario B: flaky_test_storm
# ---------------------------------------------------------------------------

def make_flaky_test_storm(max_steps: int = 20) -> list[PipelineState]:
    """Stable pipeline with repeating F-F-S failure pattern.

    Failure pattern (1-indexed): step%3 in {1,2} = fail; step%3 == 0 = success.
    retry_count is cumulative failures up to but not including this step
    (i.e. it's the state the controller sees before this execution).
    """
    states: list[PipelineState] = []
    cumulative_retries = 0
    for n in range(1, max_steps + 1):
        # F-F-S: positions 1,2 fail; position 3 (0-indexed 2) succeeds
        success = (n % 3 == 0)
        states.append(PipelineState(
            step=n,
            active_jobs=3,
            total_jobs_spawned=3 * n,
            retry_count=cumulative_retries,
            success=success,
        ))
        if not success:
            cumulative_retries += 1
    return states


# ---------------------------------------------------------------------------
# Scenario C: gradual_flakiness
# ---------------------------------------------------------------------------

def make_gradual_flakiness(max_steps: int = 20) -> list[PipelineState]:
    """Persistent 50% failure rate with stable structure — gradual flakiness.

    Failure pattern: F, S alternating.
        active_jobs       = 2  (stable, modest parallelism)
        total_jobs_spawned = 2 * step  (linear, no explosion)
        retry_count       = cumulative failures so far
        success           = False for odd steps, True for even steps

    Why RNOS misses this:
        active_jobs_score = 0.8 (constant)
        spawned_score grows slowly: log2(2*20)*0.5=2.66 (cap 3.0 not reached)
        retry_score accumulates: 10 retries by step 20 → capped at 3.0
        Max entropy at step 20: 0.8+2.66+3.0 = 6.46 — never reaches DEGRADE (8.0).

    Why CB misses this (window=5, threshold=0.6, strict >):
        Alternating F,S window oscillates at failure_rate=0.6 or 0.4 → never > 0.6.

    Why Persistence catches this (window=10, refuse_threshold=0.50):
        At step 11 (first full window, steps 1-10):
            failure_rate    = 5/10 = 0.50
            time_above_floor = 7/10 = 0.70  (entropy > 3.0 from step ~4)
            score = 0.7*0.50 + 0.3*0.70 = 0.56 >= 0.50 → REFUSE
    """
    states: list[PipelineState] = []
    cumulative_retries = 0
    for n in range(1, max_steps + 1):
        success = (n % 2 == 0)        # F on odd, S on even
        states.append(PipelineState(
            step=n,
            active_jobs=2,
            total_jobs_spawned=2 * n,
            retry_count=cumulative_retries,
            success=success,
        ))
        if not success:
            cumulative_retries += 1
    return states
