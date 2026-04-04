"""Scenario definitions for DB control experiment.

Two failure geometries designed to stress-test different control primitives:

Scenario A — cascading_query_explosion (RNOS strength)
-------------------------------------------------------
Each step adds one JOIN level and doubles estimated cost:
    step N: join_depth=N, estimated_cost=2^N

All queries succeed — there is no failure signal for the CB to detect.
RNOS entropy grows with structural complexity:
    entropy(N) = min(N, 5) + min(log2(2^N)*0.5, 4) + cumulative_cost_score

RNOS trips at step ~7 (DEGRADE) and refuses by step ~8.
CB never trips (0 failures in window).
Hybrid matches RNOS.

Scenario B — lock_contention (CB strength)
-------------------------------------------
Queries have stable low complexity (join_depth=2, estimated_cost=10) but
high lock contention. Outcomes follow the pattern F-F-S repeating:
    steps 1,2 = fail; step 3 = success; steps 4,5 = fail; step 6 = success; ...

lock_wait_ms is elevated on failure steps (500 ms) vs. success steps (20 ms).

RNOS entropy stays low (no join growth, low cost, cumulative cost grows slowly):
    entropy_max = 2.0 + 1.66 + 2.0 = 5.66 → below DEGRADE threshold (8.0)
CB trips when window fills: first full window [F,F,S,F,F] has failure_rate=0.8>0.6.
Hybrid matches CB.
"""

from __future__ import annotations

from experiments.db_control.query_model import QueryState


# ---------------------------------------------------------------------------
# Scenario A: cascading_query_explosion
# ---------------------------------------------------------------------------

def make_cascading_query_explosion(max_steps: int = 20) -> list[QueryState]:
    """Generate QueryState sequence for cascading_query_explosion.

    Step N (1-indexed):
        join_depth     = N
        estimated_cost = 2 ** N
        lock_wait_ms   = 0.0   (no contention)
        success        = True  (structural failure only, not runtime failure)
    """
    states: list[QueryState] = []
    cumulative = 0.0
    for n in range(1, max_steps + 1):
        cost = float(2 ** n)
        states.append(QueryState(
            step=n,
            join_depth=n,
            estimated_cost=cost,
            lock_wait_ms=0.0,
            success=True,
            cumulative_cost=cumulative,
        ))
        cumulative += cost
    return states


# ---------------------------------------------------------------------------
# Scenario B: lock_contention
# ---------------------------------------------------------------------------

def make_lock_contention(max_steps: int = 20) -> list[QueryState]:
    """Generate QueryState sequence for lock_contention.

    Repeating pattern: F, F, S (failure, failure, success).
        join_depth     = 2        (stable, no growth)
        estimated_cost = 10.0     (stable, no growth)
        lock_wait_ms   = 500.0 on failure, 20.0 on success
        success        = per pattern above
    """
    # F-F-S pattern
    pattern = [False, False, True]
    states: list[QueryState] = []
    cumulative = 0.0
    for n in range(1, max_steps + 1):
        success = pattern[(n - 1) % 3]
        lock_wait = 20.0 if success else 500.0
        cost = 10.0
        states.append(QueryState(
            step=n,
            join_depth=2,
            estimated_cost=cost,
            lock_wait_ms=lock_wait,
            success=success,
            cumulative_cost=cumulative,
        ))
        cumulative += cost
    return states
