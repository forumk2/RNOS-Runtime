"""Scenario definitions for Experiment 5: Hybrid RNOS + Circuit Breaker.

Two failure geometries are designed to stress-test different control primitives:

Scenario A — cascading_burst (RNOS strength)
---------------------------------------------
Failures arrive in rapid consecutive bursts starting at step 3.
Entropy accumulates quickly via retry_score (consecutive failures), causing
RNOS to intervene before the CB's sliding window fills.

Failure pattern (30 steps):
  steps  1-2:  success (80 ms)         — stable baseline
  steps  3-9:  consecutive failures    — burst (400-460 ms escalating)
  steps 10+:   certain failure (480ms) — absorbing regime

Ground-truth action: intervene (structural failure).
Expected: RNOS catches early (step ~8); CB catches later (window fills at
step 11 after 10 executions); Hybrid = RNOS.

Scenario B — distributed_low_rate (CB strength)
------------------------------------------------
Failures are distributed evenly in a repeating F-F-S pattern (67% failure
rate, ≤2 consecutive failures). This pattern keeps RNOS's retry_score low
(max 2.0) and failure_score below the DEGRADE threshold under EXP2_POLICY
(degrade_entropy=9.0). The AdaptiveCircuitBreaker trips when its sliding
window fills with ≥7/10 failures.

Failure pattern (30 steps):
  F, F, S, F, F, S, F, F, S, ...  repeating (window_size=10 fills after 10 execs)

Latency: 200 ms on failure, 80 ms on success — no runaway latency spike,
so RNOS's latency_score stays low (~0.04-0.10).

Ground-truth action: intervene (chronic structural failure).
Expected: RNOS stays ALLOW throughout; CB trips at step 11; Hybrid = CB.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.configurable_api import ConfigurableAPI


# ---------------------------------------------------------------------------
# Scenario A: cascading_burst
# ---------------------------------------------------------------------------

def make_cascading_burst(seed: int = 42) -> ConfigurableAPI:
    """Rapid consecutive-failure burst followed by absorbing failure regime.

    Steps 1-2:  success (80 ms)
    Steps 3-9:  consecutive failures, escalating latency (400-460 ms)
    Steps 10+:  certain failure (fail_probs=[1.0], 480 ms)

    Design rationale
    ----------------
    * RNOS's retry_score accumulates linearly with consecutive failures
      (1.0/failure, capped at 4.0). By step 9 (7 consecutive), retry_score=4.0
      and failure_score=3.0 (capped). With EXP2_POLICY (degrade=9.0),
      RNOS degrades at step ~8 (entropy ~10.8) and refuses at step ~9.
    * AdaptiveCircuitBreaker (window_size=10, threshold=0.6) requires 10
      executions before evaluating the window. After 10 execs the window
      holds [S,S,F,F,F,F,F,F,F,F] = 8/10=0.80 > 0.60 → trips at step 11.
    * Hybrid catches at the same step as RNOS (~step 8) because RNOS raises
      severity before the CB window fills.
    """
    schedule = [
        {"success": True},   # 1
        {"success": True},   # 2
        {"success": False},  # 3 — burst begins
        {"success": False},  # 4
        {"success": False},  # 5
        {"success": False},  # 6
        {"success": False},  # 7
        {"success": False},  # 8
        {"success": False},  # 9
    ]
    latency = [
        80.0,  80.0,          # 1-2 stable
        400.0, 415.0, 430.0,  # 3-5 burst onset
        440.0, 450.0, 455.0,  # 6-8 escalating
        460.0,                # 9
    ] + [480.0] * 21         # 10-30 absorbing regime
    cost = (
        [0.010, 0.010]       # 1-2
        + [0.020 + i * 0.005 for i in range(7)]  # 3-9
        + [0.040] * 21       # 10-30
    )
    return ConfigurableAPI(
        name="cascading_burst",
        step_schedule=schedule,
        fail_probs=[1.0],     # steps 10+: certain failure
        latency_profile=latency,
        cost_profile=cost,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Scenario B: distributed_low_rate
# ---------------------------------------------------------------------------

def make_distributed_low_rate(seed: int = 42) -> ConfigurableAPI:
    """Persistent 67% failure rate with no consecutive runs exceeding 2.

    The failure pattern repeats as F-F-S (fail, fail, success):
        steps 1,2,3 = F,F,S  →  4,5,6 = F,F,S  → … (30 steps total)

    This gives:
    * Failure rate in any 10-step window: ~7/10 = 0.70 > 0.60 threshold
      → AdaptiveCircuitBreaker trips after 10 executions (at step 11).
    * Consecutive failure runs: max 2 → RNOS retry_score ≤ 2.0.
    * failure_score (last 5): ~3 failures → 3*0.65 = 1.95, never 3.0.
    * entropy peak (with EXP2_POLICY, cost/repeated_tool floor of ~4.0):
        ~0 + 2.0 + 1.95 + 2.0 + 0.10 + 2.0 = 8.05 → below DEGRADE (9.0).
    * RNOS stays ALLOW throughout; hybrid catches via CB at step 11.

    Latency: 200 ms on failure, 80 ms on success — no runaway spike that
    would elevate RNOS's latency_score above 0.10 per step.
    """
    # Build a 30-step explicit schedule repeating F,F,S
    schedule = []
    for _ in range(10):        # 30 steps = 10 repetitions of F,F,S
        schedule.extend([
            {"success": False},
            {"success": False},
            {"success": True},
        ])
    latency = []
    for _ in range(10):
        latency.extend([200.0, 200.0, 80.0])
    cost = []
    for _ in range(10):
        cost.extend([0.020, 0.020, 0.010])

    return ConfigurableAPI(
        name="distributed_low_rate",
        step_schedule=schedule,
        fail_probs=[0.667],   # if run exceeds 30 steps, continue ~67% failure
        latency_profile=latency,
        cost_profile=cost,
        seed=seed,
    )
