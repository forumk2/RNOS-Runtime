"""
RNOS Slow Drift Showcase
========================
Demonstrates a known structural boundary of RNOS:
distributed instability (no retry branching, no cascade) is
invisible to entropy-based structural gating.

A sliding-window circuit breaker detects it cleanly.

This is a credibility demonstration — honest about what RNOS does and does not do.

Failure model
-------------
  - Base failure rate: ~40%
  - Max 2 consecutive failures (streak is capped to prevent cascade-like structure
    that would give RNOS a signal it was designed for)
  - Failures are distributed across the timeline: "smoldering instability"

Why RNOS misses it
------------------
  RNOS entropy for sequential, non-branching calls (fanout=1, depth=1):

      fanout_score   = log2(1) * 1.2          = 0.0   (no branching)
      depth_score    = 1 * 0.6                = 0.6   (flat call chain)
      requests_score = log2(step) * 0.5       <= 2.0  (caps quickly)
      ------------------------------------------------
      max entropy                             ~= 2.6

  DEGRADE threshold: 8.0 — never reached.
  REFUSE  threshold: 10.0 — never reached.

  Formula mirrors experiments/microservice_control/controllers.py:RNOSMSController.

Why CB catches it
-----------------
  A 5-step window filling with 4 failures (rate=0.80 > 0.60 threshold) trips the
  breaker.  CB is blind to structure; it only counts failure density.  That is
  exactly what this scenario produces.
"""

from __future__ import annotations

import math
import random
from collections import deque

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FAILURE_RATE = 0.40     # base probability of failure per step
MAX_STREAK   = 2        # max consecutive failures (caps at 2 to suppress cascade signal)
MAX_STEPS    = 150      # total steps to simulate

CB_WINDOW    = 5        # circuit breaker sliding window length
CB_THRESHOLD = 0.60     # failure rate above this trips the breaker (exclusive: > not >=)

# RNOS thresholds (from RNOSMSController)
RNOS_DEGRADE = 8.0
RNOS_REFUSE  = 10.0


# ---------------------------------------------------------------------------
# Failure sequence generation
# ---------------------------------------------------------------------------

def generate_failures(n: int, seed: int = 42) -> list[bool]:
    """
    Generate a deterministic failure sequence with controlled distribution.

    - Uses FAILURE_RATE as base probability.
    - After MAX_STREAK consecutive failures, the next step is forced to succeed.
      This prevents long failure streaks, which would give RNOS an unfair signal
      (cascade-like structure).  The overall failure rate stays near FAILURE_RATE.
    """
    rng = random.Random(seed)
    sequence: list[bool] = []
    consecutive = 0

    for _ in range(n):
        if consecutive >= MAX_STREAK:
            failed = False                        # force success — no long streaks
        else:
            failed = rng.random() < FAILURE_RATE

        sequence.append(failed)
        consecutive = (consecutive + 1) if failed else 0

    return sequence


# ---------------------------------------------------------------------------
# RNOS entropy (mirrors RNOSMSController, fanout=1 depth=1)
# ---------------------------------------------------------------------------

def rnos_entropy(step: int) -> float:
    """
    RNOS structural entropy for a flat sequential call at `step`.

    fanout=1, depth=1: this is a single-level API call with no retry branching.
    Mirrors the formula in experiments/microservice_control/controllers.py.

    Ceiling: 0.0 + 0.6 + 2.0 = 2.6 — far below DEGRADE threshold of 8.0.
    """
    fanout_score   = min(math.log2(max(1, 1)) * 1.2, 5.0)    # 0.0  — fanout=1, no branching
    depth_score    = min(1 * 0.6, 4.0)                         # 0.6  — depth=1, flat chain
    requests_score = min(math.log2(max(step, 1)) * 0.5, 2.0)  # grows slowly, hard cap at 2.0
    return fanout_score + depth_score + requests_score


# ---------------------------------------------------------------------------
# Baseline: no control
# ---------------------------------------------------------------------------

def run_baseline(failures: list[bool]) -> tuple[int, str]:
    return len(failures), f"COMPLETED (unstable, {sum(failures)}/{len(failures)} failures)"


# ---------------------------------------------------------------------------
# RNOS simulation
# ---------------------------------------------------------------------------

def run_rnos(failures: list[bool]) -> tuple[int, str, float]:
    """
    Steps through the failure sequence, checking RNOS entropy before each call.
    Returns (steps, status, peak_entropy).

    Expected result: runs all MAX_STEPS because entropy never approaches threshold.
    """
    peak_entropy = 0.0

    for step, _failed in enumerate(failures, 1):
        e = rnos_entropy(step)
        peak_entropy = max(peak_entropy, e)

        if e >= RNOS_REFUSE:
            return step, "REFUSED", peak_entropy
        # ALLOW — proceed (failure or success does not change RNOS signal here)

    return len(failures), "COMPLETED (missed)", peak_entropy


# ---------------------------------------------------------------------------
# Circuit Breaker simulation
# ---------------------------------------------------------------------------

def run_cb(failures: list[bool]) -> tuple[int, str, float]:
    """
    Sliding-window circuit breaker.

    Maintains a rolling window of the last CB_WINDOW outcomes.
    Trips (OPEN) when failure_rate > CB_THRESHOLD and window is full.
    Returns (steps_run, status, trigger_failure_rate).
    """
    window: deque[int] = deque(maxlen=CB_WINDOW)

    for step, failed in enumerate(failures, 1):
        window.append(1 if failed else 0)

        if len(window) == CB_WINDOW:
            rate = sum(window) / CB_WINDOW
            if rate > CB_THRESHOLD:
                return step, "STOPPED (detected)", rate

    final_rate = sum(window) / len(window) if window else 0.0
    return len(failures), "COMPLETED (missed)", final_rate


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _failure_map(failures: list[bool], width: int = 50) -> str:
    """Compact visual of the failure timeline. F=failure, .=success."""
    chunk = max(1, len(failures) // width)
    chars = []
    for i in range(0, len(failures), chunk):
        chars.append("F" if any(failures[i : i + chunk]) else ".")
    return "".join(chars[:width])


def _steps_bar(steps: int, total: int, width: int = 30) -> str:
    filled = int(width * steps / max(total, 1))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    failures = generate_failures(MAX_STEPS)
    actual_rate = sum(failures) / MAX_STEPS

    print()
    print("=== RNOS SLOW DRIFT SHOWCASE ===")
    print()
    print("Scenario:")
    print(f"  * Distributed failures (~{int(FAILURE_RATE * 100)}%)")
    print(f"  * No long streaks (max {MAX_STREAK} consecutive)")
    print(f"  * Slow instability buildup over {MAX_STEPS} steps")
    print()
    print(f"  Failure map (F=fail, .=ok):")
    print(f"    {_failure_map(failures)}")
    print(f"  Actual failure rate: {sum(failures)}/{MAX_STEPS} = {actual_rate:.0%}")
    print()

    baseline_steps, baseline_status               = run_baseline(failures)
    rnos_steps,     rnos_status,     peak_entropy  = run_rnos(failures)
    cb_steps,       cb_status,       cb_rate        = run_cb(failures)

    print("Results:")
    print()
    print("  Baseline (no control):")
    print(f"    Steps:    {baseline_steps:>5}")
    print(f"    Status:   {baseline_status}")
    print(f"    Progress: {_steps_bar(baseline_steps, MAX_STEPS)}")
    print()
    print("  RNOS:")
    print(f"    Steps:    {rnos_steps:>5}")
    print(f"    Status:   {rnos_status}")
    print(f"    Peak entropy:  {peak_entropy:.2f}  (DEGRADE @ {RNOS_DEGRADE}, REFUSE @ {RNOS_REFUSE})")
    print(f"    Progress: {_steps_bar(rnos_steps, MAX_STEPS)}")
    print()
    print("  Circuit Breaker:")
    print(f"    Steps:    {cb_steps:>5}")
    print(f"    Status:   {cb_status}")
    if "STOPPED" in cb_status:
        print(f"    Window failure rate at trigger: {cb_rate:.0%} (threshold: >{CB_THRESHOLD:.0%})")
    print(f"    Progress: {_steps_bar(cb_steps, MAX_STEPS)}")
    print()
    print("---")
    print()
    print("Conclusion:")
    print("  RNOS does not detect distributed instability in this regime.")
    print(f"  Peak entropy reached: {peak_entropy:.2f} -- nowhere near the {RNOS_REFUSE} refuse threshold.")
    print("  No structural explosion = no expanding call graph = no entropy signal.")
    print()
    print("  Circuit breaker succeeds due to failure density tracking.")
    if "STOPPED" in cb_status:
        print(f"  Failure density crossed {CB_THRESHOLD:.0%} within a {CB_WINDOW}-step window at step {cb_steps}.")
    print()
    print("  This is the intended boundary. RNOS and CB are complementary.")
    print()


if __name__ == "__main__":
    main()
