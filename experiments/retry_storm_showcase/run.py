"""
RNOS Retry Storm Showcase
=========================
Demonstrates pre-execution entropy gating (RNOS) vs uncontrolled retry explosion.

Retry storm model
-----------------
  - Unstable API: FAILURE_RATE=70%
  - Each failure spawns FANOUT=2 children (retries with branching)
  - Net amplification per step: 0.70 * 2 = 1.4x  →  supercritical, grows without bound

RNOS model
----------
  - Entropy budget: fixed pool of capacity units (ENTROPY_BUDGET)
  - Each prospective operation costs: BASE_COST + depth * DEPTH_COST
  - Gate is checked BEFORE execution; if cost > remaining budget, execution is refused
  - This is structural pre-execution admissibility — not a reactive circuit breaker
"""

from __future__ import annotations

import random
from collections import deque

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FAILURE_RATE        = 0.70    # probability any single API call fails
FANOUT              = 2       # retry branches spawned per failure
ENTROPY_BUDGET      = 50.0    # RNOS total entropy capacity
BASE_COST           = 1.0     # entropy cost: fixed component per operation
DEPTH_COST          = 0.5     # entropy cost: per depth level (penalises cascade depth)
MAX_BASELINE_CALLS  = 10_000  # safety cap for baseline (prevents infinite loop)


# ---------------------------------------------------------------------------
# Shared: simulated unstable API call
# ---------------------------------------------------------------------------

def call_api() -> bool:
    """Returns True on success (~30%), False on failure (~70%)."""
    return random.random() >= FAILURE_RATE


# ---------------------------------------------------------------------------
# Baseline: no control
# ---------------------------------------------------------------------------

def run_baseline() -> tuple[int, str]:
    """
    BFS retry expansion with no gating.

    Every failed call spawns FANOUT children.  With a net amplification of
    0.70 * 2 = 1.4, the queue grows unboundedly.  The only stop condition is
    natural completion (all calls succeed) or the safety cap.
    """
    queue: deque[int] = deque([0])  # each entry is a call depth
    total_calls = 0

    while queue:
        if total_calls >= MAX_BASELINE_CALLS:
            return total_calls, "COLLAPSED"

        depth = queue.popleft()
        total_calls += 1
        success = call_api()

        if not success:
            for _ in range(FANOUT):
                queue.append(depth + 1)

    return total_calls, "COMPLETED"


# ---------------------------------------------------------------------------
# RNOS-controlled
# ---------------------------------------------------------------------------

def operation_cost(depth: int) -> float:
    """Entropy cost for one operation at a given call depth."""
    return BASE_COST + depth * DEPTH_COST


def run_rnos() -> tuple[int, str, float]:
    """
    Same BFS retry expansion, but gated by an entropy budget.

    Before dequeuing each call, RNOS checks: can we afford this operation?
    If the remaining budget is insufficient, execution is refused immediately —
    before any work is done.  Shallow calls are cheap; deep retry chains are
    expensive.  The budget naturally drains fastest when the storm deepens.
    """
    queue: deque[int] = deque([0])
    total_calls = 0
    entropy_remaining = ENTROPY_BUDGET

    while queue:
        depth = queue[0]          # peek — do NOT dequeue yet
        cost  = operation_cost(depth)

        if cost > entropy_remaining:
            # Pre-execution gate: refuse before executing
            return total_calls, "REFUSED (early)", entropy_remaining

        # Admitted — deduct budget and execute
        queue.popleft()
        entropy_remaining -= cost
        total_calls += 1
        success = call_api()

        if not success:
            for _ in range(FANOUT):
                queue.append(depth + 1)

    return total_calls, "COMPLETED", entropy_remaining


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _bar(value: int, cap: int, width: int = 30) -> str:
    filled = int(width * min(value, cap) / cap)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("=== RNOS RETRY STORM SHOWCASE ===")
    print()
    print("Scenario:")
    print(f"  * Unstable API ({int(FAILURE_RATE * 100)}% failure rate)")
    print(f"  * Retries + branching (fanout = {FANOUT} per failure)")
    print(f"  * Net amplification per wave: {FAILURE_RATE * FANOUT:.1f}x  (supercritical)")
    print()

    # --- Run both simulations from the same seed ---
    random.seed(42)
    baseline_calls, baseline_status = run_baseline()

    random.seed(42)
    rnos_calls, rnos_status, entropy_left = run_rnos()

    # --- Results ---
    print("Results:")
    print()
    print("  Baseline (no control):")
    print(f"    Total Calls:  {baseline_calls:>7,}")
    print(f"    Status:       {baseline_status}")
    print(f"    Scale:        {_bar(baseline_calls, MAX_BASELINE_CALLS)}")
    print()
    print("  RNOS:")
    print(f"    Total Calls:  {rnos_calls:>7,}")
    print(f"    Status:       {rnos_status}")
    print(f"    Entropy left: {entropy_left:.1f} / {ENTROPY_BUDGET:.0f}")
    print(f"    Scale:        {_bar(rnos_calls, MAX_BASELINE_CALLS)}")
    print()
    print("---")
    print()

    # --- Reduction ---
    if baseline_calls > 0 and baseline_calls != rnos_calls:
        reduction = (1.0 - rnos_calls / baseline_calls) * 100.0
        print("Reduction:")
        print(f"  Calls Reduced:  {reduction:.1f}%")
        print(f"  ({baseline_calls:,} baseline  ->  {rnos_calls} RNOS)")
    print()

    # --- Conclusion ---
    print("Conclusion:")
    print("  RNOS halts execution before cascade amplification.")
    print("  Pre-execution admissibility gating stops the storm at the gate,")
    print("  not after damage is done.")
    print()


if __name__ == "__main__":
    main()
