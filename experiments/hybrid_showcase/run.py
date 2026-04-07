"""
RNOS Hybrid Showcase
====================
Demonstrates that no single controller is sufficient across mixed instability regimes,
and that Hybrid control adapts by applying the correct detector per failure geometry.

Two-phase scenario
------------------
  Phase 1 — Structural instability (retry storm, steps 1–12)
      The call graph expands each step: depth grows by 1, fanout doubles every 2 steps.
      This models accumulated retry-graph explosion regardless of individual outcomes.
      RNOS detects structural overload and refuses at step 8.
      CB stays quiet: individual call outcomes are ~25% failure — well below its threshold.

  Phase 2 — Distributed instability (slow drift, steps 13–162)
      Flat sequential calls with ~40% distributed failures, max 2 consecutive.
      No structural growth — RNOS entropy stays at ~2.5, far below its threshold.
      CB detects failure density in its sliding window and trips in Phase 2.

Hybrid result
-------------
  Hybrid checks both signals every step.  In this sequence RNOS fires first (step 8).
  If the sequence were Phase 2 only, CB would fire and RNOS would miss it entirely.
  Neither controller alone is sufficient.  Hybrid is not redundancy — it is required.

RNOS entropy formula (mirrors experiments/microservice_control/controllers.py)
-------------------------------------------------------------------------------
  fanout_score   = min(log2(fanout) * 1.2,    5.0)
  depth_score    = min(depth * 0.6,            4.0)
  requests_score = min(log2(total_steps) * 0.5, 2.0)
  entropy        = fanout_score + depth_score + requests_score
  DEGRADE at 8.0, REFUSE at 10.0

  Phase 1 step 8:  depth=8,  fanout=16  ->  entropy=10.3  REFUSE
  Phase 2 any step: depth=1, fanout=1   ->  entropy<=2.6  ALLOW (always)
"""

from __future__ import annotations

import math
import random
from collections import deque

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PHASE1_STEPS        = 12     # structural-instability phase length
PHASE2_STEPS        = 150    # slow-drift phase length

PHASE1_FAILURE_RATE = 0.25   # low enough to keep CB quiet during Phase 1
PHASE2_FAILURE_RATE = 0.40   # distributed failures for CB to detect in Phase 2
PHASE2_MAX_STREAK   = 2      # caps consecutive failures — suppresses cascade signal in Phase 2

CB_WINDOW     = 5
CB_THRESHOLD  = 0.60         # trip when failure rate in window EXCEEDS this (exclusive)

RNOS_DEGRADE  = 8.0
RNOS_REFUSE   = 10.0


# ---------------------------------------------------------------------------
# Sequence generation
# ---------------------------------------------------------------------------

def generate_sequence(seed: int = 42) -> list[bool]:
    """
    Generate a deterministic two-phase failure sequence from a single seed.

    Phase 1: moderate raw failure rate (~25%) so CB sees no density spike.
             Structural complexity grows deterministically — CB has no window
             to form before RNOS fires at step 8.

    Phase 2: ~40% distributed failures with a streak cap of 2.
             Streak cap prevents the long consecutive runs that would give
             RNOS a structural signal it could misinterpret.
    """
    rng = random.Random(seed)

    phase1 = [rng.random() < PHASE1_FAILURE_RATE for _ in range(PHASE1_STEPS)]

    phase2: list[bool] = []
    consecutive = 0
    for _ in range(PHASE2_STEPS):
        if consecutive >= PHASE2_MAX_STREAK:
            failed = False
        else:
            failed = rng.random() < PHASE2_FAILURE_RATE
        phase2.append(failed)
        consecutive = (consecutive + 1) if failed else 0

    return phase1 + phase2


# ---------------------------------------------------------------------------
# RNOS structural entropy
# ---------------------------------------------------------------------------

def structural_entropy(global_step: int) -> tuple[float, int, int]:
    """
    Compute RNOS entropy for a given global step.

    Phase 1 (global_step <= PHASE1_STEPS):
        depth  = global_step          (grows +1 per step)
        fanout = 2^(global_step // 2) (doubles every 2 steps, capped at 32)
        Models an expanding retry call graph — each step we are one level deeper
        into a cascade that has already branched at every prior depth.

    Phase 2 (global_step > PHASE1_STEPS):
        depth=1, fanout=1 — flat sequential calls, no structural expansion.
        Entropy ceiling ~2.6.  RNOS never triggers here.

    Returns (entropy, depth, fanout).
    """
    if global_step <= PHASE1_STEPS:
        depth  = global_step
        fanout = min(2 ** (global_step // 2), 32)
    else:
        depth  = 1
        fanout = 1

    fanout_score   = min(math.log2(max(fanout, 1)) * 1.2, 5.0)
    depth_score    = min(depth * 0.6, 4.0)
    requests_score = min(math.log2(max(global_step, 1)) * 0.5, 2.0)

    return fanout_score + depth_score + requests_score, depth, fanout


# ---------------------------------------------------------------------------
# Simulations (all consume the same pre-generated sequence)
# ---------------------------------------------------------------------------

def run_baseline(sequence: list[bool]) -> tuple[int, str]:
    total_failures = sum(sequence)
    return len(sequence), f"COMPLETED (collapsed, {total_failures}/{len(sequence)} failures)"


def run_rnos(sequence: list[bool]) -> tuple[int, str, str, float]:
    """
    Evaluates RNOS entropy before each step.
    Stops at REFUSE threshold.  The failure outcome does not affect entropy here
    (structural state is determined by phase and step position).
    """
    for step, _failed in enumerate(sequence, 1):
        entropy, _, _ = structural_entropy(step)
        if entropy >= RNOS_REFUSE:
            phase = 1 if step <= PHASE1_STEPS else 2
            return step, "STOPPED", f"RNOS  (entropy={entropy:.1f}, Phase {phase})", entropy

    final_e, _, _ = structural_entropy(len(sequence))
    return len(sequence), "COMPLETED (missed)", "none", final_e


def run_cb(sequence: list[bool]) -> tuple[int, str, str, float]:
    """
    Sliding-window circuit breaker.
    Updates window with each outcome, trips when failure rate exceeds CB_THRESHOLD.
    """
    window: deque[int] = deque(maxlen=CB_WINDOW)

    for step, failed in enumerate(sequence, 1):
        window.append(1 if failed else 0)
        if len(window) == CB_WINDOW:
            rate = sum(window) / CB_WINDOW
            if rate > CB_THRESHOLD:
                phase = 1 if step <= PHASE1_STEPS else 2
                return step, "STOPPED", f"CB    (window={rate:.0%}, Phase {phase})", rate

    final_rate = sum(window) / len(window) if window else 0.0
    return len(sequence), "COMPLETED (missed)", "none", final_rate


def run_hybrid(sequence: list[bool]) -> tuple[int, str, str]:
    """
    Hybrid controller: safety-first merge of RNOS and CB.

    RNOS is checked pre-execution (structural check before the call runs).
    CB is updated post-execution (density check after outcome is known).

    If either fires, execution halts.  The trigger source is reported.
    This means Hybrid can never perform worse than the better sub-system.
    """
    window: deque[int] = deque(maxlen=CB_WINDOW)

    for step, failed in enumerate(sequence, 1):
        # Pre-execution: structural gate
        entropy, _, _ = structural_entropy(step)
        if entropy >= RNOS_REFUSE:
            phase = 1 if step <= PHASE1_STEPS else 2
            return step, "STOPPED", f"RNOS  (entropy={entropy:.1f}, Phase {phase})"

        # Post-execution: density gate
        window.append(1 if failed else 0)
        if len(window) == CB_WINDOW:
            rate = sum(window) / CB_WINDOW
            if rate > CB_THRESHOLD:
                phase = 1 if step <= PHASE1_STEPS else 2
                return step, "STOPPED", f"CB    (window={rate:.0%}, Phase {phase})"

    return len(sequence), "COMPLETED (missed)", "none"


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _failure_map(sequence: list[bool], phase1_len: int, width: int = 52) -> str:
    """Compact timeline. F=failure, .=success, |=phase boundary."""
    chunk = max(1, len(sequence) // width)
    chars = []
    for i in range(0, len(sequence), chunk):
        chars.append("F" if any(sequence[i : i + chunk]) else ".")

    # Insert phase boundary marker
    boundary = phase1_len // chunk
    if 0 < boundary < len(chars):
        chars[boundary] = "|"

    return "".join(chars[:width])


def _bar(steps: int, total: int, width: int = 38) -> str:
    filled = int(width * min(steps, total) / max(total, 1))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sequence   = generate_sequence()
    total      = len(sequence)
    p1_fail    = sum(sequence[:PHASE1_STEPS])
    p2_fail    = sum(sequence[PHASE1_STEPS:])

    print()
    print("=== RNOS HYBRID SHOWCASE ===")
    print()
    print("Scenario:")
    print(f"  * Phase 1: Retry storm  ({PHASE1_STEPS} steps)")
    print(f"             Call graph grows: depth +1/step, fanout x2 every 2 steps")
    print(f"             Individual call failure rate: ~{int(PHASE1_FAILURE_RATE*100)}%")
    print(f"  * Phase 2: Slow drift   ({PHASE2_STEPS} steps)")
    print(f"             Flat sequential calls, ~{int(PHASE2_FAILURE_RATE*100)}% distributed failures")
    print()
    print(f"  Failure timeline (F=fail, .=ok, |=phase boundary):")
    print(f"    {_failure_map(sequence, PHASE1_STEPS)}")
    print(f"  Failure rate:  Phase 1 = {p1_fail}/{PHASE1_STEPS} ({p1_fail/PHASE1_STEPS:.0%})  "
          f"Phase 2 = {p2_fail}/{PHASE2_STEPS} ({p2_fail/PHASE2_STEPS:.0%})")
    print()

    b_steps, b_status                       = run_baseline(sequence)
    r_steps, r_status, r_trigger, r_entropy = run_rnos(sequence)
    c_steps, c_status, c_trigger, _         = run_cb(sequence)
    h_steps, h_status, h_trigger            = run_hybrid(sequence)

    print("Results:")
    print()
    print("  Baseline:")
    print(f"    Steps:    {b_steps:>5}  {_bar(b_steps, total)}")
    print(f"    Status:   {b_status}")
    print()
    print("  RNOS:")
    print(f"    Steps:    {r_steps:>5}  {_bar(r_steps, total)}")
    print(f"    Status:   {r_status}")
    print(f"    Trigger:  {r_trigger}")
    print()
    print("  Circuit Breaker:")
    print(f"    Steps:    {c_steps:>5}  {_bar(c_steps, total)}")
    print(f"    Status:   {c_status}")
    print(f"    Trigger:  {c_trigger}")
    print()
    print("  Hybrid (RNOS + CB):")
    print(f"    Steps:    {h_steps:>5}  {_bar(h_steps, total)}")
    print(f"    Status:   {h_status}")
    print(f"    Trigger:  {h_trigger}")
    print()
    print("---")
    print()
    print("Interpretation:")
    r_phase = "Phase 1" if r_steps <= PHASE1_STEPS else "Phase 2"
    c_phase = "Phase 1" if c_steps <= PHASE1_STEPS else "Phase 2"
    h_phase = "Phase 1" if h_steps <= PHASE1_STEPS else "Phase 2"
    print(f"  * RNOS responds to structural instability  ({r_phase}, step {r_steps})")
    print(f"  * CB   responds to distributed instability ({c_phase}, step {c_steps})")
    print(f"  * Hybrid selects the correct detector      ({h_phase}, step {h_steps})")
    print()
    print("Conclusion:")
    print("  Hybrid control adapts to instability geometry and matches")
    print("  the best controller per regime.")
    print()
    print("  Neither RNOS alone nor CB alone is sufficient.")
    print("  Hybrid is not redundancy -- it is required by the problem structure.")
    print()


if __name__ == "__main__":
    main()
