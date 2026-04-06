# Slow Drift Showcase

A credibility demonstration: RNOS has a known structural boundary, and a circuit breaker fills the gap.

## What this demonstrates

Distributed instability — failures scattered across time with no long consecutive streaks — does not produce the structural expansion that RNOS is designed to detect. RNOS runs all 150 steps without triggering. A sliding-window circuit breaker stops at step 51 when failure density in a 5-step window hits 80%.

**This is intentional and correct behavior. RNOS and CB are complementary, not competing.**

## How to run

```
python experiments/slow_drift_showcase/run.py
```

No dependencies beyond the standard library.

## Expected outcome

```
Baseline:         150 steps  COMPLETED (unstable)
RNOS:             150 steps  COMPLETED (missed)   peak entropy: 2.60 / 10.0
Circuit Breaker:   51 steps  STOPPED (detected)   window rate: 80%
```

## Why RNOS misses this

RNOS entropy measures structural expansion: call-graph fanout, cascade depth, cumulative request volume. For sequential, non-branching API calls (fanout=1, depth=1), the entropy ceiling is ~2.6 — far below the 8.0 degrade and 10.0 refuse thresholds. No retry branching means no expanding call graph, which means no RNOS signal. Smoldering failures with no cascade are structurally invisible to it.

## Why the circuit breaker succeeds

The CB tracks failure density within a rolling window, independent of structure. It does not care whether failures cascade or not — only whether the recent failure rate exceeds the threshold. Distributed failures that fill a 5-step window (e.g. 4 out of 5 = 80%) trip the breaker cleanly. This is exactly the signal RNOS cannot see.
