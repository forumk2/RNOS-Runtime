# Hybrid Showcase

The third in a series of three showcases. Demonstrates that no single controller is
sufficient across mixed instability regimes, and that Hybrid control is not redundancy
— it is required by the structure of the problem.

## What this demonstrates

A two-phase execution sequence exposes each controller's blind spot:

- **Phase 1 (structural instability):** A retry call graph expands exponentially — depth
  grows by one and fanout doubles every two steps. RNOS sees this structural explosion
  and refuses at step 8. The circuit breaker stays quiet because the raw per-call failure
  rate does not yet saturate its density window.

- **Phase 2 (distributed instability):** Sequential flat calls with ~40% distributed
  failures and no long streaks. RNOS sees nothing (entropy ~2.5, far below threshold).
  The circuit breaker detects rising failure density and trips in Phase 2.

Hybrid checks both signals at every step. It fires at step 8 via RNOS — earlier than
either controller would fire alone if the other phase dominated first.

## How to run

```
python experiments/hybrid_showcase/run.py
```

No dependencies beyond the standard library.

## Expected outcome

```
Baseline:         162 steps  COMPLETED (collapsed)
RNOS:               8 steps  STOPPED   trigger: RNOS  (entropy=10.3, Phase 1)
Circuit Breaker:   14 steps  STOPPED   trigger: CB    (window=80%,  Phase 2)
Hybrid:             8 steps  STOPPED   trigger: RNOS  (Phase 1)
```

## Multi-geometry instability

Real systems fail in multiple modes simultaneously. A single controller optimized for one
failure geometry has a structural blind spot for the other:

| Failure geometry | RNOS | CB |
|---|---|---|
| Retry cascade (deep, branching) | detects | misses until window fills |
| Slow drift (flat, distributed) | misses entirely | detects |

Hybrid resolves this by composing both observers under a safety-first merge rule: the
highest-severity signal wins. The cost of a false safe is low (unnecessary halt); the
cost of a missed signal is high (uncontrolled cascade or silent degradation).

## Why Hybrid works

Hybrid is two independent observers with orthogonal signal domains sharing a single
decision gate. RNOS measures structural complexity (call-graph expansion). CB measures
temporal failure density (sliding window). These signals are independent — neither can
be derived from the other — which means combining them provides strictly more coverage
than either alone, with no increase in false-positive rate for scenarios where only one
signal is present.
