# Retry Storm Showcase

A proof artifact demonstrating RNOS early containment vs baseline explosion.

## What this demonstrates

A 70% failure rate API with fanout-2 retry branching is supercritical (net amplification 1.4x per wave). Without control, calls grow exponentially. RNOS stops the cascade at 20 calls. The baseline hits the 10,000-call safety cap — a **99.8% reduction**.

## How to run

```
python experiments/retry_storm_showcase/run.py
```

No dependencies beyond the standard library.

## Expected outcome

```
Baseline (no control):
  Total Calls:   10,000
  Status:       COLLAPSED

RNOS:
  Total Calls:       20
  Status:       REFUSED (early)

Calls Reduced:  99.8%
```

## Why it works

RNOS uses **pre-execution admissibility checking**: each prospective operation is costed against an entropy budget *before* it runs. Deep retry chains are expensive; when the budget is exhausted, the next operation is refused at the gate — not after the cascade has already multiplied. Reactive controls (circuit breakers, rate limiters) wait for evidence of failure; RNOS refuses structurally dangerous execution before the first byte is sent.
