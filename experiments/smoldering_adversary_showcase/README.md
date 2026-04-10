# Smoldering Adversary Showcase

This showcase models a deterministic low-and-slow adaptive pressure agent on the same synthetic service graph used by the burstier adversarial showcase. It is a defensive research harness for comparing Baseline, RNOS, CB, and HYBRID under a gradual strain pattern.

## What A Smoldering Adversary Means Here

- lower per-step intensity
- slower ramp
- more blending and cooldown behavior
- fewer sharp bursts
- repeated moderate pressure on slow-recovering services
- damage that grows through recovery debt and fatigue rather than one dramatic step

The action set is symbolic only. There is no real exploit logic, scanning, credential behavior, shell execution, payload generation, or network attack behavior.

## Why Low-And-Slow Pressure Matters

Not all dangerous adaptive pressure arrives as a burst. Some of it looks reasonable if you inspect a single step in isolation. The strain appears across time:

- services never fully recover
- moderate failure density accumulates
- background wear rises even when no step looks catastrophic
- operators may see drift instead of a clear incident moment

This makes the scenario useful for illustrating the complementarity between RNOS and CB:

- RNOS can damp structural pressure early when entropy begins to rise
- CB can contribute when repeated moderate failures accumulate across its rolling window
- HYBRID can benefit from both

## How This Differs From The Bursty Adversary

- Bursty adversary: leverage-seeking, sharper concentration, earlier entropy spikes.
- Smoldering adversary: slow drift, fatigue accumulation, target rotation, delayed detection.

The smoldering scenario is intentionally tuned so that individual steps look less dramatic while cumulative damage still becomes meaningful.

## Scenario Mechanics

The graph reuses the same logical services:

- `public_api`
- `auth_service`
- `cache`
- `database`
- `worker_pool`
- `internal_admin_endpoint`
- `billing_service`

Additional synthetic properties drive the low-and-slow behavior:

- `recovery_lag`
- `latent_fragility`
- `sustained_load_penalty`
- per-step `recovery_debt`
- accumulated `fatigue`

The agent moves through four phases:

1. `Blend`: broad sampling with low visible instability.
2. `Lean`: moderate sustained pressure on one or two services.
3. `Drift`: repeated persistence with pivots that avoid obvious spikes.
4. `Smolder`: lower-throughput persistence that keeps recovery debt elevated.

## How To Run

Comparative run:

```bash
python -m experiments.smoldering_adversary_showcase.run --scenario smoldering_adversary --mode all --seed 42
```

Single-mode runs:

```bash
python -m experiments.smoldering_adversary_showcase.run --scenario smoldering_adversary --mode rnos --seed 42
python -m experiments.smoldering_adversary_showcase.run --scenario smoldering_adversary --mode hybrid --seed 1337
```

Canonical seed comparison:

```bash
python -m experiments.smoldering_adversary_showcase.run --scenario smoldering_adversary --mode all --all-seeds
```

Optional tuning:

```bash
python -m experiments.smoldering_adversary_showcase.run --mode all --seed 42 --stealth 0.78 --budget 32
```

## Artifacts

Each run writes under `results/smoldering_adversary_showcase/seed_<seed>/`:

- `summary.json`
- `summary.md`
- `timeline.csv`
- `events_<mode>.txt`
- `timeline_entropy.png`
- `pressure_damage.png`
- `service_pressure_breakdown.png`
- `recovery_debt_over_time.png`

## Interpretation Notes

The intended reading is modest and scenario-specific:

- `baseline` carries the longest-tail damage because moderate pressure keeps compounding.
- `rnos` may damp some steps but can miss or delay hard containment while entropy remains soft.
- `cb` can matter even without sharp bursts because repeated moderate failures eventually accumulate in-window.
- `hybrid` should usually perform best overall by combining partial early damping with later hard containment.

These are synthetic observations, not claims about real-world offensive behavior.

## Limitations

- This models adaptive pressure, not real exploitation.
- The strain variables are synthetic abstractions chosen to make cumulative wear legible.
- Results are deterministic per seed and intended as a behavioral demonstration, not a calibrated benchmark.
