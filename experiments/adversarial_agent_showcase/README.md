# Synthetic Adversarial Agent Showcase

This showcase simulates adaptive pressure from a deterministic synthetic agent against a small service graph. It is a defensive research harness for comparing Baseline, RNOS, CB, and HYBRID behavior under the same reproducible trajectory.

## What It Is

- A seeded, policy-driven agent that probes a simulated graph of services such as `public_api`, `auth_service`, `database`, `worker_pool`, and `internal_admin_endpoint`.
- A behavioral model of adaptive pressure geometry: probing, retries, fanout, pivots, and lower-and-slower persistence.
- A controller comparison harness that reuses the repo's existing RNOS runtime, adaptive circuit breaker, and hybrid merge.

## What It Is Not

- Not real exploitation.
- Not vulnerability research on any real software.
- Not scanning, shell execution, credential use, payload generation, or network attack logic.

The action set is symbolic only. Each action translates into synthetic load on a deterministic service graph.

## Why This Matters

Frontier models are increasingly agentic. A system does not need literal exploit code to become operationally dangerous: adaptive retries, concentrated pressure, pivots across service structure, and persistence after partial containment can still create instability.

The point of this showcase is to test containment behavior against autonomous stress patterns, not to simulate real intrusion techniques.

## Scenario Shape

The agent moves through four phases:

1. `Explore`: shallow probing across visible services.
2. `Focus`: concentrated pressure on the highest-scoring target.
3. `Escalate`: wider pressure chains and pivot attempts toward higher-leverage paths.
4. `Persist`: lower-and-slower pressure on a secondary path after containment or exhaustion on the primary path.

Target scores are deterministic and combine:

- privilege value
- sensitivity
- retry amplification
- dependency depth
- containment already encountered

Supported symbolic actions:

- `probe(target)`
- `retry(target)`
- `fanout(targets)`
- `pivot(target)`
- `cooldown()`
- `persist(target)`
- `blend_in()`

## Controllers

- `baseline`: no containment, full synthetic pressure is admitted.
- `rnos`: pre-execution entropy gate from the repo's RNOS runtime.
- `cb`: adaptive circuit breaker from `baselines/adaptive_circuit_breaker.py`.
- `hybrid`: safety-first merge of RNOS and CB via `rnos/hybrid.py`.

## How To Run

Full comparative showcase with canonical seed:

```bash
python -m experiments.adversarial_agent_showcase.run --scenario adversarial_agent --mode all --seed 42
```

Single-mode runs:

```bash
python -m experiments.adversarial_agent_showcase.run --scenario adversarial_agent --mode baseline --seed 1337
python -m experiments.adversarial_agent_showcase.run --scenario adversarial_agent --mode hybrid --seed 7
```

Canonical seed comparison:

```bash
python -m experiments.adversarial_agent_showcase.run --scenario adversarial_agent --mode all --all-seeds
```

Optional tuning:

```bash
python -m experiments.adversarial_agent_showcase.run --mode all --seed 42 --stealth 0.55 --goal-bias disruption --budget 28
```

## Artifacts

Each run writes under `results/adversarial_agent_showcase/seed_<seed>/`:

- `summary.json`
- `summary.md`
- `timeline.csv`
- `events_<mode>.txt`
- `timeline_entropy.png`
- `pressure_damage.png`
- `service_pressure_breakdown.png`

When `--all-seeds` is used, the runner also writes `results/adversarial_agent_showcase/seed_comparison.md`.

## Reading The Output

The summary reports:

- total simulated calls
- refused calls
- completion step
- peak entropy
- first refusal step
- first CB trip step
- top pressured service
- cumulative damage score
- pivots attempted
- whether the persistence phase still succeeded

Expected interpretation for the default seed is modest:

- `baseline` usually carries the highest damage and longest uninterrupted escalation.
- `rnos` tends to intervene earlier on concentrated, structurally risky pressure.
- `cb` tends to react once bad outcomes become dense enough in its window.
- `hybrid` usually matches the earlier of RNOS or CB on this synthetic trajectory.

These are scenario-specific observations, not general claims.

## Interpretation Notes

- This simulates adaptive pressure, not real exploitation.
- The goal is to test containment behavior against autonomous stress patterns.
- Results are deterministic per seed and intended as a behavioral demonstration.
- Minor path differences across seeds are expected, but the broad story should stay stable.
