# RNOS Hybrid Gate — Scripts

A minimal "compute gate" that enforces safe execution boundaries on a CI pipeline
by composing two orthogonal instability observers.

## What this is

A **compute gate** evaluates instability signals at each pipeline step and decides
whether execution is permitted to continue:

- `ALLOW`   — safe to proceed
- `DEGRADE` — proceed but flagged; instability is building
- `REFUSE`  — halt immediately; execution would exceed safe boundaries

The gate is **pre-execution** in principle: it checks accumulated state before
committing to the next unit of work.  In this implementation the gate runs after
each step's outcome is recorded, so a REFUSE terminates the step and prevents all
subsequent steps from running.

## Scripts

| Script | Role |
|--------|------|
| `simulate_failure.ps1` | Returns the deterministic outcome (0/1) for a given step |
| `update_state.ps1`     | Updates `.rnos/state.json` after an outcome is observed |
| `rnos_gate.ps1`        | RNOS-only structural entropy gate |
| `hybrid_gate.ps1`      | Hybrid gate: RNOS + Circuit Breaker, safety-first merge |

## Controllers

### RNOS (structural instability)

Computes four entropy components from the pipeline state:

```
retry_score           = min(consecutiveFailures    * 0.8,  3.0)
fanout_score          = min(jobsSpawned            * 0.4,  5.0)
repeated_target_score = min(repeatedTargetFailures * 0.5,  2.0)
cost_score            = min(computeMinutes         * 0.05, 1.0)
entropy               = sum of above
```

Thresholds (from `rnos-policy.json`, matching `configs/default.yaml`):

```
DEGRADE  entropy >= 4.5
REFUSE   entropy >= 7.0
```

Formula mirrors `experiments/ci_control/controllers.py:RNOSCIController`.

### Circuit Breaker (density instability)

Maintains a rolling window of the last 5 step outcomes.
Fires REFUSE when the window is full and `failure_rate > 0.6`.

### Hybrid merge rule

```
if RNOS entropy >= refuse_entropy → REFUSE (trigger: RNOS)
elif CB window full AND rate > threshold → REFUSE (trigger: CB)
elif RNOS entropy >= degrade_entropy → DEGRADE (trigger: RNOS)
else → ALLOW
```

Hybrid can never perform worse than the better sub-system.

## Failure sequence and expected output

The deterministic failure sequence `[0, 1, 1, 1, 1, 0, 1, 0, 1, 0]` produces:

```
Step 1  failure=0  entropy=0.50  CB=1/5 0.00   →  ALLOW
Step 2  failure=1  entropy=2.70  CB=2/5 0.50   →  ALLOW
Step 3  failure=1  entropy=4.90  CB=3/5 0.67   →  DEGRADE  (RNOS)
Step 4  failure=1  entropy=7.10  CB=4/5 0.75   →  REFUSE   (RNOS)
Steps 5–10: skipped (gate closed)
```

RNOS catches the structural burst at step 4 (3 consecutive failures, entropy 7.1).
Without RNOS, the CB window would fill to rate=0.80 at step 6 — two steps later.

## How to run locally

```powershell
# From repo root
./scripts/update_state.ps1 -Init

for ($step = 1; $step -le 10; $step++) {
    $f = [int](./scripts/simulate_failure.ps1 -Step $step)
    ./scripts/update_state.ps1 -Failure $f -Step $step
    ./scripts/hybrid_gate.ps1  -Step $step
    if ($LASTEXITCODE -ne 0) { break }
}
```

## Policy

Gate thresholds are in `rnos-policy.json` at the repo root.
State is read from and written to `.rnos/state.json` each step.
