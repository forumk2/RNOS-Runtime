# RNOS Runtime

**RNOS Runtime is an experimental control layer for AI agent loops that enforces early containment via a graduated refusal primitive.**

When an agent loop becomes unstable — retrying failed tools, compounding errors, accumulating structural cost — RNOS evaluates each proposed action against a cumulative entropy score and issues one of three decisions: **ALLOW**, **DEGRADE**, or **REFUSE**. The loop terminates on REFUSE. No action is taken without passing the gate.

Traditional approaches (circuit breakers, retry limits, monitoring) detect failure after it accumulates. RNOS gates execution before each action, using state that persists across the entire run — not just a recent window.

---

## Quick Example

```
$ python scripts/run_agent.py --max-steps 20 --seed 4 --dry-run

[step 01] entropy=0.000  trust=0.850  decision=ALLOW   → SUCCESS
[step 02] entropy=1.900  trust=1.000  decision=ALLOW   → SUCCESS
[step 03] entropy=3.800  trust=0.883  decision=DEGRADE → FAILURE  (side effects disabled)
[step 04] entropy=6.350  trust=0.337  decision=REFUSE  → stop
```

RNOS terminated at step 4. An unprotected baseline ran all 20 steps; 18 failed. The adaptive circuit breaker reached the same endpoint at step 18 via binary block/allow cycling.

---

## How It Works

RNOS evaluates two signals before each action:

**Entropy** — a composite instability score. Six weighted components accumulate across the run:

| Component | Captures |
|---|---|
| `retry_score` | Consecutive failures |
| `failure_score` | Failure rate over the last 5 actions |
| `cost_score` | Total executed steps (cumulative, does not reset on success) |
| `repeated_tool` | Same tool called repeatedly |
| `latency_score` | Planner inference time as a stress signal |
| `depth_score` | Execution depth in the call chain |

**Trust** — a confidence score (0.0–1.0) based on recent success rate, penalized by entropy.

These combine into three decisions:

| Decision | Condition | Effect |
|---|---|---|
| **ALLOW** | entropy < 3.0, trust > 0.45 | Execute normally |
| **DEGRADE** | entropy 3.0–6.0 or trust 0.2–0.45 | Execute with constraints: no side effects, limited retries |
| **REFUSE** | entropy ≥ 6.0 or trust ≤ 0.2 | Terminate the loop |

REFUSE terminates the entire agent loop, not just the current tool call. This means neither tool execution nor planner inference continues — both sources of wasted compute stop.

---

## Key Design Properties

**Cumulative state.** `cost_score` reaches its cap at 7 steps and does not reset when the agent succeeds. This creates a structural entropy floor that grows with run length, independently of recent failure rate. A 3-failure burst in a fresh run looks different to RNOS than the same burst at step 11 of a long run.

**Reactive, not predictive.** RNOS does not infer future trajectory. It responds to observable signal in the execution trace. When two scenarios are entropy-matched, RNOS withholds judgment — this is correct behavior, not a limitation.

**Complementary to circuit breakers.** RNOS and circuit breakers have different detection profiles. RNOS's cumulative entropy gives it an advantage on structured cascading failure. Circuit breakers' sliding-window density gives them an advantage on diffuse, non-consecutive failure. The experiments below characterize this boundary precisely.

---

## Experimental Results

Four experiments test RNOS across progressively harder discrimination tasks. Each runs RNOS, an adaptive circuit breaker (CB), and an unprotected baseline against the same scenarios. The goal is to characterize where entropy-based control works, where it fails, and what the failure boundary looks like mechanically.

- **RNOS** — entropy-based policy with fixed thresholds (degrade at 9.0, refuse at 11.0)
- **Adaptive CB** — sliding-window failure-rate breaker with exponential backoff and adaptive threshold
- **Baseline** — unprotected execution

| Experiment | Scenario Type | RNOS | CB | Hybrid | Baseline | Key Finding |
|---|---|---|---|---|---|---|
| 2 — Selective Containment | Cascade vs. recoverable instability | 3/3 | 3/3 | — | 2/3 | Both strategies match; baseline cannot discriminate |
| 2.5 — Matched-Entropy Discrimination | Identical-state divergence | 4/4 | 4/4 | — | 2/4 | RNOS withholds judgment until signal is observable |
| 3 — Intermittent Cascading Failure | Bursty failure with deceptive recovery | 4/4 | 4/4 | — | 2/4 | RNOS detects 7 steps earlier via cumulative entropy |
| 4 — Distributed Instability | Diffuse, non-consecutive failure | **3/4** | **4/4** | — | 2/4 | CB detects what RNOS misses; entropy ceiling exposed |
| 5 — Hybrid Cooperative Control | Cascading burst + distributed low-rate | 7 exec | 10 exec | **7 exec** | 30 exec | Hybrid ≥ best(RNOS, CB) in both geometries; trigger source identified per scenario |

---

### Experiment 2 — Selective Containment

RNOS correctly contains a runaway cascade while allowing recoverable instability to complete. This is the minimum requirement for a useful control policy: selectivity over blunt intervention.

Phase transition sweep (varying failure run length):

| Failure run length | RNOS decision |
|---|---|
| 1–3 | ALLOW |
| 4 | DEGRADE |
| 5+ | REFUSE |

RNOS refuses the runaway cascade at step 7 (3 wasted steps). The baseline exhausts all 20 steps (16 wasted). Selectivity: RNOS 3/3, CB 3/3, Baseline 2/3.

---

### Experiment 2.5 — Evidence-Driven Behavior

Two scenarios — `matched_recovery` and `matched_collapse` — have identical failure schedules through step 6. Entropy is verified identical at step 6 (7.000, absolute difference 0.0) and still identical at step 7 (8.950). RNOS issues ALLOW for both at step 7.

This is correct. When two scenarios are entropy-matched, withholding judgment is the right outcome. A policy that acted at step 7 would be speculating, not detecting.

Discrimination occurs at step 8, one step after the scenarios diverge:

| Scenario | Step 8 entropy | Decision |
|---|---|---|
| `matched_recovery` | 6.125 | ALLOW |
| `matched_collapse` | 10.810 | DEGRADE |

At step 9, `matched_collapse` reaches 11.225 and RNOS issues REFUSE. The entropy gap between scenarios opened by 4.685 in a single step — the post-divergence signal is unambiguous. Selectivity: 4/4.

---

### Experiment 3 — Intermittent Cascading Failure

This is RNOS's strongest result. Two scenarios share the same surface burst-and-recovery pattern but differ in structural outcome.

- **`bursty_recovery`**: two short failure bursts, genuine recovery, sustained success. Ground truth: recoverable.
- **`intermittent_cascade`**: three failure bursts with elevated-latency recovery windows. Third burst arrives at step 14, after a deceptively clean 3-step recovery window. Ground truth: structural failure.

Both strategies reach correct final decisions (4/4). The difference is when.

**Step 11 divergence.** At step 11, after burst 2's third consecutive failure, RNOS and CB make different decisions for the first time.

RNOS entropy at step 11:

| Component | Value |
|---|---|
| retry_score (3 consecutive failures) | 3.0 |
| cost_score (saturated at 7 steps) | 2.0 |
| repeated_tool | 2.0 |
| failure_score (3/5 recent) | 1.95 |
| latency_score (430 ms) | 0.215 |
| **Total** | **9.165 → DEGRADE** |

CB at step 11: window [S,S,F,F,F] = 3/5 = 0.60. The CB threshold uses a strict `>` check — 0.60 does not exceed 0.60. Result: ALLOW. The CB issues its first intervention at step 18 (window reaches 0.80). RNOS precedes it by 7 steps.

**Why.** The RNOS structural floor — `cost_score` (2.0) + `repeated_tool` (2.0) = 4.0 — exists before any failure-specific signal is added at step 11. The same 3-consecutive-failure burst in a fresh run would produce entropy ~3.64, well below DEGRADE. The CB has no equivalent mechanism: its sliding window at step 13 is [F,F,F,S,S] = 0.40 — burst 1 is gone, burst 2 is fading. RNOS at the same step shows 6.1.

**On `bursty_recovery`.** RNOS peak entropy is 8.650 — 0.35 below DEGRADE. No intervention; the task completes in 20 steps. The 0.35 margin is mechanically explained: `bursty_recovery` has 2 failures in burst 2 vs. 3 in `intermittent_cascade`, shifting retry and failure scores by 1.65 on top of the 4.0 floor.

---

### Experiment 4 — Distributed Instability

This experiment defines RNOS's structural boundary.

`smoldering_instability` maintains a 30–40% failure rate across 20 steps with no consecutive run exceeding 2 failures. `noisy_recovery` has an identical failure schedule through step 10, then genuinely stabilizes. The entropy-band assertion confirms the scenarios are indistinguishable through the noisy phase:

- `noisy_recovery` max entropy (steps 3–10): 7.11
- `smoldering_instability` max entropy (steps 3–10): 7.11
- diff: 0.0

**RNOS result.** No intervention on `smoldering_instability`. Peak entropy: 8.805. DEGRADE threshold: 9.0. Miss gap: 0.195 units.

This is not a calibration issue — it is structural. Under a ≤2 consecutive failure constraint, the entropy ceiling is bounded:

| Component | Max value | Reason |
|---|---|---|
| retry_score | 2.0 | consecutive failures capped at 2 |
| failure_score | 2.6 | at most 4/5 recent failures |
| structural floor | 4.0 | cost_score + repeated_tool |
| latency_score | ~0.2 | 410 ms latency |
| **Ceiling** | **~8.8** | |

RNOS cannot reach the 9.0 DEGRADE threshold when consecutive failures are capped at 2, regardless of how long the instability persists. Lowering the threshold to close the gap would cause false positives on `noisy_recovery`, which reaches 7.11 during its noisy phase. The tradeoff cannot be resolved within the current entropy composition.

**CB result.** Detects `smoldering_instability` at step 18. The FFSFF pattern in steps 13–17 fills the window with 4/5 = 0.80, exceeding the 0.60 threshold. The CB accumulates failure density regardless of consecutiveness — the structural property RNOS's retry-based scoring cannot replicate.

**Persistence signals** were logged observationally and are not part of the RNOS entropy computation:

| Metric | `noisy_recovery` | `smoldering_instability` |
|---|---|---|
| stability_score (final) | 9 | 0 |
| chronic_instability_flag | 0 | 1 |
| above_floor_count (final) | 9 | 14 |
| rolling_failure_rate_10 (final) | 0.1 | 0.4 |
| avg_latency_last_5 | 80 ms | 282 ms |

`stability_score` diverges by step 15. `chronic_instability_flag` activates on smoldering after step 10 and never activates on `noisy_recovery`. The discrimination signal exists in the data; the current entropy formula does not capture it.

---

---

### Experiment 5 — Hybrid Cooperative Control

Experiments 1–4 established that RNOS and CB have complementary detection profiles. Experiment 5 asks: does composing them into a single hybrid controller produce a dominant architecture — one that is at least as good as either sub-system on every failure geometry?

The hybrid uses a **safety-first merge**: RNOS and CB both evaluate each step; the more-severe decision wins. A `trigger_source` field records which sub-system drove each intervention ("rnos", "cb", or "both").

Two scenarios target each sub-system's known strength:

**Scenario A — `cascading_burst`** (RNOS strength): 7 consecutive failures beginning at step 3, absorbing thereafter. RNOS's `retry_score` accumulates 1.0 per consecutive failure, crossing the DEGRADE threshold before the CB's 10-step window fills.

**Scenario B — `distributed_low_rate`** (CB strength): repeating F-F-S pattern (67% failure rate, ≤2 consecutive). `retry_count` resets every third step, capping `retry_score` at 2.0. RNOS entropy peaks at 8.7 — 0.3 below DEGRADE (9.0). The CB's window fills with 7/10 failures after 10 executions and trips.

Results (tool executions before termination):

| Scenario | Baseline | RNOS | CB | Hybrid | Best |
|---|---|---|---|---|---|
| `cascading_burst` | 30 | 7 | 10 | **7** | RNOS = Hybrid |
| `distributed_low_rate` | 30 | 30 | 10 | **10** | CB = Hybrid |

**Trigger source** confirms the mechanism: hybrid intervention on `cascading_burst` is `"rnos"` (CB window not yet full); on `distributed_low_rate` it is `"cb"` (RNOS never reaches its threshold).

**Conclusion:** Hybrid performs ≥ best(RNOS, CB) in both scenarios and strictly outperforms each sub-system on at least one axis — 3 fewer wasted executions than CB on cascading failure, 20 fewer than RNOS on distributed failure. The safety-first merge is sufficient to achieve cooperative dominance without requiring coordination between sub-systems.

---

### Key Takeaways

- RNOS and CB have complementary detection profiles. Framing them as competitors misrepresents the results.
- RNOS detects structured cascading failure earlier: 7-step advantage on `intermittent_cascade`, explained by cumulative entropy preserving cross-burst state that CB's sliding window discards.
- CB detects distributed failure density better: catches `smoldering_instability` at step 18; RNOS does not catch it at any step.
- RNOS has a structural blind spot when consecutive failure streaks are capped at ≤2. The retry-based entropy component cannot rise high enough to trigger DEGRADE under that constraint, regardless of sustained failure rate.
- **Hybrid composition (Experiment 5) resolves the complementarity directly.** A safety-first merge of RNOS + CB matches or beats both sub-systems on every tested failure geometry. The `trigger_source` field makes the contributing sub-system observable per-step.
- The persistence signals logged in Experiment 4 clearly separate the scenarios RNOS cannot distinguish. These are observational only and are not currently modeled in the entropy formula.

---

## Limitations

**RNOS is not predictive.** Detection requires observable divergence in the execution trace. Experiment 2.5 confirms this directly: when two scenarios are entropy-matched, RNOS withholds judgment and correctly does nothing.

**Structural entropy ceiling.** When consecutive failures are capped at ≤2, the maximum reachable entropy (~8.8) falls below the DEGRADE threshold (9.0). RNOS cannot detect diffuse, non-consecutive instability regardless of threshold adjustment without introducing false positives on recoverable scenarios.

**No persistence modeling.** RNOS does not model sustained failure rate, stability streaks, or time-above-floor. Experiment 4 shows these signals are sufficient to discriminate the scenarios RNOS misses. They are not currently part of the entropy composition.

**Evaluated on synthetic deterministic schedules.** All scenarios use fixed failure schedules. Results may not generalize to real workloads with stochastic failure timing, variable latency distributions, or non-deterministic recovery patterns.

**Entropy weights are hand-tuned.** Component coefficients and caps were set by design, not optimization. Different weight assignments produce different detection boundaries.

**CB is a strong baseline, not a strawman.** The adaptive circuit breaker matches RNOS selectivity on three of four experiments and outperforms it on the fourth. These results characterize two complementary detection profiles; they do not establish RNOS superiority.

---

## Quick Start

### Prerequisites

- Python 3.11+
- LM Studio (optional — `--dry-run` works without it)

### Install

```bash
pip install -e .
```

### Run a Single Mode

```bash
# RNOS (default)
python scripts/run_agent.py --max-steps 20 --seed 4

# Circuit breaker
python scripts/run_agent.py --max-steps 20 --seed 4 --circuit-breaker

# Hybrid (RNOS + AdaptiveCircuitBreaker, safety-first merge)
python scripts/run_agent.py --max-steps 20 --seed 4 --hybrid

# Baseline (no protection)
python scripts/run_agent.py --max-steps 20 --seed 4 --no-rnos

# Dry run (no LM Studio required)
python scripts/run_agent.py --max-steps 20 --seed 4 --dry-run
```

### Run All Four Modes and Generate Report

```bash
python scripts/run_comparison.py --max-steps 20 --seed 4 --tag "my-test"
python scripts/run_comparison.py --max-steps 20 --seed 4 --dry-run --tag "verify"
```

### Run Experiment 5 (Hybrid Cooperative Control)

```bash
python experiments/experiment_5_hybrid/run_experiment_5.py
python experiments/experiment_5_hybrid/run_experiment_5.py --seed 42 --max-steps 30
```

Results are written to `results/experiment_5/` (per-step CSVs) and `docs/experiment_5_hybrid.md`.

### Generate Report from Existing Data

```bash
python scripts/generate_report.py --tag "my-test"
python scripts/generate_report.py --seed 4
python scripts/generate_report.py --no-chart   # skip PNG generation
```

Results are saved to `results/runs.jsonl`. Reports and charts go to `results/`.

### Planner Personas

```bash
# Adversarial (default): retries indefinitely
python scripts/run_agent.py --max-steps 15 --seed 4 --persona adversarial

# Cautious: stops after two failures
python scripts/run_agent.py --max-steps 15 --seed 4 --persona cautious

# Mixed: retries three times then switches tools
python scripts/run_agent.py --max-steps 15 --seed 4 --persona mixed
```

---

## Architecture

```
User
  |
Agent (LLM Planner)
  |
RNOS Runtime  <-- gates every proposed action before execution
  |
Tools (APIs, DB, File System)
```

RNOS sits between the planner and execution. It does not replace the planner — it evaluates the planner's output before any action is taken.

### RNOS vs. Circuit Breaker vs. Hybrid

| Property | RNOS | Circuit Breaker | Hybrid |
|---|---|---|---|
| State model | Cumulative across full run | Sliding window (recent N steps) | Both |
| Response | Graduated: ALLOW / DEGRADE / REFUSE | Binary: allow or block | Graduated (max-severity merge) |
| On REFUSE | Terminates agent loop | Blocks tool; planner keeps running | Terminates agent loop |
| Advantage | Structured cascading failure (cross-burst memory) | Diffuse failure density (non-consecutive) | ≥ best of both on any geometry |
| Trigger visibility | entropy + trust signals | window failure rate | `trigger_source`: "rnos" / "cb" / "both" |
| Standard | Experimental | Production (AWS, gRPC, Kubernetes) | Experimental |

---

## Project Structure

```
rnos/
  entropy.py           # Entropy calculation (6 weighted components)
  trust.py             # Trust model (success-rate baseline minus entropy penalty)
  policy.py            # ALLOW / DEGRADE / REFUSE policy engine
  runtime.py           # Main evaluation loop
  hybrid.py            # HybridController (RNOS + CB, safety-first merge)
  types.py             # Shared data structures

baselines/
  circuit_breaker.py          # Exponential-backoff circuit breaker
  adaptive_circuit_breaker.py # Sliding-window adaptive circuit breaker

agent/
  planner.py           # LM Studio OpenAI-compatible client
  parser.py            # Action parser (CALL <tool> [payload])
  loop.py              # Agent loop (legacy)

tools/
  unstable_api.py      # Failure-prone API simulation
  calculator.py        # Safe arithmetic tool
  file_ops.py          # Sandboxed file operations

scripts/
  run_agent.py                 # Single-mode runner (--rnos / --circuit-breaker / --hybrid / --no-rnos)
  run_comparison.py            # Four-way batch runner + report
  generate_report.py           # Markdown + chart report generator
  generate_entropy_chart.py    # Entropy / trust progression chart

experiments/
  experiment_5_hybrid/
    scenarios.py       # cascading_burst and distributed_low_rate scenario definitions
    run_experiment_5.py # 4-mode batch runner; writes CSVs + docs/experiment_5_hybrid.md

docs/                  # Analysis reports (committed)
results/               # Run data (gitignored)
```

---

## Motivation

AI agent loops can continue executing after they have become unstable. Traditional safeguards — monitoring, retry limits, circuit breakers — detect failure after it accumulates or block individual calls while the loop keeps running. Neither provides a principled answer to the question: *should this execution continue at all?*

RNOS treats refusal as a first-class primitive. The loop terminates when accumulated evidence — across depth, retries, failure rate, latency, and structural cost — crosses a threshold. The system does not retry indefinitely or degrade silently; it stops and says why.

This is an experimental exploration of that primitive, not a production system.

---

## License

MIT

## Author

Rowan Ashford
