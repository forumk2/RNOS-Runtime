# RNOS Runtime

> Compute like water. Contain like fire.

**RNOS Runtime is a control layer for autonomous AI systems that determines when execution should stop.**

In dry-run testing with seed=4 and 20 max steps, RNOS reduced tool failures by 94% and terminated runaway execution 16 steps before an uncontrolled baseline.

![RNOS Comparison Chart](docs/comparison_chart.png)

---

## Experimental Results

### 1. Overview

These experiments test RNOS across four progressively harder discrimination tasks: baseline containment, evidence-driven timing, cross-burst memory, and distributed instability. The goal is to characterize where entropy-based control works, where it fails, and what the failure boundary looks like mechanically.

RNOS uses a cumulative entropy score that captures retry depth, recent failure density, structural cost, and latency. The adaptive circuit breaker (CB) uses a sliding failure-rate window with a strict threshold. These are complementary detection architectures: entropy accumulates state across time; the sliding window measures local density within a fixed horizon. Experiments 2 and 3 favor RNOS; Experiment 4 favors CB.

Each experiment runs three strategies against the same scenarios:

- **RNOS** — entropy-based policy with fixed thresholds (degrade at 9.0, refuse at 11.0)
- **Adaptive circuit breaker** — sliding-window failure-rate breaker with exponential backoff and adaptive threshold
- **Baseline** — unprotected execution

| Experiment | Scenario Type | RNOS Selectivity | CB Selectivity | Baseline Selectivity | Key Finding |
|---|---|---|---|---|---|
| 2 — Selective Containment | Cascade vs recoverable instability | 3/3 | 3/3 | 2/3 | Both strategies match; baseline cannot discriminate |
| 2.5 — Matched-Entropy Discrimination | Identical-state divergence | 4/4 | 4/4 | 2/4 | RNOS withholds judgment until signal is observable |
| 3 — Intermittent Cascading Failure | Bursty failure with deceptive recovery | 4/4 | 4/4 | 2/4 | RNOS detects 7 steps earlier via cumulative entropy |
| 4 — Distributed Instability | Smoldering, diffuse failure | **3/4** | **4/4** | 2/4 | CB detects what RNOS misses; entropy ceiling exposed |

---

### 2. Selective Containment (Experiment 2)

RNOS correctly contains a runaway cascade while allowing recoverable instability to complete. This establishes the minimum requirement for a useful control policy: selectivity over blunt intervention.

Phase transition sweep (varying failure length):

| Failure run length | RNOS decision |
|---|---|
| 1–3 | No intervention |
| 4 | DEGRADE |
| 5+ | REFUSE |

The transition is clean and two-step. RNOS refuses the runaway cascade at step 7 (3 wasted steps). The baseline exhausts all 20 steps (16 wasted steps). Selectivity: RNOS 3/3, CB 3/3, Baseline 2/3.

---

### 3. Evidence-Driven Behavior (Experiment 2.5)

Two scenarios — `matched_recovery` and `matched_collapse` — were constructed with identical failure schedules through step 6. The step-6 entropy assertion is verified empirically: both scenarios produce entropy 7.000 at step 6, absolute difference 0.0.

At step 7, entropy is still identical for both scenarios (8.950). RNOS issues ALLOW for both. This is correct: withholding judgment when the evidence is genuinely ambiguous is not a limitation, it is the expected result of a policy that responds to observable signal rather than speculation. The assertion methodology proves the scenarios were identical when RNOS withheld judgment.

Discrimination occurs at step 8, one step after the scenarios diverge in their actual outcomes:

| Scenario | Step 8 entropy | Decision |
|---|---|---|
| matched_recovery | 6.125 | ALLOW |
| matched_collapse | 10.810 | DEGRADE |

At step 9, `matched_collapse` reaches entropy 11.225 and RNOS issues REFUSE. Selectivity: RNOS 4/4, CB 4/4, Baseline 2/4.

The mechanism is delayed magnitude discrimination. Once the post-divergence outcome is visible, the entropy gap opens by 4.685 in a single step. RNOS does not infer future trajectory — it responds to the observed signal as soon as it is informative.

---

### 4. Intermittent Cascading Failure (Experiment 3)

This is RNOS's strongest result. The two primary scenarios share the same burst-and-recovery surface pattern but differ in whether recovery is genuine.

**`bursty_recovery`**: two short failure bursts separated by a recovery window, followed by sustained success. Ground truth: recoverable.

**`intermittent_cascade`**: three failure bursts with dirty recovery windows between them. Recovery windows have persistently elevated latency. The third burst arrives at step 14, after a deceptively long 3-step recovery window. Ground truth: structural failure.

Both strategies produce correct final decisions on all four scenarios (4/4). The difference is detection timing.

#### Step 11 divergence

At step 11 — immediately after burst 2's third consecutive failure — RNOS and the adaptive CB make different decisions for the first time.

RNOS entropy composition at step 11:

| Component | Value |
|---|---|
| retry_score (3 consecutive failures) | 3.0 |
| cost_score (cumulative calls, saturated) | 2.0 |
| repeated_tool | 2.0 |
| failure_score (3 failures in last 5) | 1.95 |
| latency_score (430 ms) | 0.215 |
| **Total entropy** | **9.165 → DEGRADE** |

Adaptive CB at step 11: window [S,S,F,F,F] = 3/5 = 0.60. The CB uses a strict `>` check — 0.60 does not exceed 0.60. Result: ALLOW.

The CB issues its first intervention at step 18, when the window reaches 0.80 > 0.60. RNOS's first DEGRADE precedes the CB's first action by 7 steps.

#### Cross-burst memory

The RNOS structural floor — `cost_score` (2.0) + `repeated_tool` (2.0) = 4.0 — is the source of the timing difference. `cost_score` reaches its cap at 7 executed steps and does not reset on success. By step 11, before any failure-specific signal is added, the entropy baseline is already 4.0. The same 3-consecutive-failure burst in a fresh run would produce entropy of approximately 3.64 — well below the DEGRADE threshold.

The CB has no equivalent. Its sliding window discards history older than 5 steps. The CB window at step 13 (end of recovery window 2) is [F,F,F,S,S] = 0.40 — burst 1 is gone, burst 2 is fading. RNOS at the same step shows entropy 6.1, still elevated, with the cost_score floor intact.

#### Behavior on bursty_recovery

RNOS peak entropy on `bursty_recovery` is 8.650, which is 0.35 below the DEGRADE threshold. RNOS issues no intervention and the task completes in 20 steps. The CB similarly issues no intervention. The 0.35 margin separating the two scenarios reflects the burst length difference: `bursty_recovery` has 2 failures in burst 2, `intermittent_cascade` has 3 — a one-failure difference that shifts combined retry and failure scores by 1.65 on top of the 4.0 structural floor.

---

### 5. Distributed Instability (Experiment 4)

This is the most important section for an honest account of RNOS's limitations.

`smoldering_instability` maintains a 30–40% failure rate across 20 steps with no consecutive run exceeding 2 failures. `noisy_recovery` has an identical failure schedule through step 10 and then genuinely stabilizes. The entropy-band assertion confirms the scenarios are indistinguishable through the noisy phase:

- noisy_recovery max entropy (steps 3–10): 7.11
- smoldering max entropy (steps 3–10): 7.11
- diff: 0.0

**RNOS result:** No intervention on `smoldering_instability`. Peak entropy: 8.805. DEGRADE threshold: 9.0. Miss gap: 0.195 units.

This is not a threshold calibration issue — it is structural. Under a ≤2 consecutive failure constraint, the maximum reachable entropy is bounded:

| Component | Max value | Reason |
|---|---|---|
| retry_score | 2.0 | consecutive failures capped at 2 |
| failure_score | 2.6 | at most 4/5 recent failures |
| structural floor | 4.0 | cost_score + repeated_tool, always present |
| latency_score | ~0.2 | 410ms prev-step latency |
| **Ceiling** | **~8.8** | |

RNOS cannot reach the 9.0 DEGRADE threshold when consecutive failures are limited to 2, regardless of how long the instability persists.

**CB result:** Detects `smoldering_instability` at step 18. The FFSFF pattern in steps 13–17 fills the sliding window with 4/5 = 0.80, exceeding the 0.60 threshold. CB accumulates failure density within its window without regard for whether failures are consecutive — the structural property RNOS's retry-based scoring cannot replicate.

**Persistence metrics** were logged observationally and are not part of the RNOS entropy computation:

| Metric | noisy_recovery | smoldering_instability |
|---|---|---|
| stability_score (final) | 9 | 0 |
| chronic_instability_flag | 0 | 1 |
| above_floor_count (final) | 9 | 14 |
| rolling_failure_rate_10 (final) | 0.1 | 0.4 |
| avg_latency_last_5 | 80ms | 282ms |

These signals cleanly separate the two scenarios. `stability_score` diverges by step 15; `chronic_instability_flag` activates unambiguously on smoldering after step 10 and never activates on noisy_recovery. The discrimination signal exists in the data. The current entropy formula does not capture it.

---

### 6. Key Takeaways

- RNOS and CB have complementary detection profiles. Framing them as competitors misrepresents the results.
- RNOS detects structured cascading failure earlier: 7-step advantage on `intermittent_cascade` (Experiment 3), explained by cumulative entropy preserving cross-burst state that CB's sliding window discards.
- CB detects distributed failure density better: CB catches `smoldering_instability` at step 18; RNOS does not catch it at any step (Experiment 4).
- RNOS has a structural blind spot when consecutive failure streaks are capped at ≤2. The retry-based component of entropy cannot rise high enough to trigger DEGRADE under that constraint, regardless of sustained failure rate.
- The persistence signals logged in Experiment 4 (stability_score, chronic_instability_flag, above_floor_count) clearly separate the scenarios RNOS cannot distinguish. These are observational only and are not currently modeled in the entropy formula.

---

### 7. Limitations

**RNOS is not predictive.** Experiment 2.5 confirms this directly: RNOS cannot act at step 7 when scenarios are entropy-matched, and correctly does not. Detection requires observable divergence in the actual execution trace.

**Threshold calibration dependency.** The 0.195 entropy gap between RNOS's peak (8.805) and the DEGRADE threshold (9.0) in Experiment 4 quantifies an architectural boundary. Closing this gap by lowering the threshold risks false positives on `noisy_recovery`, which reaches 7.11 in the noisy phase. The threshold cannot be adjusted without introducing a new tradeoff.

**No persistence modeling.** RNOS does not incorporate signals for sustained failure rate, stability streaks, or time-above-floor. Experiment 4 demonstrates that these signals are sufficient to separate the two distributed scenarios; the current entropy composition is not.

**Evaluated on synthetic deterministic schedules.** All scenarios use fixed failure schedules without stochastic variation. Results may not generalize to real workloads where failure timing, latency distributions, and recovery patterns are non-deterministic.

**CB is a strong baseline, not a strawman.** The adaptive circuit breaker matches RNOS selectivity on three of four experiments and outperforms it on the fourth. Results should be interpreted as a characterization of detection profiles, not a demonstration of RNOS superiority.

**Entropy weights are hand-tuned.** Component coefficients and caps were set by design, not by optimization. Different weight assignments would produce different detection boundaries.

---

## How It Works

RNOS evaluates every proposed action before execution using two signals:

**Entropy** — a composite instability score derived from:
- Execution depth (how deep in the call chain)
- Retry count (consecutive failures)
- Recent failure rate (last 5 actions)
- Tool repetition (same tool called repeatedly)
- Planner latency (LLM inference time as a stress signal)
- Cumulative cost (total work done in the loop)

**Trust** — a confidence score (0.0–1.0) based on recent success rate, penalized by entropy.

These combine into three decisions:

| Decision | Condition | Effect |
|---|---|---|
| **ALLOW** | entropy < 3.0, trust > 0.45 | Execute normally |
| **DEGRADE** | entropy 3.0–6.0 or trust 0.2–0.45 | Execute with constraints (no side effects, limited retries) |
| **REFUSE** | entropy ≥ 6.0 or trust ≤ 0.2 | Terminate execution |

![Entropy Progression](docs/entropy_progression.png)

Entropy accumulates across steps as failures compound. Trust degrades inversely. In the test run above, RNOS transitioned from ALLOW (steps 1–2) to DEGRADE (step 3, entropy=3.8) to REFUSE (step 4, entropy=6.35) as instability escalated. The circuit breaker reached the same endpoint through 14 more steps.

---

## Terminal Output

```
$ python scripts/run_agent.py --max-steps 20 --seed 4 --dry-run

[DRY RUN] LM Studio not called -- planner returns 'CALL unstable_api' always
=== LM Studio RNOS Loop ===
mode=rnos seed=4 max_steps=20 persona=adversarial

[step 01] llm_output='CALL unstable_api' depth=0 entropy=0.000 trust=0.850 decision=ALLOW retry_count=0
           tool_result=SUCCESS (API call succeeded)
           phase=stable call_count=1 failure_streak=0

[step 02] llm_output='CALL unstable_api' depth=1 entropy=1.900 trust=1.000 decision=ALLOW retry_count=0
           tool_result=SUCCESS (API call succeeded)
           phase=stable call_count=2 failure_streak=0

[step 03] llm_output='CALL unstable_api' depth=2 entropy=3.800 trust=0.883 decision=DEGRADE retry_count=0
           degraded_mode=True constraints={"allow_side_effects": false, "max_additional_steps": 1}
           tool_result=FAILURE (transient_failure)
           phase=unstable call_count=3 failure_streak=1

[step 04] llm_output='CALL unstable_api' depth=3 entropy=6.350 trust=0.337 decision=REFUSE retry_count=1
           stop=RNOS refused execution
```

---

## Architecture

```
User
  |
Agent (LLM Planner)
  |
RNOS Runtime  <-- evaluates every action before execution
  |
Tools (APIs, DB, File System)
```

RNOS sits between decision and action. It does not replace the planner — it gates the planner's output.

---

## Compared Approaches

### RNOS (Entropy-Aware Gate)
- Tracks system-wide instability across six weighted signals
- Graduated response: ALLOW → DEGRADE → REFUSE
- Terminates the loop on REFUSE — saves both tool and planner compute
- First intervention at step 3 via DEGRADE (one constrained retry allowed)

### Circuit Breaker (Exponential Backoff)
- Standard production pattern (AWS, gRPC, Kubernetes)
- Binary: allow or block (no degraded mode)
- Blocks tool calls but the agent loop keeps running — planner still infers on every blocked step
- Recovery probes into a dead endpoint waste additional tool calls
- Reached PERMANENTLY_OPEN at step 18 after 10 blocked steps

### Baseline (No Intervention)
- Loop runs until step budget exhausted
- All failures absorbed, no termination signal
- 18 of 20 calls failed; execution continued regardless

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

# Baseline (no protection)
python scripts/run_agent.py --max-steps 20 --seed 4 --no-rnos

# Dry run (no LM Studio needed)
python scripts/run_agent.py --max-steps 20 --seed 4 --dry-run
```

### Run All Three and Generate Report
```bash
python scripts/run_comparison.py --max-steps 20 --seed 4 --tag "my-test"

# With live LM Studio
python scripts/run_comparison.py --max-steps 20 --seed 4 --tag "live-qwen3-4b"

# Dry run
python scripts/run_comparison.py --max-steps 20 --seed 4 --dry-run --tag "verify"
```

### Generate Report from Existing Data
```bash
python scripts/generate_report.py --tag "my-test"
python scripts/generate_report.py --seed 4
python scripts/generate_report.py --no-chart   # skip PNG generation
```

Results are saved to `results/runs.jsonl`. Reports and charts go to `results/`.

### Planner Personas
```bash
# Adversarial (default): "retry forever"
python scripts/run_agent.py --max-steps 15 --seed 4 --persona adversarial

# Cautious: "stop after two failures"
python scripts/run_agent.py --max-steps 15 --seed 4 --persona cautious

# Mixed: "try 3 times then switch tools"
python scripts/run_agent.py --max-steps 15 --seed 4 --persona mixed
```

---

## Project Structure

```
rnos/                  # Core runtime
  entropy.py           # Entropy calculation (6 weighted signals)
  trust.py             # Trust model (success-rate baseline minus entropy penalty)
  policy.py            # ALLOW/DEGRADE/REFUSE policy engine
  runtime.py           # Main evaluation loop
  types.py             # Shared data structures

baselines/             # Non-RNOS comparison strategies
  circuit_breaker.py   # Exponential backoff circuit breaker

agent/                 # LLM planner integration
  planner.py           # LM Studio OpenAI-compatible client
  parser.py            # Action parser (CALL <tool> [payload])
  loop.py              # Agent loop (legacy, see run_agent.py)

tools/                 # Tool implementations
  unstable_api.py      # Failure-prone API simulation
  calculator.py        # Safe arithmetic tool
  file_ops.py          # Sandboxed file operations

scripts/               # Entry points
  run_agent.py                 # Single-mode runner
  run_comparison.py            # Three-way batch runner
  generate_report.py           # Markdown + chart report generator
  generate_entropy_chart.py    # Entropy/trust progression chart

docs/                  # README assets (committed)
results/               # Run data (gitignored)
```

---

## Why Refusal Matters

As AI agents become more capable, they also become more unpredictable. They take actions, call tools, make decisions in loops. Traditional approaches — monitoring, logging, retrying, scaling — answer "what happened?" but not "should this continue?"

RNOS introduces refusal as a first-class primitive. Instead of retrying indefinitely or continuing blindly, the system can determine that execution has become unsafe and stop.

This is not about making systems perfect. It is about making systems that know when they have lost the right to continue.

> A system should know when it has lost the right to continue.

---

## License

MIT

## Author

Rowan Ashford
