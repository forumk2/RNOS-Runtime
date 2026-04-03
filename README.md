# RNOS Runtime

> Compute like water. Contain like fire.

**RNOS Runtime is a control layer for autonomous AI systems that determines when execution should stop.**

In dry-run testing with seed=4 and 20 max steps, RNOS reduced tool failures by 94% and terminated runaway execution 16 steps before an uncontrolled baseline.

![RNOS Comparison Chart](docs/comparison_chart.png)

---

## Experimental Results

Three experiments test RNOS's ability to discriminate between recoverable instability and structural failure across progressively harder scenarios. The core claim is not that RNOS catches more failures than a circuit breaker — it is that RNOS uses a composite signal that accumulates across execution history rather than a windowed rate, and that this difference affects detection timing under bursty, recurring failure patterns.

Each experiment runs three strategies against the same scenarios:

- **RNOS** — entropy-based policy with fixed thresholds (degrade at 9.0, refuse at 11.0)
- **Adaptive circuit breaker** — sliding-window failure-rate breaker with exponential backoff and adaptive threshold
- **Baseline** — unprotected execution

### Summary

| Experiment | Scenarios | RNOS selectivity | CB selectivity | Baseline selectivity | Key finding |
|---|---|---|---|---|---|
| 2: Selective Containment | 4 (3 scored) | 3/3 | 3/3 | 2/3 | Both strategies match; baseline cannot discriminate |
| 2.5: Matched-Entropy Discrimination | 4 | 4/4 | 4/4 | 2/4 | Correct discrimination at step 8, one step after scenarios diverge |
| 3: Intermittent Cascading Failure | 4 | 4/4 | 4/4 | 2/4 | RNOS detects 7 steps earlier via cumulative floor; CB forgives recovery windows |

---

### Selective Containment (Experiment 2)

Four scenarios with clear ground-truth labels: `transient_blip` and `rough_patch` (recoverable), `runaway_cascade` (structural failure), and `slow_burn` (excluded as borderline). RNOS and the adaptive CB both achieve 3/3 selectivity; the baseline achieves 2/3 by correctly handling recoverable cases but running without limit on the cascade.

On `runaway_cascade`, RNOS issues REFUSE at step 7, resulting in 3 wasted steps (steps executed in the absorbing failure regime). The baseline runs all 20 steps, producing 16 wasted steps.

A phase-transition sweep varies `rough_patch` failure length from 1 to 10, holding all other parameters fixed. The result is a clean two-step transition:

| Failure length | RNOS behavior |
|---|---|
| 1–3 | No intervention |
| 4 | DEGRADE at step 7 |
| 5+ | REFUSE at step 7 |

The CB triggers at the same step for length 4+, but ends in `max_steps_exhausted` rather than clean refusal. The behavioral difference is in termination style, not intervention step.

---

### Matched-Entropy Discrimination (Experiment 2.5)

This experiment removes the entropy-magnitude advantage. Two scenarios — `matched_recovery` and `matched_collapse` — are constructed with identical failure schedules through step 6. The step-6 entropy assertion is verified empirically: both scenarios produce entropy 7.000 at step 6, with an absolute difference of 0.0.

At step 7, entropy is still identical for both scenarios (8.950). RNOS issues ALLOW for both. This is the correct behavior: withholding judgment when the evidence is genuinely ambiguous is not a limitation, it is the expected result of a policy that responds to observable signal rather than speculation.

Discrimination occurs at step 8, one step after the scenarios diverge in their actual outcomes:

| Scenario | Step 8 entropy | Decision |
|---|---|---|
| matched_recovery | 6.125 | ALLOW |
| matched_collapse | 10.810 | DEGRADE |

At step 9, `matched_collapse` reaches entropy 11.225 and RNOS issues REFUSE. RNOS 4/4, adaptive CB 4/4, baseline 2/4.

The mechanism here is delayed magnitude discrimination. Once the post-divergence outcome is visible, the entropy gap opens by 4.685 in a single step. RNOS does not infer the future trajectory — it responds to the observed signal as soon as it is informative.

---

### Intermittent Cascading Failure (Experiment 3)

This is the most structurally complex experiment. The two primary scenarios share the same burst-and-recovery surface pattern but differ in whether recovery is genuine.

**`bursty_recovery`**: two short failure bursts (3 failures, then 2 failures) separated by a recovery window, followed by sustained success. Ground truth: recoverable.

**`intermittent_cascade`**: three failure bursts (3 failures each) with dirty recovery windows between them. Recovery windows have persistently elevated latency. The third burst arrives at step 14, after a deceptively long 3-step recovery window (steps 11–13). Ground truth: structural failure.

Both strategies produce correct final decisions on all four scenarios (4/4). The difference is in when and how they detect the cascade.

#### Step 11 divergence

At step 11 — the first step of recovery window 2, immediately after burst 2's third consecutive failure — RNOS and the adaptive CB make different decisions for the first time.

RNOS entropy composition at step 11 evaluation:

| Component | Value |
|---|---|
| retry_score (3 consecutive failures) | 3.0 |
| cost_score (cumulative calls, saturated) | 2.0 |
| repeated_tool | 2.0 |
| failure_score (3 failures in last 5) | 1.95 |
| latency_score (430 ms) | 0.215 |
| **Total entropy** | **9.165** |

Result: DEGRADE (threshold 9.0).

Adaptive CB at step 11: window = [S, S, F, F, F] = 3/5 = 0.60. The CB uses a strict `>` check. 0.60 does not exceed 0.60. Result: ALLOW.

The CB issues its first intervention at step 18, when burst 3 has accumulated four consecutive failures and the window reaches [S, F, F, F, F] = 0.80 > 0.60. RNOS's first DEGRADE precedes the CB's first action by 7 steps.

#### Cross-burst memory

The RNOS structural floor — `cost_score` (2.0) + `repeated_tool` (2.0) = 4.0 — is the source of the timing difference. `cost_score` is computed as `min(cumulative_calls * 0.3, 2.0)`. It reaches its cap at 7 executed steps and remains at 2.0 through all subsequent recovery windows. It does not reset on success.

This means that by step 11, before a single failure-specific signal is added, the entropy baseline is already 4.0. A burst of 3 consecutive failures then contributes `retry_score` (3.0) and `failure_score` (1.95), pushing entropy to 9.165. The same 3-consecutive-failure burst in a fresh run with no prior execution history would produce entropy of approximately 3.64 — well below the DEGRADE threshold.

The CB has no equivalent. Its sliding window discards history older than 5 steps. The CB window at step 13 (end of recovery window 2) is [F, F, F, S, S] = 0.40 — it has largely forgotten burst 1 and partially forgotten burst 2. RNOS at the same step shows entropy 6.1, still elevated, with the cost_score floor intact.

#### Behavior on bursty_recovery

RNOS peak entropy on `bursty_recovery` is 8.650 at step 6 (burst 1 end). This is 0.35 below the DEGRADE threshold. RNOS issues no intervention and the task completes in 20 steps. The CB similarly issues no intervention (peak window rate 0.60 at multiple steps, never exceeding strict threshold).

The 0.35 margin between `bursty_recovery`'s peak (8.650) and `intermittent_cascade`'s burst-2 peak (9.165) reflects the burst length difference: `bursty_recovery` has 2 failures in burst 2, `intermittent_cascade` has 3. That difference of one failure shifts the retry_score by 1.0 and the failure_score by 0.65, a combined 1.65 change on top of the 4.0 structural floor.

#### Summary of behavioral difference

| Property | RNOS | Adaptive CB |
|---|---|---|
| First action on intermittent_cascade | Step 11 (DEGRADE, temporary caution) | Step 18 (OPEN, hard block) |
| Mechanism | Composite signal accumulation | Sliding-window failure rate |
| Cross-burst memory | Persistent via cost_score floor | None — window discards history |
| Behavior at recovery windows | Entropy remains elevated (6.1 at step 13) | Rate drops to 0.40 at step 13 |
| Response on bursty_recovery | No intervention (peak 8.650, below threshold) | No intervention |

---

### Key Takeaways

- RNOS and the adaptive CB produce identical selectivity scores across all three experiments. The difference is not in whether they get the right answer, but in how and when they get it.
- RNOS detects structural instability earlier when entropy accumulates across burst boundaries. The mechanism is not a separate memory structure — it is the monotonically increasing `cost_score` acting as a passive execution budget.
- The adaptive CB forgives recovery windows. A 3-step recovery between burst 2 and burst 3 is enough to reduce the CB's window rate from 0.60 to 0.40, substantially reducing its sensitivity to the next burst.
- RNOS does not over-trigger on genuinely recoverable burst patterns. `bursty_recovery` stays 0.35 below the DEGRADE threshold throughout.
- In all experiments, RNOS correctly withholds judgment when evidence is ambiguous (Experiment 2.5, step 7) and acts when evidence is sufficient. This is the same behavior a well-calibrated circuit breaker exhibits, expressed through a different signal composition.

---

### Limitations

**RNOS is not predictive.** Experiment 2.5 confirms this directly: RNOS cannot act at step 7 when scenarios are entropy-matched, and correctly does not. Detection requires observable divergence in the actual execution trace. Systems that fail silently or whose divergence is delayed past the entropy observation window will not be detected earlier than a breaker.

**Detection timing depends on threshold calibration.** The DEGRADE threshold (9.0) and REFUSE threshold (11.0) used in these experiments are hand-tuned to account for the structural floor produced by `repeated_tool` and `cost_score` in the specific scenario structure. Different execution patterns, tool diversities, or latency profiles would shift these thresholds. The current results do not generalize directly to uncalibrated configurations.

**The adaptive CB is a single baseline.** A more sophisticated multi-signal breaker that also tracks consecutive failure count and cumulative execution cost alongside windowed failure rate might narrow or close the 7-step detection gap observed in Experiment 3. The comparison in this paper is specifically between RNOS and a sliding-window rate threshold — not between RNOS and the space of all possible circuit breaker designs.

**All scenarios use deterministic synthetic schedules.** Real-world failure modes are stochastic, often correlated, and may involve transient partial recoveries that produce mixed signals. Behavior on real system traces may differ from what is observed here.

**Entropy composition weights are not optimized.** The weights used (`retry: 1.0`, `failure: 0.65`, `latency: 0.5`, `cost: 0.3`) were set by hand and have not been tuned against a validation set. The relative contribution of each signal to discrimination outcomes could shift under different weight assignments.

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
