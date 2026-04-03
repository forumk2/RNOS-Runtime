# D3: Deterministic Degradable Distributed
## Refusal as a First-Class Execution Primitive

**Rowan Ashford**

*Working draft v0.2 — April 2026*

---

## Abstract

We present RNOS (Runtime for Naturally Operating Systems), an experimental runtime that treats refusal as a first-class execution primitive rather than an error-handling afterthought. RNOS gates every proposed action against an entropy budget — a composite instability score that accumulates across the full execution run — and issues one of three decisions (ALLOW, DEGRADE, REFUSE) before any action is taken. At fanout-16, RNOS contains cascade growth to 65 contexts against 69,905 unprotected (1,075× reduction) while consuming 22.5 units of entropy against 1,216,337. Under adversarial composition, budget fragmentation attacks achieve 96% cascade execution before refusal, exposing a structural gap in RNOS's per-context greedy evaluation that is independently confirmed by per-step evaluation failures in a companion experiment suite. Experiments use synthetic deterministic workloads; entropy weights are hand-tuned. Results characterize the boundary between what entropy-gated refusal can and cannot contain — not a finished system.

---

## 1. Introduction

Recursive systems, distributed retry loops, and AI agent executors share a common failure mode: they continue executing after the justification for continuation has degraded. Traditional runtime safeguards — timeouts, retry limits, circuit breakers — answer the question "has this call failed?" They do not answer "should execution continue at all?"

The dominant response to this gap has been monitoring and post-hoc intervention: observe failure accumulating, then act. This paper explores a different position: that refusal should be a pre-execution primitive, evaluated before each action, using evidence that persists across the entire run rather than a recent sliding window.

We implement this position in RNOS, a runtime that enforces bounded execution via an entropy budget system. The budget is shared across all contexts in a run; each proposed action is evaluated against the remaining budget before it executes. When the budget is exhausted or a structural limit is reached, the runtime refuses — not the tool call, but the entire execution. No further actions are taken and no planner inference continues.

The contribution of this paper is threefold:

1. A concrete implementation of refusal as a pre-execution primitive, with a compositional entropy model and an effects staging system.
2. Experimental characterization of where entropy-gated refusal outperforms circuit-breaker approaches (structured cascading failure) and where it fails (diffuse non-consecutive failure, distributed cost fragmentation).
3. Identification of a shared architectural gap — greedy per-unit evaluation cannot detect instability distributed across individually-benign units — confirmed independently in two experimental regimes.

The paper proceeds as follows. Section 2 surveys related work. Section 3 describes the RNOS architecture. Section 4 explains the entropy composition model. Section 5 presents experimental results across three regimes: discrimination properties, containment properties, and adversarial analysis. Section 6 discusses implications. Section 7 enumerates limitations.

---

## 2. Background and Related Work

### 2.1 Circuit Breakers

The circuit breaker pattern, popularized by Hystrix and adopted in resilience4j, gRPC, and Kubernetes health check infrastructure, monitors failure rate within a sliding window and transitions between CLOSED, OPEN, and HALF-OPEN states. When the failure rate within the window exceeds a threshold, the breaker opens, blocking calls to the failing endpoint.

Circuit breakers are production-proven and widely deployed. Their sliding-window design is intentional: it provides recency sensitivity, allowing recovery to be detected quickly after the failure clears. The tradeoff is that the window discards older history. A failure pattern that distributes failures across time — two failures per burst, three bursts over 20 steps — may never fill the window past threshold even as accumulated instability is clearly present.

RNOS and circuit breakers are not competing approaches. They have different detection profiles. Experiments in Section 5 characterize the boundary precisely: RNOS detects structured cascading failure 7 steps earlier than the adaptive CB; the adaptive CB detects diffuse non-consecutive failure that RNOS cannot catch at all. The appropriate framing is complementary, not adversarial.

### 2.2 Chaos Engineering and Fault Injection

Chaos engineering (Chaos Monkey, Gremlin, LitmusChaos) identifies system weaknesses by deliberately injecting failures into running systems. Fault injection is diagnostic: it tests how systems respond to specific failure modes under controlled conditions.

RNOS is not fault injection. It is runtime control. Where chaos engineering introduces failures to observe response, RNOS gates execution to prevent responses that would cause unbounded expansion. The relationship is that chaos engineering might reveal the failure modes that RNOS is designed to contain; the two approaches operate at different phases of the system lifecycle.

### 2.3 Entropy in Systems

The term "entropy" in systems contexts is used in several ways. In information theory, entropy measures uncertainty in a probability distribution. In thermodynamics, it measures disorder. In distributed systems literature, entropy sometimes refers to state divergence in eventually-consistent systems.

RNOS uses entropy operationally: as a conservative estimate of execution uncertainty injected by an execution path, capturing branching depth, retry pressure, failure accumulation, and effect irreversibility. The kernel specification formalizes this as a bound on the reachable state-space: E_t(s) = log|R_t(s)|, where R_t(s) is the set of reachable states after t steps from state s. The bound is enforced as a finite budget B: execution is denied when estimated E_t(s) would exceed B.

This operational definition diverges from information-theoretic entropy but retains the key property: a finite entropy budget implies a finite number of entropy-expanding steps (Theorem 0.2 of the kernel specification). The budget is not a measure of information content; it is a schedulable resource analogous to CPU time or memory.

### 2.4 Related Runtime Approaches

Quota-based execution systems (AWS Lambda concurrency limits, Kubernetes resource quotas) cap compute consumption but do not respond to the structure of execution — a deep recursive chain and a flat parallel fanout consume the same resources but represent different structural risk profiles. RNOS's cost model charges more for depth and sibling expansion than for flat execution, making structural properties visible as cost.

Resource governors in database systems (query budget limits, query plan cost estimation) share the goal of bounding execution cost before it is committed. RNOS's gate evaluation before each context spawn is architecturally similar to a query cost estimator, but extended to cover execution depth, effect staging, and cross-context budget sharing.

---

## 3. Architecture

RNOS is structured as a small runtime with five core components: the entropy budget, the execution gate, the context model, the effects system, and the trace system.

### 3.1 Entropy Budget

`EntropyBudget` maintains three values: `total` (the initial allocation), `remaining` (budget not yet consumed), and `reserved_for_commit` (budget held for staged effects not yet committed). The admission test is:

```
can_afford(amount) ≡ (remaining − reserved_for_commit) ≥ amount
```

This two-level accounting separates task execution cost from effect commit cost. A context can be admitted, execute, stage effects, and then have those effects blocked if the sandbox policy prohibits them — the task start cost is charged, but the effect reservation is released. This produces the Refund Illusion vulnerability documented in Section 5.3.

The budget is shared across all contexts in a run. When a parent spawns a child context, both draw from the same `EntropyBudget` object. There is no per-context budget isolation; depletion by one context reduces what is available for all subsequent contexts. This is the mechanism by which Phase 1 of the composite cascade experiment depletes headroom for Phases 2–4.

### 3.2 Execution Gate

`ExecutionGate` performs four checks before admitting a context:

1. **Depth limit**: `context.depth > sandbox.max_depth`
2. **Fanout limit**: `next_sibling_index > sandbox.max_fanout`
3. **Descendants limit**: `root.descendant_count > sandbox.max_total_descendants`
4. **Budget sufficiency**: `not budget.can_afford(estimated_cost)`

All four checks run on every spawn. A context is refused if any check fails; all failing reasons are recorded. The gate evaluates each context independently at spawn time: it does not project aggregate cost across planned siblings, does not reserve budget for known future phases, and does not consider the cost trajectory implied by the current execution graph. This per-context greedy evaluation is the source of the Budget Fragmentation and Delayed Amplification vulnerabilities.

### 3.3 Cost Model

The `CostModel` estimates execution cost from structural properties:

```
task_cost(depth, sibling) =
  base_task_start (1.0)
  + depth × additional_depth_level (1.5) if depth > 0
  + child_spawn (2.5) if depth > 0
  + (sibling − 1) × additional_sibling (1.0) if sibling > 1
```

Effect costs: log/internal = 0.5, network = 4.0, local_write = 2.0, irreversible surcharge = 6.0. A sandbox boundary crossing adds 5.0.

An empirically-calibrated variant (`EMPIRICAL_COST_MODEL`) was derived via univariate sensitivity sweep and joint grid search over an adversarial scenario suite. The three structurally load-bearing parameters — `base_task_start`, `child_spawn`, `additional_depth_level` — were found to be at the lower boundary of the search range; doubling them improved aggregate containment by 7.1% across a 30-trial validation. The static defaults are used in all experiments reported here; the empirical model requires proportionally larger budgets to preserve the same admission count and is not yet the system default.

### 3.4 Effects System

Side effects are declared on each context via `declared_effects`. When a context emits an effect, the gate evaluates it against sandbox policy before staging: undeclared effects, network effects in a non-network sandbox, and write effects in a write-restricted sandbox are blocked. Allowed effects are staged (budget reserved) and committed at context completion; blocked effects trigger context rollback (budget reservation released, task start cost retained).

The effects system operates as an independent enforcement layer from the budget gate. When the budget gate admits a context but sandbox policy blocks its effects, the context rolls back without the effect costs being charged. This layering provided partial containment in the Refund Illusion attack (Section 5.3) that the budget gate alone did not.

### 3.5 Execution Modes

RNOS supports four graduated execution modes that correspond to increasing levels of restriction:

- **Green**: Nominal operation. All declared effects permitted, full depth and fanout available.
- **Yellow**: Light degradation. Side effects permitted but flagged; additional retry steps limited.
- **Orange**: Heavy degradation. No side effects; reduced fanout ceiling; escalation required for sandbox crossing.
- **Red**: Refusal. No execution. Structured refusal result returned with reasons.

The current implementation collapses Green/Yellow into ALLOW and Orange into DEGRADE; the four-level distinction is the intended target state.

### 3.6 Instruction Interface (Planned)

The kernel specification defines four primitive instructions for the entropy-gated execution model:

- **EBIND**: Bind an entropy budget to an execution context, establishing the budget envelope for the context's lifetime.
- **ITAG**: Attach an intent tag and declared effect list to a context, establishing what the context is permitted to do.
- **QEMIT**: Qualified effect emission — stage an effect against the gate before committing it.
- **TFALLBK**: Trust-based fallback — invoke a lower-trust execution path when the primary path would exceed budget.

These represent the kernel-level interface for entropy-gated execution. The current implementation realizes EBIND and QEMIT; ITAG is approximated by the `intent` field on contexts; TFALLBK is not yet implemented.

---

## 4. Entropy Composition

In RNOS-Runtime (the discrimination experiments), entropy is computed as a weighted sum of six signals evaluated before each agent step:

| Component | Captures | Notes |
|---|---|---|
| `retry_score` | Consecutive failures | Caps at 3 failures |
| `failure_score` | Failure rate over last 5 steps | 0.0–2.6 range |
| `cost_score` | Total executed steps (cumulative) | Caps at 7 steps; does not reset on success |
| `repeated_tool` | Same tool invoked repeatedly | Binary × weight |
| `latency_score` | Planner inference time | Milliseconds → entropy units |
| `depth_score` | Execution depth in call chain | Linear with depth |

The distinguishing property is `cost_score`: it grows monotonically with run length and does not reset when the agent succeeds. By step 7, `cost_score` reaches its cap of 2.0 and remains at 2.0 for the rest of the run regardless of subsequent success or failure. Combined with `repeated_tool` (2.0 when a tool is called repeatedly), this creates a structural floor of 4.0 that precedes any failure-specific signal.

### Why the Structural Floor Matters

In a fresh run, a 3-consecutive-failure burst produces entropy approximately:

```
retry_score (3 failures) = 3.0
failure_score (3/5) = 1.95
cost_score (3 steps) ≈ 1.0
repeated_tool = 2.0
latency_score ≈ 0.215
Total ≈ 8.165 (below DEGRADE threshold of 9.0)
```

The same burst at step 11 of a sustained run produces:

```
retry_score = 3.0
failure_score = 1.95
cost_score (saturated) = 2.0
repeated_tool = 2.0
latency_score = 0.215
Total = 9.165 (DEGRADE triggered)
```

The 1.0 unit difference between these outcomes is the structural floor's contribution above the early-run baseline. Without it, the burst at step 11 would not trigger DEGRADE; with it, it does. This is the mechanism behind RNOS's 7-step detection advantage on `intermittent_cascade` (Section 5.1.3).

A circuit breaker's sliding window has no equivalent. Its window at step 11 reflects only recent history; burst 1 (steps 4–6) has partially or fully faded by the time burst 2 arrives. RNOS retains the cost signal from burst 1 implicitly via `cost_score`, even though the failure events themselves are not directly visible.

### Why Composite Signals Differ from Single-Metric Thresholds

A single-metric threshold (failure rate > 0.60) is falsified by distributing failures across time. The structural floor is not falsifiable by distributing failures; it grows regardless of failure pattern, based solely on run length. The failure-specific components (`retry_score`, `failure_score`) can be suppressed by engineering a low failure rate; the structural components (`cost_score`, `repeated_tool`) cannot. This is also why the structural floor creates a blind spot: when consecutive failures are capped at ≤2, `retry_score` is capped at 2.0, and the maximum reachable entropy from failure-specific signals falls below the DEGRADE threshold regardless of how long the run continues (Section 5.1.4).

---

## 5. Experimental Results

### 5.1 Discrimination Properties (RNOS-Runtime)

Four experiments test whether RNOS can distinguish structurally different failure modes in a step-by-step agent execution setting. Each experiment runs RNOS, an adaptive circuit breaker (CB), and an unprotected baseline against matched scenario pairs.

- **RNOS**: entropy-based policy, degrade at 9.0, refuse at 11.0
- **Adaptive CB**: sliding-window failure-rate breaker, threshold 0.60, exponential backoff
- **Baseline**: unprotected; exhausts step budget

| Experiment | Scenario Type | RNOS | CB | Baseline | Key Finding |
|---|---|---|---|---|---|
| 2 — Selective Containment | Cascade vs. recoverable instability | 3/3 | 3/3 | 2/3 | Both match; baseline cannot discriminate |
| 2.5 — Matched-Entropy | Identical-state divergence | 4/4 | 4/4 | 2/4 | RNOS withholds judgment until signal is present |
| 3 — Intermittent Cascade | Bursty failure with deceptive recovery | 4/4 | 4/4 | 2/4 | RNOS detects 7 steps earlier |
| 4 — Distributed Instability | Diffuse, non-consecutive failure | **3/4** | **4/4** | 2/4 | CB detects what RNOS misses |

#### 5.1.1 Selective Containment (Experiment 2)

RNOS correctly refuses the runaway cascade at step 7 (3 wasted steps) while allowing recoverable instability to complete all 20 steps. The baseline runs all 20 steps on both scenarios (16 wasted). The failure-length phase transition is clean: failure runs of 1–3 produce no intervention, 4 produces DEGRADE, 5+ produces REFUSE. Selectivity: RNOS 3/3, CB 3/3, Baseline 2/3.

This result establishes the minimum criterion: selectivity over blunt intervention. A control policy that refuses everything or nothing is trivially correct on one end; usefulness requires discrimination.

#### 5.1.2 Matched-Entropy Discrimination (Experiment 2.5)

Two scenarios — `matched_recovery` and `matched_collapse` — were constructed with identical failure schedules through step 6. Entropy is empirically verified identical at step 6 (7.000, diff = 0.0) and step 7 (8.950). RNOS issues ALLOW for both at step 7.

This is correct behavior. When execution traces are entropy-matched, withholding judgment is the only defensible action. A policy that acted at step 7 would be responding to noise, not signal.

Discrimination occurs at step 8, one step after the scenarios diverge:

| Scenario | Step 8 entropy | Decision |
|---|---|---|
| `matched_recovery` | 6.125 | ALLOW |
| `matched_collapse` | 10.810 | DEGRADE |

The entropy gap opened by 4.685 in a single step. At step 9, `matched_collapse` reaches 11.225 and RNOS issues REFUSE. The mechanism is delayed magnitude discrimination, not trajectory prediction. RNOS responded to observable signal; it did not anticipate the divergence. Selectivity: 4/4.

#### 5.1.3 Intermittent Cascading Failure (Experiment 3)

`bursty_recovery` has two failure bursts with genuine recovery. `intermittent_cascade` has three bursts with elevated-latency dirty recovery windows. Both produce correct final decisions (4/4). The difference is timing.

At step 11, after `intermittent_cascade`'s burst 2 third consecutive failure, RNOS and CB diverge for the first time:

| Component | Value |
|---|---|
| retry_score (3 consecutive failures) | 3.0 |
| cost_score (saturated) | 2.0 |
| repeated_tool | 2.0 |
| failure_score (3/5 recent) | 1.95 |
| latency_score (430 ms) | 0.215 |
| **Total** | **9.165 → DEGRADE** |

CB at step 11: window [S,S,F,F,F] = 3/5 = 0.60. Strict `>` check: 0.60 does not exceed 0.60. Result: ALLOW. CB issues its first intervention at step 18 (window reaches 0.80 > 0.60). The detection gap is 7 steps.

At step 13 (end of recovery window 2): CB window [F,F,F,S,S] = 0.40 — burst 1 is gone, burst 2 is fading. RNOS entropy = 6.1, with the 4.0 structural floor intact.

`bursty_recovery` RNOS peak: 8.650, 0.35 below DEGRADE. No intervention; task completes in 20 steps. The 0.35 margin is mechanically explained: `bursty_recovery` has 2 failures in burst 2 vs. 3 in `intermittent_cascade` — a one-failure difference that shifts retry and failure scores by 1.65 above the 4.0 floor.

#### 5.1.4 Distributed Instability (Experiment 4)

`smoldering_instability` maintains 30–40% failure rate across 20 steps with no consecutive failure run exceeding 2. `noisy_recovery` has an identical failure schedule through step 10, then stabilizes. Entropy is verified identical through the noisy phase (both peak at 7.11, diff = 0.0).

**RNOS result**: No intervention on `smoldering_instability`. Peak entropy 8.805. DEGRADE threshold 9.0. Miss gap: 0.195 units.

This is not a calibration issue. Under ≤2 consecutive failures, the entropy ceiling is bounded:

| Component | Max value | Reason |
|---|---|---|
| retry_score | 2.0 | consecutive cap |
| failure_score | 2.6 | at most 4/5 recent failures |
| structural floor | 4.0 | cost_score + repeated_tool |
| latency_score | ~0.2 | 410 ms |
| **Ceiling** | **~8.8** | |

Lowering the threshold to close the 0.195 gap produces false positives on `noisy_recovery` (peak 7.11 during noisy phase). The tradeoff cannot be resolved within the current entropy composition.

**CB result**: Detects `smoldering_instability` at step 18. FFSFF in steps 13–17 fills window to 4/5 = 0.80 > 0.60. The CB's failure density accumulation is independent of consecutiveness — the structural property RNOS's retry-based scoring cannot replicate.

Persistence signals observed but not modeled in RNOS:

| Metric | `noisy_recovery` | `smoldering_instability` |
|---|---|---|
| stability_score (final) | 9 | 0 |
| chronic_instability_flag | 0 | 1 |
| rolling_failure_rate_10 (final) | 0.1 | 0.4 |
| avg_latency_last_5 | 80 ms | 282 ms |

`stability_score` diverges by step 15. `chronic_instability_flag` activates on smoldering after step 10 and never activates on `noisy_recovery`. The discrimination signal exists; the current entropy formula does not capture it.

---

### 5.2 Containment Properties (RNOS-2.0)

#### 5.2.1 Cascade Growth Baseline

A fanout cascade was evaluated at fanout 2, 4, 8, 16, comparing unprotected and RNOS-protected runs. The budget is shared; each spawn reduces available entropy for subsequent contexts.

| Fanout | Unprotected contexts | Protected contexts | Containment factor |
|---|---|---|---|
| 2 | 31 | 9 | 3.4× |
| 4 | 341 | 17 | 20.1× |
| 8 | 4,681 | 33 | 141.8× |
| 16 | 69,905 | 65 | 1,075.5× |

At fanout-16, RNOS limits execution to 65 contexts with 61 refusals, consuming 22.5 entropy units against 1,216,337 unprotected. The growth regime shifts from exponential to approximately linear with fanout count. The containment factor grows super-linearly because deeper trees exhaust the budget sooner in proportion to their total potential size.

#### 5.2.2 Composite Cascade

Four failure modes — retry storm, spawn race, partial partition, delayed rollback — were chained into a single execution with a shared 50-unit budget. Each phase inherits the depleted budget state from the prior phase; phases are not given fresh budgets.

The uncontrolled baseline (all phases complete, all effects succeed) produced 105.5 entropy units — 3.25× the entropy of the hardest individual phase (retry storm, 32.5 units). This confirms that composition amplifies entropy super-linearly; isolated phases do not predict composed behavior.

| Phase | Isolated entropy | Composed baseline | Amplification |
|---|---|---|---|
| retry_storm | 32.5 | 105.5 | 3.25× |
| spawn_race | 13.5 | 105.5 | 7.81× |
| partial_partition | 9.5 | 105.5 | 11.11× |
| delayed_rollback | 17.5 | 105.5 | 6.03× |

RNOS result (seed=42): refusal at step 8, entropy 50.0, containment classification "early" (refusal step ≤ 2 × phase1_steps and rnos_entropy < 0.5 × baseline_entropy). baseline_entropy = 105.5, threshold = 52.75, rnos_entropy = 50.0.

**Multi-seed evaluation (6 seeds)**: 100% early containment, average entropy savings 54.4%, standard deviation 4.8 units (CV ≈ 8.9%). Two distinct phase orderings observed (phases 2 and 3 swap); ordering did not affect containment classification. Refusal step was 8 in all 6 runs.

---

### 5.3 Adversarial Analysis (RNOS-2.0)

Four attacks were designed against specific named architectural properties of RNOS. Results are reported across 6 seeds; attacks are white-box (full knowledge of cost model and gate logic).

#### Attack 1: Budget Fragmentation

**Target**: per-context greedy evaluation.

Six fragment branches each spawn three micro-tasks, distributing 25 contexts with individually affordable costs (5–10 entropy units each). No single context looks dangerous. Budget depletes at the 24th context.

| Metric | Value |
|---|---|
| Contexts before refusal | 24 of 25 (96%) |
| Entropy savings | 8.8% |
| Containment classification | mid (all seeds, deterministic) |

The 8.8% entropy savings figure quantifies the cost of RNOS's per-context evaluation model when applied to a distributed-cost attack. It is the most effective entropy attack of the four. The step criterion for "early" fails: refusal at step 24 vs. threshold of 2 × phase1_steps = 10. The entropy criterion also fails: rnos_entropy = 176.5 vs. 0.5 × baseline = 96.75.

#### Attack 2: Delayed Amplification

**Target**: no look-ahead and no phase-level budget reservation.

Early phases are made trivially cheap (three hollow phases, phase1_steps = 4, early threshold = step 8). A recursive fanout tree (fanout 2–4, seed-dependent) is placed last.

| Seed fanout | Containment | Savings |
|---|---|---|
| 2 (seeds 1, 21) | none | 0.0% |
| 3 (seeds 7, 13, 99) | mid | 56.3% |
| 4 (seed 42) | mid | 80.2% |

In two seeds (fanout = 2), the entire cascade fits within the 200-unit budget. RNOS issues no refusal; containment_phase = "none"; entropy savings = 0.0%. The attacker sized the attack to fit the budget envelope. Average savings 41.5%, minimum 0.0%.

#### Attack 3: False Stability Trap

**Target**: correct non-intervention as attack surface.

A metronomic retry chain (5–7 levels deep, seed-dependent) runs first, consuming 72–88% of the budget through entirely legitimate, individually-correct contexts. When a spike phase (fanout of 4 children each attempting a network effect) arrives, RNOS has no clean refusal point; it has already consumed more than 0.5 × baseline_entropy on the stable phase, making "early" classification impossible.

Results: 100% mid containment across all 6 seeds, savings 12.3–34.7%. RNOS made correct decisions throughout; the containment classification failed because correct decisions during the stable phase consumed the budget that early containment required. The adversary's mechanism is not defeating RNOS's detection logic — it is exhausting headroom before the detectable signal arrives.

#### Attack 4: Refund Illusion

**Target**: budget recycling via rollback.

A primer phase runs, then five decoy tasks each stage an expensive network effect (4.0 units) that is immediately blocked by sandbox policy and rolled back. Each rollback releases the network reservation, allowing the gate to re-evaluate the next decoy against the restored apparent balance. Five decoys run before the sixth is refused.

| Metric | Value |
|---|---|
| Refusal step | 8 (vs. early threshold = step 4) |
| Entropy savings | 44.7–60.5% (seed-dependent) |
| Containment classification | mid (all seeds) |

Savings remain high because the effects system provided independent containment: network effects were never committed, regardless of how many decoys ran. The budget gate alone did not achieve early containment; the effects gate compensated. This is the one attack where the two enforcement layers diverged: budget gate failed early; effects gate succeeded independently.

---

### 5.4 Cross-Repository Architectural Finding

RNOS-Runtime Experiment 4 and RNOS-2.0 Budget Fragmentation are independent confirmations of the same architectural vulnerability:

> **RNOS's greedy per-unit evaluation cannot detect instability that is distributed across many individually-benign units.**

In Experiment 4 (per-step evaluation): `smoldering_instability` maintains a 30–40% failure rate with no consecutive run exceeding 2. Each step looks acceptable; the accumulated pattern does not. RNOS misses the scenario entirely; peak entropy 8.805, gap 0.195.

In Budget Fragmentation (per-context evaluation): 25 contexts each look individually affordable. The aggregate is catastrophic. RNOS runs 96% of the cascade before refusing; entropy savings 8.8%.

The failure mode is structurally identical: a local affordability check that passes individual units but lacks a mechanism for aggregate cost assessment. The experiments were designed independently in different codebases using different architectures (step-level entropy composition vs. context-level budget gate). The convergence on the same failure mode strengthens the finding.

The circuit breaker's sliding-window approach closes this gap partially for the step-level case (Experiment 4) by accumulating failure density without requiring consecutive failures. No equivalent mechanism exists in the budget gate for the context-level case.

---

## 6. Discussion

### Refusal as a Compositional Primitive

The composite cascade result shows that composed failure modes produce entropy that cannot be predicted from isolated components: 105.5 composed vs. 32.5 maximum isolated (3.25× amplification). This super-linearity is a direct consequence of cross-phase budget depletion — each phase inherits a degraded resource state, not a fresh one. A containment approach that evaluated each phase independently would underestimate the cascaded cost by at least 3×.

The RNOS budget system provides containment under composition precisely because the budget is shared: phase 1 depletion is visible to the gate when phase 2 requests admission. This is the property that makes refusal useful as a compositional primitive rather than a per-call guard.

### Complementary Detection Profiles

The adaptive CB outperforms RNOS on Experiment 4 and matches it on Experiments 2, 2.5, and 3 (with a 7-step timing advantage to RNOS on Experiment 3). The results do not support a claim that one approach is superior. They support a precise characterization:

- RNOS's structural floor (cumulative state preserved across recovery windows) is an advantage when failure is structured and bursty.
- CB's recency sensitivity (sliding window discards old history) is an advantage when failure is diffuse and non-consecutive.

These are architectural properties, not implementation details. Combining both in a single runtime would likely capture what each misses independently. The current RNOS implementation has no equivalent of the CB's failure density accumulation, and the CB has no equivalent of RNOS's cross-run structural floor.

### The Defense-in-Depth Observation

Attack 4 (Refund Illusion) demonstrated that the budget gate and effects gate operate as independent enforcement layers with different detection properties. The budget gate failed to achieve early containment; the effects gate blocked the underlying damage. In a system where the effects being staged are policy-allowed (rather than blocked by sandbox), the effects gate would not have compensated. The independence of the two layers provides robustness when they have complementary blind spots — but only when the effects in question are subject to the relevant policy restriction.

### Open Comparison

The adaptive circuit breaker has not been tested against the RNOS-2.0 composite cascade. The existing Experiment 4 result suggests the CB would perform well on smoldering distributed failure; it is an open question whether the CB also achieves early containment under the cross-phase budget depletion that the composite cascade tests. If the CB achieves early containment on the composite cascade, it would challenge the claim that entropy-based composition tracking provides a unique advantage. This comparison should be run before any strong claim about RNOS's superiority in multi-phase scenarios is made.

---

## 7. Limitations

**Synthetic deterministic workloads only.** All scenarios use fixed failure schedules without stochastic variation. Real workloads have non-deterministic failure timing, variable latency distributions, and recovery patterns that may not match the experimental scenarios. Whether the detection profiles observed here generalize to real execution traces is untested.

**Entropy weights are hand-tuned.** Component coefficients, caps, and thresholds were set by design judgment, not systematic optimization. The empirical calibration study (+7.1% aggregate containment improvement) was conducted over three adversarial scenarios using a univariate sweep and joint grid search; this is not a comprehensive optimization. Different weight assignments would produce different detection boundaries.

**White-box adversarial attacks.** Full knowledge of RNOS internals — cost model constants, gate evaluation logic, budget accounting mechanics — was used to design each attack. A gray-box or black-box attacker would face additional uncertainty. The measured attack effectiveness quantifies what is achievable with deliberate optimization, not what typical workloads produce.

**Four attacks are not exhaustive.** The adversarial experiments cover four properties identified through inspection of the architecture. Other properties — sandbox crossing surcharges, depth-dependent cost scaling, descendant count limits, the empirical cost model — may be exploitable through attack compositions not evaluated here.

**Circuit breaker is a single baseline.** The adaptive CB (sliding window, exponential backoff, adaptive threshold) is one instantiation of the circuit breaker pattern. Multi-signal breakers that combine failure density with latency, error type classification, or saturation metrics might perform differently. The result that CB outperforms RNOS on Experiment 4 is specific to this CB configuration.

**Persistence signals identified but not implemented.** Experiment 4 identifies five signals that cleanly discriminate `smoldering_instability` from `noisy_recovery`. None are currently part of the RNOS entropy computation. The discrimination gap is architecturally quantified but not closed.

**The 0.195 entropy gap and 96% pre-refusal execution are structural boundaries.** The 0.195 miss gap (Experiment 4) cannot be closed by threshold adjustment without introducing false positives. The 96% cascade execution (Budget Fragmentation) cannot be reduced by threshold adjustment alone — it requires aggregate cost projection across sibling contexts, which is not part of the current gate evaluation. Both represent structural changes, not parameter tuning.

**The adaptive CB has not been tested on the composite cascade.** The cross-phase composition result (Section 5.2.2) has not been replicated with a CB baseline. This is the largest gap in the current experimental coverage.

---

## 8. Conclusion

Refusal as a first-class execution primitive, evaluated before each action against a shared entropy budget, demonstrably contains cascade growth: at fanout-16, RNOS limits cascade size to 65 contexts against 69,905 unprotected. Under composed failure modes, the shared budget provides cross-phase containment that isolated per-phase evaluation cannot replicate, with early containment achieved consistently across six seeds.

The approach has specific, quantified failure modes: diffuse non-consecutive failure (0.195 entropy gap, structural ceiling), and distributed-cost attacks (96% cascade execution before refusal, greedy per-context evaluation). These are not threshold calibration issues. They identify the structural changes — aggregate cost tracking across siblings, phase-level budget reservation, persistence signal integration — that would be required to close the identified gaps.

The adversarial experiments produce a map of known failure boundaries with measured impact values, not a pass/fail verdict. The goal is to characterize where entropy-gated refusal applies and where it does not — and to provide a concrete architecture for researchers who want to explore the approach, extend it, or determine that a different design is more appropriate for their setting.

---

## Appendix: Experiment Reference

| Experiment | Source | Key metric | Result |
|---|---|---|---|
| Selective Containment | RNOS-Runtime, Exp 2 | Selectivity | RNOS 3/3, CB 3/3, Baseline 2/3 |
| Matched-Entropy | RNOS-Runtime, Exp 2.5 | Step of discrimination | Step 8 (entropy gap 4.685) |
| Intermittent Cascade | RNOS-Runtime, Exp 3 | Detection timing gap | RNOS −7 steps vs CB |
| Distributed Instability | RNOS-Runtime, Exp 4 | Miss gap | 0.195 entropy units (structural) |
| Cascade Growth | RNOS-2.0, core battery | Containment at fanout-16 | 1,075.5× |
| Composite Cascade | RNOS-2.0, experiments | Composition amplification | 3.25×–11.11× vs isolated |
| Multi-seed eval | RNOS-2.0, multi_seed | Early containment rate | 6/6 (100%) |
| Budget Fragmentation | RNOS-2.0, adversarial | Pre-refusal execution | 96% cascade before refusal |
| Delayed Amplification | RNOS-2.0, adversarial | Complete bypass | 2/6 seeds (fanout=2), 0% savings |
| False Stability | RNOS-2.0, adversarial | Min savings | 12.3% (5/6 seeds below 30%) |
| Refund Illusion | RNOS-2.0, adversarial | Containment delay | Budget gate bypassed; effects gate saved |
