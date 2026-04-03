# D3: Deterministic Degradable Distributed
## A Compute Architecture for Execution Under Uncertainty

**Rowan Ashford**

*Working draft v0.2 — April 2026*

---

## Abstract

Modern execution systems fail by continuing: they permit operation under degraded conditions, compounding instability through retries, recursive expansion, and effect commitment until the system enters a state from which recovery is no longer possible. D3 (Deterministic Degradable Distributed) addresses this structurally by treating correctness as a consumable resource. Each operation is evaluated against an entropy budget before execution; trust — a composite of confidence, integrity, and environmental stability — gates continuation. When admissibility fails, the system issues a structured refusal, an integrity-preserving terminal state that halts further execution rather than permitting silent degradation. We present the D3 formal kernel specification, an instruction set that exposes entropy and trust as architecturally visible state, and a five-layer execution stack that connects mission-level policy to fault containment. Two reference implementations — RNOS-2.0 and RNOS-Runtime — instantiate key aspects of the architecture in software and provide empirical validation across containment, discrimination, and adversarial regimes. At fanout-16, RNOS limits cascade growth to 65 contexts against 69,905 unprotected (1,075×); under composed failure, early containment is achieved consistently across six seeds with average entropy savings of 54%. Adversarial evaluation reveals that greedy per-unit admissibility cannot detect instability distributed across individually-benign operations, a boundary confirmed independently in two experimental regimes. Entropy weights are hand-tuned; all experiments use synthetic deterministic workloads; the ISA and hardware layer are not yet implemented.

---

## 1. Introduction

Modern execution systems fail in a consistent and under-addressed way: they continue operating after the justification for continuation has degraded. Availability is treated as sufficient for execution, while correctness is evaluated only after the fact. As a result, systems often fail not by producing incorrect results immediately, but by compounding instability through continued execution—amplifying retries, expanding call graphs, and committing effects under increasingly degraded conditions.

Existing control mechanisms—timeouts, retry limits, and circuit breakers—are reactive. They detect failure after it has manifested in observable signals such as error rates or latency thresholds. These mechanisms answer the question "has this call failed?" but not the more fundamental question: "should execution continue at all?" In scenarios where instability accumulates without crossing local thresholds—such as distributed low-grade failure or cost fragmentation—these approaches permit execution to proceed well beyond the point of recoverability.

This paper introduces D3 (Deterministic Degradable Distributed), a compute architecture that treats execution eligibility as a resource-governed property. In D3, continuation is not assumed; it is granted. Each operation is evaluated against a bounded model of execution uncertainty—formalized as entropy—and is permitted to proceed only if sufficient budget remains. When admissibility fails, the system issues a structured refusal: a terminal state that preserves integrity by halting further execution.

D3 is not a runtime optimization or a single system implementation. It is an architectural model spanning a formal kernel specification, an instruction set exposing entropy and trust as first-class state, and a scheduling framework that conditions execution on system integrity. This paper presents two reference implementations—RNOS-2.0 and RNOS-Runtime—that instantiate key aspects of the architecture in software, enabling empirical evaluation of its behavior under controlled conditions.

The experimental results demonstrate three primary properties. First, D3 enforces bounded execution under composition, limiting cascade growth by constraining execution at the point of admissibility. Second, it achieves selective containment, distinguishing between recoverable and non-recoverable execution trajectories without premature intervention. Third, adversarial evaluation reveals clear architectural boundaries: specifically, that greedy per-unit admissibility cannot detect instability distributed across individually-benign units. These boundaries are quantified and shown to arise from the structure of the admissibility formulation rather than implementation error.

The contribution of this paper is not to claim a complete solution to execution under uncertainty, but to define a coherent architectural approach, validate it empirically, and identify its limits precisely. D3 establishes a foundation for treating correctness as a schedulable resource and refusal as a first-class primitive in execution systems.

The remainder of the paper is structured as follows. Section 2 presents the architectural model, including entropy budgeting, admissibility, and mode control. Section 3 describes the two reference implementations. Section 4 reports experimental results across containment, discrimination, and adversarial regimes. Section 5 discusses implications and architectural boundaries. Section 6 enumerates limitations. Section 7 surveys related work, and Section 8 concludes.

---

## 2. Architecture

D3 is defined by a formal kernel specification. This section presents the core formalism and key architectural components. The full specification — 19 sections covering entropy, trust, admissibility, mode control, execution classes, redundancy semantics, validation, fault propagation, and safety guarantees — is maintained as a separate document and referenced where appropriate.

### 2.1 Core Principles

The D3 kernel is founded on four principles that distinguish it from availability-based execution models.

**Trust-governed execution.** A task is not admitted because resources are available; it is admitted because the current system state satisfies the task's trust requirements. Formally, task $\tau$ is admissible iff $T(\tau, s) \geq T_{\text{req}}(\tau)$, where $T(\tau, s)$ is the trust of $\tau$ under system state $s$ and $T_{\text{req}}(\tau)$ is the task's minimum trust threshold.

**Observable correctness.** Every output must carry integrity metadata. The output contract requires that all emitted results include confidence $C$, integrity $I$, entropy $E$, and mode context $M$. Silent production of trusted-wrong outputs violates a core kernel invariant (Section 2.6).

**Refusal as an integrity-preserving terminal state.** Refusal is not an error condition. It is the correct outcome when admissibility fails. A system that refuses is behaving correctly; a system that continues past admissibility failure is not.

**Graceful degradation via mode control.** Between full admission and refusal lies a graduated response. As entropy increases, the system reduces throughput, increases validation overhead, and tightens admission — responding proportionally rather than failing abruptly.

### 2.2 System State and Trust

The D3 kernel is formally defined as:

$$\mathcal{K} = (\mathcal{S}, \mathcal{T}, \mathcal{P}, \mathcal{O}, \mathcal{F})$$

where $\mathcal{S}$ is the system state space, $\mathcal{T}$ is the task space, $\mathcal{P}$ is the execution planning function, $\mathcal{O}$ is the output space, and $\mathcal{F}$ is the feedback function. The system state at any time is:

$$S = \{E, M, R, T_{\text{rad}}, V, D\}$$

where $E$ is entropy, $M$ is mode, $R$ is radiation level, $T_{\text{rad}}$ is thermal state, $V$ is voltage stability, and $D$ is divergence. For execution to produce an output, the kernel computes $O = \mathcal{K}(\tau, S)$ subject to trust admissibility.

Trust is defined as:

$$T = C \cdot I \cdot (1 - E)$$

where $C \in [0,1]$ is confidence (model or system certainty estimate), $I \in [0,1]$ is integrity ($I = 1 - \text{divergence\_rate}$), and $E \in [0,1]$ is entropy (normalized instability measure). Three axioms follow directly:

$$\frac{\partial T}{\partial E} \leq 0, \quad \frac{\partial T}{\partial I} \geq 0, \quad \frac{\partial T}{\partial C} \geq 0$$

Two limit theorems bound the trust function's behavior: as $E \to 1$, $T \to 0$ (no execution is trustworthy under maximum instability); if $I = 0$, then $T = 0$ regardless of confidence or entropy (integrity loss is absolute). Trust is not recovered within a single execution trace — once degraded, it constrains all subsequent admissibility decisions for the duration of the run.

### 2.3 Admissibility

Execution is admitted at step $t$ when both trust and budget conditions are satisfied:

$$A_t = \begin{cases} 1 & \text{if } T_t \geq \tau_T \text{ and } H_t \geq \rho(o_{t+1}) \\ 0 & \text{otherwise} \end{cases}$$

The entropy condition is a **budget sufficiency check**: the system requires enough remaining entropy budget $H_t$ to cover the estimated cost $\rho(o_{t+1})$ of the next operation. This is operation-dependent — a cheap operation can proceed at lower remaining budget than an expensive one. The execution rule is:

$$\text{continue}(t) \iff A_t = 1 \qquad \text{refuse}(t) \iff A_t = 0$$

The admissibility gate is evaluated **before** each operation, not after. This is the structural property that distinguishes D3 from reactive approaches: the gate fires on the next action, not on the current action's failure.

The current formulation is per-operation and greedy: each admission decision considers only the cost of the immediate next operation against the remaining budget. It does not project aggregate cost across planned siblings, reserve budget for future phases, or reason about the cost trajectory of the current execution graph. This greedy property has architectural consequences explored in Section 4.5.

### 2.4 Entropy Composition

In the RNOS implementations, the entropy budget is consumed by a charge function $\rho : \mathcal{O} \to \mathbb{R}^+$ mapping operation classes to costs:

$$H_t = H_{t-1} - \rho(o_t)$$

Seven operation classes are defined:

| Operation Class | Description | Symbol |
|---|---|---|
| `task_start` | Initiating a new execution unit | $\rho_s$ |
| `child_spawn` | Creating a child context | $\rho_c$ |
| `depth_level` | Each additional nesting level | $\rho_d$ |
| `sibling` | Each additional sibling context | $\rho_b$ |
| `local_write` | Local state mutation | $\rho_w$ |
| `network_call` | External dependency invocation | $\rho_n$ |
| `irreversible_action` | Non-reversible external effect | $\rho_i$ |

The charge function satisfies a **monotonicity contract**: operations with strictly greater structural complexity — greater branching, depth, or irreversibility — must not cost less than simpler operations:

$$\rho(o_a) \geq \rho(o_b) \quad \text{whenever } o_a \text{ induces greater structural risk than } o_b$$

This ensures the budget depletes faster under conditions of increasing structural risk.

**Empirical calibration.** Initial charge values were set heuristically. A systematic calibration swept each parameter independently from 0.5× to 2.0× its heuristic value across the adversarial benchmark battery, measuring containment quality against premature refusal rate. Of seven parameters, three are load-bearing: $\rho_s$ (task start), $\rho_c$ (child spawn), and $\rho_d$ (depth level). These govern budget depletion under multi-context expansion — the dominant mechanism in retry storms and fanout cascades. The remaining four showed low sensitivity; their heuristic values are confirmed adequate. Optimal values for the three load-bearing parameters are 2× their heuristic settings, producing +7.1% aggregate improvement in containment quality (+10.5% retry storm, +2.2% fanout explosion, +9.5% cascading failure) with no increase in premature refusal across 30 deterministic trials.

The finding that heuristic charges preserved correct relative ordering but systematically under-charged expansion operations is consistent with the observation that D3 failure modes are dominated by multiplicative growth rather than additive accumulation.

In the RNOS-Runtime (discrimination experiments), entropy is expressed as an accumulated composite score rather than a depleting budget: six weighted signals are summed at each step, and the result is compared against mode thresholds. The two representations are dual: a growing score toward a ceiling is equivalent to a depleting budget toward zero. The structural floor discussed in Section 4.3 (cost_score + repeated_tool = 4.0, non-resetting) is the RNOS-Runtime instantiation of accumulated execution cost that persists independent of recent failure patterns.

### 2.5 Mode Control

The D3 kernel maps entropy levels to execution control surfaces via a mode function $M = g(E)$. Four modes are defined, each specifying a tuple $\mathcal{C}(M) = (\kappa, N, V_l, A)$ of clock scale, redundancy level, validation level, and admission policy:

| Mode | Entropy Range | Clock $\kappa$ | Redundancy $N$ | Validation | Admission |
|---|---|---|---|---|---|
| Green | $E \in [0, 0.25)$ | 1.0× | 1× | None | Open |
| Yellow | $E \in [0.25, 0.5)$ | 0.8× | 2× | Selective | Filtered |
| Orange | $E \in [0.5, 0.75)$ | 0.6× | 2–3× | Dual | Restricted |
| Red | $E \in [0.75, 1]$ | 0.4× | 3× | Triple | Critical-only |

Mode transitions are monotonic: as entropy increases, clock scale decreases, redundancy increases, validation strengthens, and admission tightens. The system slows before it fails; it validates more before it commits; it refuses more before it collapses. Critical tasks (Class A) remain admissible deeper into Red mode than non-critical tasks, which are deferred or refused as the system enters constrained operation.

The RNOS reference implementations currently operate with a three-level approximation: Green maps to ALLOW, Yellow+Orange collapse into DEGRADE, and Red maps to REFUSE. The four-level distinction is the target state; the software approximation validates the mode-transition logic while deferring clock scaling and hardware-level redundancy to physical implementation.

### 2.6 Safety Guarantees

The kernel specification defines three formal safety guarantees, conditional on correct operation of the kernel control loop.

**No Silent Corruption** (Section 8.1). A system must not emit incorrect outputs without signaling reduced integrity or increased entropy:

$$y \neq y_{\text{true}} \Rightarrow I < 1 \;\lor\; E > 0$$

With redundancy $N \geq 2$ and validation enabled, $P(\text{silent corruption}) \to 0$. Errors may occur; they cannot be hidden.

**Bounded Degradation** (Section 8.2). System performance degrades gracefully under increasing entropy: throughput is a decreasing function of $E$, validation overhead is an increasing function of $E$. The system slows instead of breaking. A minimum service guarantee ensures critical tasks remain executable through the degradation trajectory.

**Fault Containment** (Section 8.3). Fault propagation must not be unbounded. A fault is contained if:

$$\Delta E_{\text{propagated}} \leq \Delta E_{\text{local}}$$

For a task sequence $\tau_1 \to \tau_2 \to \cdots \to \tau_n$, containment requires that $A(\tau_i, S) = 0$ whenever $T(\tau_i, S) < T_{\text{req}}$, truncating the propagation chain at the point of trust failure.

### 2.7 ISA Primitives

The D3 ISA exposes entropy and trust as architecturally visible state rather than hidden runtime metadata. Programs and policies can read, bind, and condition execution on trust values directly. Four instruction classes are defined:

**Entropy control**: `ECHK` reads the current entropy level; `EMODE` enforces a maximum permitted execution mode for the current context; `EBIND` binds an entropy budget to a task, establishing the depletion envelope for its lifetime.

**Validation**: `VSET` configures the validation policy (none/selective/dual/triple); `VCMP` compares outputs from redundant execution paths; `VCHK` checks divergence against a threshold; `ITAG` attaches an integrity requirement to a task, encoding the minimum $I$ value required for trusted output.

**Trust output**: `QCONF` summarizes confidence from the execution result; `QINT` summarizes integrity from divergence and fault signals; `QEMIT` serializes the trust-qualified output — the full tuple (prediction, confidence, integrity, entropy, mode, trust_status) — to the external interface.

**Scheduling**: `TPRIO` assigns task class priority; `TGATE` enforces the admission gate before execution begins; `TFALLBK` specifies the fallback execution path invoked when trust conditions fail.

The significance of this ISA design is that trust is a first-class program-visible resource. A downstream consumer receives not just a prediction but a signed account of the conditions under which the prediction was produced. The consumer can act on this account — downgrading reliance, requesting recomputation, or refusing to act on flagged results.

### 2.8 Five-Layer Execution Stack

The D3 architecture is organized as a five-layer stack with a cross-cutting telemetry plane:

```
Mission / workload layer        — what the system is trying to accomplish
         ↓
Kernel policy layer             — admission, mode assignment, execution planning
         ↓
ISA / execution contract layer  — trust-aware instruction execution
         ↓
Compute + validation fabric     — tensor, scalar, redundant execution, quorum
         ↓
Fault containment / recovery    — isolation, restoration, scrubbing
         ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
Entropy / trust telemetry       — cross-cutting; continuously feeds all layers
```

The mission layer submits `TaskRequest` objects specifying task class (A/B/C), required integrity, required confidence, entropy bound, and fallback path. The kernel policy layer evaluates current system state, assigns mode, determines redundancy level and validation policy, and produces an `ExecutionPlan`. The ISA contract layer materializes this plan into instruction-visible primitives (e.g., `TPRIO CLASS_A; EMODE max=ORANGE; ITAG integrity>=0.99; RSET 3; VSET TRIPLE; TFALLBK SAFE_NAV`). The compute fabric executes within this policy envelope. Fault containment handles recovery paths when the fabric reports violations.

The key invariant preserved across all layers: no layer may emit an unflagged trusted output when the layer below it cannot support the required integrity. The trust contract propagates upward; fault information propagates upward; refusal propagates upward. Silent state repair is not permitted.

### 2.9 Entropy Field Model

The entropy field model extends the per-operation budget formulation to account for environmental context. Total system entropy is decomposed as:

$$E_{\text{total}} = E_{\text{task}} + E_{\text{system}} + E_{\text{environment}}$$

where $E_{\text{task}}$ captures input ambiguity and model uncertainty, $E_{\text{system}}$ captures queue depth, retry rates, and resource contention, and $E_{\text{environment}}$ captures radiation, network instability, and hardware degradation. Effective entropy — the quantity used in admission decisions — amplifies task uncertainty by system and environmental conditions:

$$E_{\text{effective}} = E_{\text{task}} \times (1 + E_{\text{system}} + E_{\text{environment}})$$

Entropy pressure, defined as $P_{\text{entropy}} = dE_{\text{total}}/dt$, determines how aggressively the admission gate should tighten. The cascade condition — when the generation rate exceeds the containment rate:

$$\frac{dE_{\text{system}}}{dt} > \text{containment\_rate}$$

— identifies the regime where proactive refusal becomes necessary before local per-operation signals become visible. This theoretical condition is empirically explored in Section 4.4.

---

## 3. Reference Implementations

### 3.1 RNOS-2.0

RNOS-2.0 is the core runtime implementation. It realizes the D3 admissibility gate, entropy budget system, effects staging, and causal trace in software.

The central object is `EntropyBudget`, which maintains three values: `total` (initial allocation), `remaining` (unconsumed budget), and `reserved_for_commit` (held for staged effects not yet committed). The admission test is:

$$\text{can\_afford}(\text{amount}) \equiv (\text{remaining} - \text{reserved\_for\_commit}) \geq \text{amount}$$

The two-level accounting separates task execution cost from effect commit cost, enabling the effects gate to operate as an independent enforcement layer from the task admission gate. `ExecutionGate` performs four checks at each spawn: depth limit, fanout limit, total-descendants limit, and budget sufficiency. All four fire before execution begins.

The `CostModel` computes task cost from structural properties: $\rho_s = 1.0$ base, plus $\rho_d \times \text{depth}$ and $\rho_c$ for child contexts, plus $(\text{sibling}-1) \times \rho_b$ for sibling ordering. Effect costs are charged separately: $\rho_n = 4.0$ for network calls, $\rho_w = 2.0$ for writes, with $\rho_i = 6.0$ irreversibility surcharge for non-reversible actions.

The budget is shared across all contexts in a run. When a parent spawns a child, both draw from the same `EntropyBudget` object — there is no per-context isolation. Depletion by one context reduces what is available for all subsequent contexts, providing the cross-phase budget depletion that the composite cascade experiment tests.

RNOS-2.0 is available at github.com/forumk2/RNOS-2.0 under Apache 2.0.

### 3.2 RNOS-Runtime

RNOS-Runtime implements the discrimination experiment suite. It uses a configurable failure source with deterministic schedules, an adaptive circuit breaker (CB) as a non-strawman baseline, and a selectivity scoring methodology that measures correct discrimination between recoverable and non-recoverable scenarios.

In RNOS-Runtime, entropy is computed as a weighted composite of six signals evaluated at each step: consecutive failures (retry_score), recent failure rate over 5 steps (failure_score), cumulative execution cost (cost_score, non-resetting), tool repetition (repeated_tool), planner inference latency (latency_score), and execution depth (depth_score). The admission decision compares the accumulated score against mode thresholds (DEGRADE at 9.0, REFUSE at 11.0).

The adaptive CB uses a sliding-window failure-rate measurement with exponential backoff and adaptive threshold adjustment. It is configured to represent a serious, tuned baseline — not a strawman. The experimental results in Section 4.3 characterize where each approach has a structural advantage.

RNOS-Runtime is available at github.com/forumk2/RNOS-Runtime under Apache 2.0.

### 3.3 Relationship to D3 Architecture

RNOS-2.0 and RNOS-Runtime implement a software subset of the D3 kernel specification. Specifically: entropy budgeting (Section 2.4), admissibility (Section 2.3), trust degradation (RNOS-Runtime), mode-based graduated response (Section 2.5), and effects containment (RNOS-2.0 effects gate). Unimplemented kernel features — full redundancy semantics, validation escalation, the hardware entropy field integration, sandbox-crossing surcharges beyond basic flag — represent extensions for future work rather than current capabilities.

The ISA primitives (Section 2.7) have no hardware implementation. RNOS validates the semantic model: EBIND corresponds to `EntropyBudget` initialization; TGATE corresponds to `ExecutionGate.evaluate()`; QEMIT corresponds to the structured `RunResult` output. The ISA articulates what these mechanisms should look like in an instruction-set-visible form; the software demonstrates that the semantic model produces correct behavior.

The Terafab D3 hardware program (Section 5.5) provides physical-layer survival mechanisms — radiation hardening, TMR on critical paths, ECC/EDAC on all memory structures, and continuous scrubbing. D3/RNOS provides the semantic execution layer: governing whether and how computation proceeds once physical errors have been detected and corrected. The two layers are complementary; neither substitutes for the other. Radiation hardening prevents bit flips; RNOS prevents continued execution under conditions where bit flips, retry storms, and cascading effects have compromised the integrity of the execution context.

---

## 4. Experimental Results

### 4.1 Core Battery v1

The Core Battery evaluates D3 against three adversarial scenarios under both baseline (no containment) and D3 protocols.

**Aggregate results across all scenarios:**

$$\overline{\Delta_C} \approx 94.999\%, \quad \overline{\Delta_E} \approx 95.080\%, \quad \overline{\Delta_H} \approx 47.013\%$$

D3 reduced context creation by ~95%, committed effects by ~95%, and entropy accumulation by ~47%.

| Experiment | Baseline contexts | D3 contexts | $\Delta_C$ | Termination |
|---|---|---|---|---|
| Retry Storm | 265,719 | 39 | 99.985% | trust_below_threshold |
| Fanout Explosion | 12,207,031 | 488,281 | 96.000% | entropy_insufficient |
| Cascading Failure | 364 | 40 | 89.011% | trust_degradation |

The Retry Storm result demonstrates early containment: D3 terminates via trust collapse before retry amplification reaches significant scale. The Fanout Explosion demonstrates late containment: D3 allows initial expansion but terminates via entropy exhaustion after growth begins. This asymmetry reflects the dual-mode containment taxonomy described in Section 5.1. The Cascading Failure result shows trust-gated propagation: degraded upstream state raises $E$ and lowers $I$, reducing $T$ below threshold before the cascade reaches full depth.

**Cascade Growth Baseline.** A pure fanout sweep (fanout 2, 4, 8, 16) isolates the containment factor as a function of cascade depth:

| Fanout | Unprotected contexts | Protected contexts | Containment factor |
|---|---|---|---|
| 2 | 31 | 9 | 3.4× |
| 4 | 341 | 17 | 20.1× |
| 8 | 4,681 | 33 | 141.8× |
| 16 | 69,905 | 65 | 1,075.5× |

The containment factor grows super-linearly with fanout because the budget depletes at a rate proportional to the cascade depth, truncating an exponentially-growing tree near its base. Unprotected entropy at fanout-16: 1,216,337 units. Protected: 22.5 units. The containment is not perfect suppression but bounded truncation — 65 contexts run before the gate refuses; those 65 represent the useful exploration before the budget is exhausted.

### 4.2 Trajectory Analysis and State Preservation

Execution trajectory data from a mixed chaos scenario shows D3 terminating at $t_{D3} = 9$ with $C_{t_{D3}} = 0.4$, while the baseline reaches collapse at $t_{\text{collapse}} = 14$ but continues execution until $t = 20$, with $C_{\text{exit}} = 0.0$.

The collapse boundary is defined as $t_{\text{collapse}} = \min\{t \mid C_t = 0\}$. The baseline continues for 6 steps in a zero-confidence regime after reaching this boundary. D3 terminates 5 steps before collapse with confidence intact:

$$t_{D3} < t_{\text{collapse}}, \quad C_{t_{D3}} = 0.4 > 0$$

State preservation is defined as $C_{t_{\text{exit}}} > 0$. The baseline exits in collapsed state; D3 exits in preserved state. This is not just a containment result — it establishes that D3 is a **state-preserving execution strategy**, terminating in a regime where meaningful recovery remains possible.

### 4.3 Discrimination Properties (RNOS-Runtime)

Four experiments test selective containment: the ability to distinguish recoverable from non-recoverable execution trajectories under matched surface conditions.

| Experiment | Scenario Type | RNOS | CB | Baseline | Key finding |
|---|---|---|---|---|---|
| 2 — Selective Containment | Cascade vs. recoverable instability | 3/3 | 3/3 | 2/3 | Both match; baseline cannot discriminate |
| 2.5 — Matched-Entropy | Identical-state divergence | 4/4 | 4/4 | 2/4 | Correct non-intervention when evidence absent |
| 3 — Intermittent Cascade | Bursty failure with deceptive recovery | 4/4 | 4/4 | 2/4 | RNOS detects 7 steps earlier |
| 4 — Distributed Instability | Diffuse, non-consecutive failure | **3/4** | **4/4** | 2/4 | CB detects what RNOS misses |

**Experiment 2** establishes the minimum criterion: selectivity over blunt intervention. The phase transition (failure lengths 1–3: ALLOW; 4: DEGRADE; 5+: REFUSE) is clean and two-step. RNOS refuses the runaway cascade at step 7 (3 wasted steps); the baseline exhausts all 20 steps (16 wasted).

**Experiment 2.5** confirms that D3 is reactive, not predictive. Two scenarios — `matched_recovery` and `matched_collapse` — are verified entropy-identical through step 6 (7.000, diff = 0.0) and step 7 (8.950). RNOS issues ALLOW for both at step 7. This is correct: the admissibility gate cannot act on evidence that does not yet exist. Discrimination occurs at step 8, one step after divergence: recovery entropy 6.125 (ALLOW), collapse entropy 10.810 (DEGRADE). The entropy gap opened by 4.685 in a single step; RNOS detected it immediately. At step 9, collapse reaches 11.225 and RNOS issues REFUSE. The mechanism is delayed magnitude discrimination — response to observable signal, not trajectory speculation.

**Experiment 3** is RNOS's strongest result. `bursty_recovery` and `intermittent_cascade` share the same surface burst-and-recovery pattern but differ in structural outcome. Both strategies reach correct final decisions (4/4). The difference is timing.

At step 11, after `intermittent_cascade`'s burst 2 third consecutive failure, RNOS entropy is 9.165 (retry 3.0 + cost 2.0 + repeated_tool 2.0 + failure 1.95 + latency 0.215 = **9.165 → DEGRADE**). The adaptive CB at the same step has window [S,S,F,F,F] = 3/5 = 0.60; the CB's strict `>` check means 0.60 does not exceed 0.60 — result: ALLOW. The CB issues its first intervention at step 18 (window reaches 0.80). The detection gap is 7 steps.

The mechanism is the structural floor: cost_score (2.0) + repeated_tool (2.0) = 4.0 is present before any failure-specific signal. The same 3-consecutive-failure burst in a fresh run produces entropy ~3.64, well below DEGRADE. At step 11 of an established run, the 4.0 floor ensures the same burst crosses the threshold. The CB's sliding window at step 13 shows [F,F,F,S,S] = 0.40 — burst 1 gone, burst 2 fading. RNOS at step 13 shows entropy 6.1, floor intact.

The `bursty_recovery` peak of 8.650 (0.35 below DEGRADE, no intervention) confirms no over-triggering. The margin is mechanically explained: one fewer failure in burst 2 shifts retry and failure scores by 1.65, keeping the total below the floor-elevated threshold.

**Experiment 4** defines RNOS's structural boundary. `smoldering_instability` maintains 30–40% failure rate with no consecutive run exceeding 2. RNOS peak entropy: 8.805; DEGRADE threshold: 9.0; miss gap: 0.195 units. Under ≤2 consecutive failures, the entropy ceiling is structurally bounded at ~8.8 (retry ≤ 2.0 + failure ≤ 2.6 + floor 4.0 + latency ~0.2). Lowering the threshold to 8.8 would cause false positives on `noisy_recovery`, which reaches 7.11 during its noisy phase.

The CB detects `smoldering_instability` at step 18 (window 4/5 = 0.80 > 0.60). The CB's failure density accumulation is independent of consecutiveness — the structural property RNOS's retry-based scoring cannot replicate. This is a detection profile difference, not a quality difference: RNOS and CB have complementary strengths across different failure regimes.

### 4.4 Compositional Stability

Four failure modes — retry storm, spawn race, partial partition, delayed rollback — were chained with a shared 50-unit budget (RNOS-2.0 composite cascade experiment). Each phase inherits the depleted budget from the prior phase; no phase receives a fresh allocation.

The uncontrolled baseline produced 105.5 entropy units — 3.25× the entropy of the hardest isolated component (retry storm, 32.5 units):

| Phase | Isolated entropy | Composed baseline | Amplification |
|---|---|---|---|
| retry_storm | 32.5 | 105.5 | 3.25× |
| spawn_race | 13.5 | 105.5 | 7.81× |
| partial_partition | 9.5 | 105.5 | 11.11× |
| delayed_rollback | 17.5 | 105.5 | 6.03× |

Isolated phase entropies do not predict composed behavior. Composition is super-linear: the joint cascade produces entropy that cannot be derived from any individual component. This validates the entropy field model's cascade condition — when entropy generation rate exceeds containment rate, individual-unit admissibility is insufficient.

RNOS containment: refusal at step 8, entropy 50.0, containment classification "early" (refusal step ≤ 2 × phase1_steps = 12, rnos_entropy 50.0 < 0.5 × baseline 52.75). Three of four phases entered before refusal; the fourth (delayed_rollback) was never reached.

Multi-seed evaluation (6 seeds, seeds 1, 7, 13, 21, 42, 99): 100% early containment, average entropy savings 54.4%, standard deviation 4.8 units (CV ≈ 8.9%). Two distinct phase orderings observed (phases 2 and 3 swap under seed control); ordering did not affect containment classification. Refusal step was 8 in all 6 runs.

### 4.5 Adversarial Analysis

Four attacks target specific named architectural properties of the D3 admissibility formulation. All are white-box attacks conducted across 6 seeds in RNOS-2.0. No attack successfully bypasses refusal entirely (except Delayed Amplification in 2/6 seeds); each delays or degrades the quality of containment in a measurable way.

**Budget Fragmentation** targets per-context greedy evaluation. Six fragment branches each spawn three micro-tasks, distributing 25 individually-cheap contexts (5–10 entropy each). No single context triggers alarm. Budget depletes at the 24th context.

- Contexts before refusal: 24 of 25 (96%)
- Entropy savings: 8.8% (deterministic, all seeds)
- Containment classification: mid (all seeds)

This is the most effective entropy attack. 96% of the cascade executes before RNOS refuses; 8.8% savings is the measured cost of greedy per-context evaluation against a distributed-cost workload.

**Delayed Amplification** targets the absence of look-ahead and phase-level reservation. Early phases are compressed to near-zero cost (phase1_steps = 4, early threshold = step 8). A recursive fanout tree (fanout 2–4, seed-dependent via jitter) is placed last.

- Fanout 2 (seeds 1, 21): complete bypass, 0.0% savings, containment_phase = none
- Fanout 3 (seeds 7, 13, 99): mid containment, 56.3% savings
- Fanout 4 (seed 42): mid containment, 80.2% savings

In two seeds, the entire cascade fits within the budget. RNOS issues no refusal; containment_phase = "none". The attacker sized the attack to fit the budget envelope; RNOS has no mechanism to anticipate this mismatch.

**Refund Illusion** targets budget recycling via rollback. A flood of decoy tasks stage expensive network effects (4.0 units each) immediately blocked by sandbox policy, releasing the reservation on rollback. RNOS keeps admitting decoys because each rollback restores apparent budget. Refusal arrives at step 8, against an early-containment threshold of step 4.

- Containment: mid (100% of seeds)
- Entropy savings: 44.7–60.5%
- Structural observation: the effects gate provided independent containment — network effects were never committed regardless of how many decoys ran. The budget gate failed to achieve early containment; the effects gate compensated.

**False Stability** targets correct non-intervention as attack surface. A metronomic stable chain runs first (depths 5–7, seed-dependent), consuming 72–88% of budget through individually-correct contexts before the spike phase begins.

- Containment: mid (100% of seeds)
- Entropy savings: 12.3–34.7%
- Structural observation: RNOS made correct decisions throughout. The containment failure occurred because correct decisions during the stable phase exhausted the budget that early containment required. The adversary's mechanism is headroom exhaustion, not detection evasion.

### 4.6 Cross-Implementation Finding

RNOS-Runtime Experiment 4 and RNOS-2.0 Budget Fragmentation independently confirm the same architectural boundary:

> **Greedy per-unit admissibility cannot detect instability distributed across individually-benign units.**

In Experiment 4 (per-step): `smoldering_instability` produces a 30–40% failure rate with no consecutive run exceeding 2. Each step looks acceptable in isolation; the accumulated pattern does not. RNOS misses the scenario entirely; peak entropy 8.805, structural ceiling ~8.8.

In Budget Fragmentation (per-context): 25 contexts each look individually affordable. The aggregate is catastrophic. RNOS runs 96% of the cascade before refusing; entropy savings 8.8%.

The failure mode is structurally identical in both cases: a local affordability check that passes individual units but lacks a mechanism for aggregate assessment. The two experimental regimes are architecturally independent — different evaluation granularity, different entropy representations, different implementation codebases — yet they converge on the same failure mode. This convergence identifies the gap as an architectural property of the current admissibility formulation (Section 2.3), not an implementation error in either codebase.

### 4.7 Summary Table

| Experiment | Source | RNOS result | CB result | Baseline result | Key finding |
|---|---|---|---|---|---|
| Retry Storm | RNOS-2.0, Core Battery | 99.985% ctx reduction | — | 0% | Early trust-triggered refusal |
| Fanout Explosion | RNOS-2.0, Core Battery | 96% ctx reduction | — | 0% | Late entropy-triggered refusal |
| Cascading Failure | RNOS-2.0, Core Battery | 89% ctx reduction | — | 0% | Trust degradation via propagation |
| Cascade fanout-16 | RNOS-2.0, bench | 1,075× containment | — | — | Budget-bounded truncation |
| Selective Containment | RNOS-Runtime, Exp 2 | 3/3 selectivity | 3/3 | 2/3 | Both match; baseline cannot discriminate |
| Matched-Entropy | RNOS-Runtime, Exp 2.5 | 4/4 selectivity | 4/4 | 2/4 | Correct non-intervention under ambiguity |
| Intermittent Cascade | RNOS-Runtime, Exp 3 | 4/4 (−7 steps vs CB) | 4/4 | 2/4 | Cross-burst memory advantage |
| Distributed Instability | RNOS-Runtime, Exp 4 | **3/4** | **4/4** | 2/4 | CB detects diffuse failure; RNOS cannot |
| Composite Cascade | RNOS-2.0, experiments | Early (6/6 seeds) | — | — | 54% avg savings, 3.25–11.11× amplification |
| Budget Fragmentation | RNOS-2.0, adversarial | 8.8% savings (mid) | — | — | 96% pre-refusal execution |
| Delayed Amplification | RNOS-2.0, adversarial | 0% savings (2/6 seeds) | — | — | Complete bypass at fanout=2 |
| False Stability | RNOS-2.0, adversarial | 12–35% savings (mid) | — | — | Correct behavior exploited as vulnerability |
| Refund Illusion | RNOS-2.0, adversarial | 44–61% savings (mid) | — | — | Budget gate bypassed; effects gate compensated |

---

## 5. Discussion

### 5.1 Refusal as a First-Class Primitive

Traditional systems treat refusal as an exception — a degraded path reached when the normal path fails. D3 inverts this: refusal is an expected, legitimate, integrity-preserving outcome that the system is designed to reach when conditions warrant. Execution integrity is:

$$\text{integrity} = \text{bounded continuation} + \text{justified refusal}$$

The dual-mode taxonomy observed empirically (Section 4.1) maps to two distinct kernel mechanisms. **Early refusal** (trust-triggered) occurs when accumulated divergence, upstream state degradation, or validation failure drives $T$ below threshold before entropy budget is exhausted — the system terminates before instability grows. **Late refusal** (entropy-triggered) occurs when execution has proceeded into deeper exploration and the budget depletes — the system terminates after exploration but before unbounded growth. Both are correct outcomes; they differ in which signal fires first.

State preservation (Section 4.2) adds a dimension to this: refusal is not just containment, it is exit with recoverable state. A system that terminates at $C = 0.4$ is in a different position from one that continues to $C = 0.0$. The former has resources remaining, context that can be examined, and a structured output explaining why execution stopped. The latter has nothing.

### 5.2 Complementary Detection Profiles

RNOS and the adaptive CB have different architectural foundations that produce complementary detection profiles. RNOS's structural floor — cumulative execution cost that persists across recovery windows — provides memory of past instability that the CB's sliding window discards. This gives RNOS a 7-step advantage on `intermittent_cascade`, where burst 1 has faded from the CB's window by the time burst 2 triggers. The CB's advantage on diffuse failure (`smoldering_instability`) arises from its failure density accumulation being independent of consecutiveness — a structural property RNOS's retry-based scoring cannot replicate.

Neither approach dominates. A system combining both — cumulative cross-run memory for structured cascades, sliding-window density for diffuse distributed failure — would likely cover the identified gaps. The current RNOS implementations contain no equivalent of the CB's density accumulation, and the CB contains no equivalent of the RNOS structural floor.

### 5.3 Architectural Boundaries

Three boundaries are empirically identified and measured:

**No aggregate cost tracking.** The fragmentation attack demonstrates this with 96% pre-refusal execution. The gate evaluates each context independently; a cascade that is safe at the per-context level can be catastrophic at the aggregate level. Addressing this requires aggregate cost projection across planned siblings before admitting the first — a look-ahead property the current formulation lacks.

**No phase-level budget reservation.** The delayed amplification attack demonstrates this with 0% savings in 2/6 seeds. With no mechanism to reserve budget for known future phases, a cheap early sequence consumes the headroom that a late expensive phase requires. In the extreme case, the entire cascade completes without refusal. Addressing this requires per-phase reservation, which the single shared-counter budget model does not provide.

**Rollback-aware accounting.** The refund illusion attack demonstrates that released effect reservations extend the admitted-context sequence beyond what committed-effect accounting would allow. The effects gate provided partial mitigation in this specific experiment (network effects were blocked by sandbox policy); that mitigation is not guaranteed when the effects in question are policy-allowed.

Each gap is quantified: 96% pre-refusal execution, 0% savings (complete bypass), containment delayed from step 4 to step 8. These are not threshold-tuning issues. They represent structural properties of the admissibility formulation that require architectural changes — aggregate cost projection, phase-level reservation, committed-effect accounting — to address.

### 5.4 Theory and Implementation

The kernel specification defines 19 sections of formal semantics. RNOS-2.0 and RNOS-Runtime implement a subset: entropy budgeting, admissibility, mode-based graduated response, effects containment, and trust degradation (RNOS-Runtime). Unimplemented kernel features — full redundancy semantics, validation escalation (VSET/VCMP/VCHK), entropy field integration, the complete ISA — represent architectural extensions rather than current capabilities.

The relationship between theory and implementation is two-directional. The kernel spec provides the formal grounding: trust function axioms, safety guarantee conditions, mode control surface definitions. The implementations provide empirical feedback: the adversarial analysis identifies which architectural properties are exploitable and quantifies the impact, pointing to specific kernel spec sections (per-operation admissibility, absence of inter-context budget reservation, rollback semantics) that require extension. The gaps identified in Section 4.6 correspond to architectural additions, not parameter tuning.

### 5.5 Relationship to Terafab D3 Hardware

The Terafab program — a radiation-hardened processor family targeting orbital AI compute — addresses the physical layer of execution reliability: ECC/EDAC on all memory structures, TMR on critical paths, continuous scrubbing, and junction temperature tolerance for vacuum thermal environments. D3/RNOS addresses the semantic layer: whether execution that has survived physical faults should continue given its current integrity and entropy state.

The two layers are complementary in a precise sense. Radiation hardening prevents physical bit corruption from propagating silently. RNOS prevents semantic corruption — retry amplification, cascading effects, budget fragmentation — from propagating silently. Neither substitutes for the other. A radiation-hardened processor running an uncontrolled retry storm still fails by continuing; an entropy-bounded runtime running on unhardened silicon still faces silent physical corruption. The complete system requires both.

The proposed ISA extensions bridge the layers: RADMON/RADCHK translate radiation error rates into entropy budget charges, enabling proactive mode transitions before semantic integrity is compromised; THERMCHK adjusts cost thresholds under thermal stress; EBUDGET provides explicit task-level entropy allocation for orbital workload scheduling. These extensions are proposed, not implemented.

---

## 6. Limitations

**Synthetic deterministic workloads.** All experiments use fixed failure schedules without stochastic variation. Results characterize behavior on the specific scenarios evaluated. Whether these profiles generalize to real workloads with non-deterministic failure timing, variable latency distributions, and concurrent interacting failure modes is untested.

**Entropy weights are hand-tuned.** Charge parameters and thresholds were set by design judgment and confirmed by a single calibration study over three adversarial scenarios. The calibration confirmed relative ordering and identified three load-bearing parameters, but the search space was limited to a univariate sweep followed by a joint grid search. Alternative weight assignments would produce different detection boundaries. The 0.195 entropy gap in Experiment 4 and the 8.8% savings in Budget Fragmentation are consequences of specific weight choices; they are not claimed to be worst-case bounds.

**White-box adversarial attacks.** Full knowledge of RNOS internals — cost model constants, gate evaluation logic, budget accounting mechanics, threshold values — was used to design each attack. A gray-box or black-box attacker faces additional uncertainty. The measured attack effectiveness quantifies what is achievable with deliberate optimization; it does not represent the expected impact of naturally-occurring workloads.

**Four attacks are not exhaustive.** The adversarial experiments cover four properties identified by inspection of the architecture. Sandbox-crossing surcharges, depth-dependent cost scaling, descendant count limits, and interactions among multiple attack patterns were not evaluated. Additional vulnerabilities may exist in compositions not examined here.

**Circuit breaker is a single baseline.** The adaptive CB with sliding-window failure rate, exponential backoff, and adaptive threshold is one instantiation of the circuit breaker pattern. Multi-signal breakers combining failure density, latency histograms, error type classification, and saturation metrics might perform differently. The finding that CB outperforms RNOS on Experiment 4 is specific to this configuration.

**Persistence signals identified but not implemented.** Experiment 4 identifies five signals that cleanly discriminate `smoldering_instability` from `noisy_recovery` (stability_score 9 vs 0, rolling_failure_rate_10 0.1 vs 0.4, avg_latency_last_5 80ms vs 282ms). These are observed and logged but not currently part of the entropy composition. The discrimination gap is architecturally quantified but not closed.

**The 0.195 entropy gap and 96% pre-refusal execution are structural.** These values cannot be improved by threshold adjustment alone. The 0.195 gap requires persistence signal integration or a reformulated failure density component. The 96% pre-refusal execution requires aggregate cost projection across siblings, which is not present in the current gate formulation. Both require architectural changes, not parameter tuning.

**RNOS implements a subset of the D3 kernel spec.** Unimplemented features — full redundancy semantics, validation escalation, entropy field model integration, ISA instruction hardware — are theoretical rather than empirically validated. Claims about these features are architectural claims, not measured results.

**The ISA has no hardware implementation.** The instruction set exists as a conceptual specification and a software-level approximation in the RNOS runtime. No silicon, no FPGA prototype, no microcode has been produced. ISA claims are semantic claims about what the instruction set should do; they are not performance claims.

**The Terafab connection is complementary positioning, not validated integration.** The D3/RNOS system and the Terafab hardware program are not formally integrated. The proposed ISA extensions (RADMON, THERMCHK, EBUDGET) are proposed; none have been implemented or tested against hardware.

**The adaptive CB has not been tested against the composite cascade.** The multi-phase composition result (Section 4.4) has no CB baseline. Whether the CB achieves early containment under cross-phase budget depletion is unknown and would directly test whether RNOS's shared budget provides an advantage that the CB's per-call evaluation cannot replicate.

---

## 7. Related Work

**Circuit breaker patterns.** The circuit breaker pattern, established by Netflix's Hystrix and widely implemented in resilience4j, Polly, and platform-level infrastructure (AWS, gRPC, Kubernetes), monitors failure rate within a sliding window and blocks calls when the rate exceeds a threshold. Circuit breakers are reactive and local: they respond to recent call failure, not to accumulated execution history. D3 differs in two structural ways: it evaluates admissibility before execution begins (not after calls fail), and it accumulates state across the full execution run rather than a recent window. Section 4.3 establishes that these differences produce complementary detection profiles across different failure regimes.

**Chaos engineering.** Netflix Chaos Monkey and subsequent chaos platforms (Gremlin, Chaos Mesh, LitmusChaos) identify system weaknesses by deliberate fault injection during normal operation. Chaos engineering is diagnostic: it tests what happens when faults occur. D3 is operational: it controls whether execution should continue given current state. Chaos engineering might identify the failure modes D3 is designed to contain; they address different phases of the engineering lifecycle.

**Entropy in computing.** The term entropy appears in several computing contexts. In information theory (Shannon, 1948), entropy measures uncertainty in a probability distribution. In distributed systems, entropy sometimes describes state divergence in eventually-consistent stores. RNOS-Runtime's entropy measure is closest to a weighted heuristic score; the D3 kernel spec formalizes entropy as a normalized instability measure bounding the reachable state-space (E_t(s) ≈ log|R_t(s)|). Neither is strictly thermodynamic; both treat entropy as a signal requiring active management rather than passive observation.

**Radiation-hardened computing.** Radiation-tolerant processors for space applications rely on TMR (triple modular redundancy), ECC/EDAC on memory paths, SEU detection and correction, and latch-up protection (see BAE Systems RAD750, Mobileye EyeQ6H radiation-tolerant variants, and emerging AI-focused orbital compute programs). These approaches address physical-layer fault tolerance; they do not address semantic-layer execution integrity. D3 proposes complementary coverage by governing execution decisions above the physical fault-correction layer.

**Structured concurrency.** Structured concurrency frameworks (Swift's TaskGroup, Kotlin coroutines, Java Loom) impose lifecycle discipline on concurrent tasks, ensuring child tasks complete before parent scope exits. This provides structural containment analogous to D3's depth and fanout limits. D3 extends this with budget-governed admission and trust-gated continuation; structured concurrency provides lifecycle management but not admissibility control based on accumulated instability.

**Resource governors and query cost estimation.** Database query planners estimate execution cost before committing to a query plan (PostgreSQL cost model, SQL Server Query Optimizer). Resource governors (SQL Server, Oracle) throttle workload groups against CPU, memory, and I/O quotas. These approaches share D3's goal of bounding execution cost in advance. D3 extends this to execution graphs rather than individual queries, incorporates dynamic runtime signals (failure rate, latency, depth), and includes an explicit graduated response (DEGRADE before REFUSE) rather than binary throttle/pass decisions.

D3 is positioned as a formal architecture that unifies trust, entropy, and refusal into a coherent execution model. It differs from circuit breakers (reactive rate threshold) by being pre-execution and history-preserving; from chaos engineering (diagnostic fault injection) by being operational runtime control; from radiation hardening (physical-layer fault correction) by addressing semantic-layer execution integrity; and from resource governors (static cost bounds) by incorporating dynamic trust signals and graduated mode transitions.

---

## 8. Conclusion

D3 defines a compute architecture in which execution eligibility is treated as a resource-governed property: correctness is consumable, trust gates every operation, and refusal is a first-class terminal state rather than a degraded error path. The formal kernel specification provides the theoretical foundation; two reference implementations provide empirical validation.

The experimental evidence establishes three concrete results. Bounded execution under composition: at fanout-16, RNOS limits cascade growth to 1,075× below uncontrolled; under composed failure modes with shared budgets, early containment is consistent across all tested seeds. Selective containment: RNOS correctly discriminates recoverable from non-recoverable trajectories without over-triggering, including withholding judgment correctly when execution traces are entropy-matched and the evidence is genuinely ambiguous. Quantified architectural boundaries: greedy per-unit admissibility cannot detect instability distributed across individually-benign units, a limit confirmed independently in both implementations and mapped to specific structural properties of the current admissibility formulation.

The adversarial analysis converts these limits from failures into extension points. Each identified gap — aggregate cost projection, phase-level reservation, committed-effect accounting — corresponds to a specific architectural addition with a quantified impact. The approach is not complete. It is a foundation with measured limits, and those limits are specified precisely enough to be addressed.

---

## Appendix A: Core Battery Summary

| Experiment | Context Reduction | Entropy Reduction | Effects Reduction | Refusal Mode |
|---|---|---|---|---|
| Retry Storm | 99.985% | 73.352% | 99.985% | Early (trust) |
| Fanout Explosion | 96.000% | 24.995% | 96.000% | Late (entropy) |
| Cascading Failure | 89.011% | 42.693% | 89.256% | Late (trust) |

---

## Appendix B: Case Studies in Missing Refusal Primitives

### B.1 Motivation

The D3 framework introduces refusal as a first-class execution outcome, enforced through admissibility constraints on entropy and trust. While Sections 2–5 demonstrate this behavior in controlled experimental settings, it is instructive to examine whether similar failure modes appear in real-world systems.

This appendix analyzes two well-documented historical failures:

- ☢️ The Chernobyl reactor accident
- 🚀 The Space Shuttle Challenger disaster

These cases are not direct analogues to D3, but structurally similar systems in which: (1) invalid system state was detectable, (2) continuation remained possible, and (3) catastrophic outcomes emerged through continued execution. We interpret both events through the lens of admissibility and refusal, identifying the absence of enforceable termination as a common failure mode.

### B.2 Case Study I: Chernobyl — Absence of Enforced Termination

#### B.2.1 Context

The Chernobyl reactor accident occurred during a systems test conducted under degraded and unstable operating conditions. The reactor design (RBMK) exhibited known instability characteristics under low-power configurations, including positive reactivity coefficients and delayed feedback effects. During the test sequence, operators reduced reactor power into an unstable regime, disabled or bypassed safety systems, and continued operation despite anomalous readings and procedural violations.

#### B.2.2 Admissibility Interpretation

Mapping to D3 variables:

- Entropy ($H$): increasing reactor instability (thermal fluctuation, xenon poisoning, reactivity variance)
- Trust ($T$): decreasing reliability of safeguards and adherence to operating procedures
- Admissibility ($A$): whether continuation of the test is justified

At multiple points in the sequence, $H$ was degraded due to unstable reactor configuration and $T$ was degraded due to disabled safeguards and procedural deviation. Under D3 admissibility:

$$A_t = 0 \quad \text{when } T_t < \tau_T \text{ or } H_t < \rho(o_{t+1})$$

These conditions were met prior to the final control actions.

#### B.2.3 Failure Mode

Despite violation of admissibility conditions, execution continued. The system lacked a mechanism to enforce shutdown when safety thresholds were exceeded, prevent continuation under degraded trust conditions, or disallow operation in known unstable regimes. This allowed the system to enter a positive feedback state, culminating in uncontrolled reactivity increase, steam explosion, and reactor core destruction.

#### B.2.4 Interpretation

The critical failure was not the initial instability, but the continuation of execution in a known invalid state. Chernobyl illustrates a system in which detection of invalid state was possible, but termination was not enforced.

### B.3 Case Study II: Challenger — Refusal Without Enforcement

#### B.3.1 Context

The Space Shuttle Challenger disaster occurred during launch under unusually low ambient temperatures. Prior engineering analysis had identified that the solid rocket booster O-ring seals were sensitive to temperature, with reduced sealing performance in cold conditions. On the day of launch, temperatures were below the range of prior safe operation, engineers from the contractor (Morton Thiokol) explicitly recommended against launch, and a formal "no launch" position was communicated.

#### B.3.2 Admissibility Interpretation

Mapping to D3 variables:

- Entropy ($H$): increased environmental uncertainty (temperature outside validated envelope)
- Trust ($T$): degraded confidence in component performance under those conditions
- Admissibility ($A$): whether launch should proceed

The engineering recommendation constitutes an explicit admissibility failure:

$$A_t = 0 \quad \text{(unsafe environmental conditions)}$$

#### B.3.3 Failure Mode

Despite the explicit refusal signal, the recommendation was revisited under managerial pressure, the decision was reversed, and launch was approved. The system exhibited detection of invalid state and generation of a refusal signal, but lack of enforcement.

#### B.3.4 Interpretation

The Challenger disaster demonstrates a distinct failure mode: refusal was identified but not enforced. The presence of a refusal signal alone was insufficient. Without a mechanism to make refusal binding, continuation was permitted, converting known risk into realized failure.

### B.4 Comparative Analysis

| Case | Detection | Refusal Signal | Enforcement | Outcome |
|---|---|---|---|---|
| Chernobyl | Partial | Implicit | Absent | Catastrophic |
| Challenger | Explicit | Present | Absent | Catastrophic |

Both systems satisfy the same pattern: (1) system enters degraded or invalid state; (2) signals indicate risk; (3) continuation remains permissible; (4) system transitions into a catastrophic regime. The shared failure mode is:

> **Systems that permit continuation after admissibility failure will eventually realize catastrophic outcomes under compounding dynamics.**

### B.5 Relevance to D3

The D3 framework addresses these requirements by defining admissibility in terms of entropy and trust, evaluating admissibility at each execution step, and enforcing refusal when conditions are violated:

$$A_t = 0 \Rightarrow \text{execution terminates}$$

In contrast to systems that allow continuation under degraded conditions, D3 ensures that refusal is not advisory but binding. The presence of a structured refusal output — identifying which constraint was violated, what state was preserved, and why execution halted — transforms termination from an exception into an observable, debuggable, integrity-preserving outcome.

The examined failures demonstrate that the absence of an enforceable refusal primitive does not prevent failure. It ensures that failure will compound until it becomes catastrophic.
