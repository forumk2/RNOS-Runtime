# D3: Deterministic Degradable Distributed
## A Compute Architecture for Execution Under Uncertainty

**Rowan Ashford**

*Working draft v0.2 — April 2026*

---

## Abstract

Modern execution systems fail by continuing: they permit operation under degraded conditions, compounding instability through retries, recursive expansion, and effect commitment until the system enters a state from which recovery is no longer possible. D3 (Deterministic Degradable Distributed) addresses this structurally by treating correctness as a consumable resource. Each operation is evaluated against an entropy budget before execution; trust — a composite of confidence, integrity, and environmental stability — gates continuation. When admissibility fails, the system issues a structured refusal, an integrity-preserving terminal state that halts further execution rather than permitting silent degradation. We present the D3 formal kernel specification, an instruction set that exposes entropy and trust as architecturally visible state, and a five-layer execution stack that connects mission-level policy to fault containment. Two reference implementations — RNOS-2.0 and RNOS-Runtime — instantiate key aspects of the architecture in software and provide empirical validation across containment, discrimination, and adversarial regimes. At fanout-16, RNOS limits cascade growth to 65 contexts against 69,905 unprotected (1,075×); under composed failure, early containment is achieved consistently across six seeds with average entropy savings of 54%. Adversarial evaluation reveals that greedy per-unit admissibility cannot detect instability distributed across individually-benign operations, a boundary confirmed independently in two experimental regimes. A cooperative control experiment demonstrates that composing RNOS with an adaptive circuit breaker under a safety-first merge matches or improves upon the better single-policy controller across both cascading and distributed failure geometries, providing initial evidence that composed admissibility observers over distinct instability signals can address the identified detection gaps. Extension of this composition to a three-observer architecture — adding a persistence observer operating on a longer temporal horizon — across database and CI pipeline simulation domains produces structurally isomorphic dominance patterns across both domains, providing preliminary cross-domain evidence that execution instability decomposes into at least three orthogonal observable geometries (structural growth, failure density, sustained drift) separable by observers at distinct temporal scales. Entropy weights are hand-tuned; all experiments use synthetic deterministic workloads; the ISA and hardware layer are not yet implemented.

---

## 1. Introduction

Modern execution systems fail in a consistent and under-addressed way: they continue operating after the justification for continuation has degraded. Availability is treated as sufficient for execution, while correctness is evaluated only after the fact. As a result, systems often fail not by producing incorrect results immediately, but by compounding instability through continued execution—amplifying retries, expanding call graphs, and committing effects under increasingly degraded conditions.

Existing control mechanisms—timeouts, retry limits, and circuit breakers—are reactive. They detect failure after it has manifested in observable signals such as error rates or latency thresholds. These mechanisms answer the question "has this call failed?" but not the more fundamental question: "should execution continue at all?" In scenarios where instability accumulates without crossing local thresholds—such as distributed low-grade failure or cost fragmentation—these approaches permit execution to proceed well beyond the point of recoverability.

This paper introduces D3 (Deterministic Degradable Distributed), a compute architecture that treats execution eligibility as a resource-governed property. In D3, continuation is not assumed; it is granted. Each operation is evaluated against a bounded model of execution uncertainty—formalized as entropy—and is permitted to proceed only if sufficient budget remains. When admissibility fails, the system issues a structured refusal: a terminal state that preserves integrity by halting further execution.

D3 is not a runtime optimization or a single system implementation. It is an architectural model spanning a formal kernel specification, an instruction set exposing entropy and trust as first-class state, and a scheduling framework that conditions execution on system integrity. This paper presents two reference implementations—RNOS-2.0 and RNOS-Runtime—that instantiate key aspects of the architecture in software, enabling empirical evaluation of its behavior under controlled conditions.

The experimental results demonstrate three primary properties. First, D3 enforces bounded execution under composition, limiting cascade growth by constraining execution at the point of admissibility. Second, it achieves selective containment, distinguishing between recoverable and non-recoverable execution trajectories without premature intervention. Third, adversarial evaluation reveals clear architectural boundaries: specifically, that greedy per-unit admissibility cannot detect instability distributed across individually-benign units. These boundaries are quantified and shown to arise from the structure of the admissibility formulation rather than implementation error. A follow-on cooperative control experiment demonstrates that composing RNOS with a complementary detection primitive — an adaptive circuit breaker with sliding-window failure density — produces a controller that matches or improves upon either component across both cascading and distributed failure geometries, pointing toward composed admissibility observers as a direction for addressing the identified structural gaps.

The contribution of this paper is not to claim a complete solution to execution under uncertainty, but to define a coherent architectural approach, validate it empirically, and identify its limits precisely. D3 establishes a foundation for treating correctness as a schedulable resource and refusal as a first-class primitive in execution systems.

Preliminary evidence from experiments across two additional simulation domains — database query execution and CI pipeline control — suggests that effective execution control may require admissibility observers at multiple temporal scales rather than a single composite signal. A persistence observer operating on a longer horizon than either RNOS or the circuit breaker detects sustained low-rate instability that neither short-horizon detector identifies, and the three-observer dominance pattern reproduces across both domains without domain-specific tuning.

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

This calibration result validates a reproducible methodology: set charges by structural reasoning about relative risk ordering, then sweep empirically to confirm or adjust magnitudes. The finding that heuristic values preserved correct relative ordering while uniformly under-charging the load-bearing parameters suggests the methodology produces sound structure even when initial magnitudes are conservative. This is relevant for deployments where the specific workload characteristics differ from the calibration battery but the structural risk ordering is preserved — the ordering can be trusted while magnitudes are re-calibrated against the target workload.

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

The Terafab D3 hardware program (Section 5.7 Future Directions) provides physical-layer survival mechanisms — radiation hardening, TMR on critical paths, ECC/EDAC on all memory structures, and continuous scrubbing. D3/RNOS provides the semantic execution layer: governing whether and how computation proceeds once physical errors have been detected and corrected. The two layers are complementary; neither substitutes for the other. Radiation hardening prevents bit flips; RNOS prevents continued execution under conditions where bit flips, retry storms, and cascading effects have compromised the integrity of the execution context.

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
| 5 — Cooperative Control | Cascading burst + distributed density | 7 / 30 exec | 10 / 10 exec | 30 / 30 exec | Hybrid matches best component in both regimes; trigger_source identifies governing signal per step (see Section 5.3) |

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
| Cooperative Control | RNOS-Runtime, Exp 5 | 7 exec (cascading) / 30 (distributed) | 10 / 10 exec | 30 / 30 exec | Hybrid = best(RNOS, CB) both regimes; trigger transparent |
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

The structural pattern of detectable invalid state combined with permissible continuation appears in well-documented real-world failures: reactor accidents where anomalous readings preceded continued operation under known-unstable conditions, and launch disasters where an explicit refusal signal was overridden rather than enforced. In both cases, the failure was not the initial degraded state but the absence of a mechanism that made termination binding once admissibility conditions were violated. These are structural analogies illustrating what the absence of an enforceable refusal primitive produces — they are not technical mappings of D3 variables to reactor physics or component failure modes, and no claim is made that entropy or trust as defined in the D3 kernel spec would have quantitatively characterized those systems. The point is narrower: across domains, the pattern of compounding failure under permissible continuation is structurally recurring, and the D3 refusal primitive is designed to interrupt it at the execution layer.

### 5.2 Complementary Detection Profiles

RNOS and the adaptive CB have different architectural foundations that produce complementary detection profiles. RNOS's structural floor — cumulative execution cost that persists across recovery windows — provides memory of past instability that the CB's sliding window discards. This gives RNOS a 7-step advantage on `intermittent_cascade`, where burst 1 has faded from the CB's window by the time burst 2 triggers. The CB's advantage on diffuse failure (`smoldering_instability`) arises from its failure density accumulation being independent of consecutiveness — a structural property RNOS's retry-based scoring cannot replicate.

Neither approach dominates. A system combining both — cumulative cross-run memory for structured cascades, sliding-window density for diffuse distributed failure — would likely cover the identified gaps. The current RNOS implementations contain no equivalent of the CB's density accumulation, and the CB contains no equivalent of the RNOS structural floor.

### 5.3 Cooperative Control Architectures

Section 5.2 establishes that RNOS and the adaptive CB have structurally complementary detection profiles: cumulative execution history advantages RNOS on intermittent cascades; failure density accumulation independent of consecutiveness advantages the CB on distributed instability. The natural question is whether this complementarity is merely diagnostic — a description of two detection regimes operating in parallel — or whether it can be operationalized into a composed architecture that inherits the advantages of both without regressing on either.

To test this, a hybrid controller was constructed by composing RNOS and the adaptive CB under a safety-first merge: both sub-systems evaluate each step independently; the more-severe decision governs the control output. Severity is defined by decision level — ALLOW at 0, DEGRADE at 1, REFUSE at 2 — with the CB's state mapped to the equivalent level (CLOSED/ALLOW; HALF_OPEN/DEGRADE; OPEN or PERMANENTLY_OPEN/REFUSE). A `trigger_source` field records which sub-system produced the governing decision at each step, making the contributing signal observable at per-step resolution rather than opaque at the controller level.

Two failure scenarios were evaluated:

**`cascading_burst`** presents rapid consecutive failures beginning at step 3, escalating into an absorbing failure regime. RNOS's `retry_score` — 1.0 per consecutive failure, cap 4.0 — accumulates quickly under this pattern. Combined with the structural floor (cost\_score + repeated\_tool = 4.0, non-resetting), the accumulated entropy score crosses the DEGRADE threshold (9.0) at step 7 before the CB's 10-step sliding window has accumulated sufficient observations to evaluate. RNOS is the governing sub-system; `trigger_source = "rnos"` at the first intervention.

**`distributed_low_rate`** presents a repeating F-F-S pattern (67% failure rate, consecutive failures capped at 2). Each success resets `retry_count`, bounding `retry_score` at 2.0. Failure score over the most recent five steps reaches at most 3/5 × 0.65 × 3 = 1.95. Total accumulated entropy — retry 2.0 + failure 1.95 + floor 4.0 + latency ~0.1 — peaks at approximately 8.7, below the DEGRADE threshold of 9.0. RNOS does not intervene at any step within the 30-step evaluation window. The CB's sliding window fills after 10 executions with 7/10 = 0.70 > 0.60, triggering intervention at step 11. The CB is the governing sub-system; `trigger_source = "cb"` at the first intervention.

**Table: tool executions before termination (30-step maximum)**

| Scenario | Baseline | RNOS | CB | Hybrid | Governing sub-system |
|---|---|---|---|---|---|
| `cascading_burst` | 30 | 7 | 10 | **7** | RNOS |
| `distributed_low_rate` | 30 | 30 | 10 | **10** | CB |

In both scenarios, the hybrid controller matches the better-performing single-policy controller exactly and strictly improves upon the weaker one. On `cascading_burst`, hybrid reduces executions to 7 against the CB's 10; on `distributed_low_rate`, hybrid reduces executions to 10 against RNOS's 30. The safety-first merge routes authority to the sub-system with the relevant detection capability in each regime without requiring cross-sub-system coordination.

Three properties of this result merit attention.

First, the bound is structural. The safety-first merge guarantees that hybrid performance cannot be worse than the better-performing component in any scenario where at least one component intervenes. This is not a tuning outcome; it follows from the severity ordering. The two scenarios above are proof-of-concept, not an exhaustive evaluation. Whether this bound holds across adversarial, compositional, or stochastic failure regimes requires further testing.

Second, the mechanism is transparent. The `trigger_source` field provides a per-step record of which detection primitive governs each decision. A composed controller that cannot identify which of its components is responsible for a given refusal has traded the interpretability of each component for combined coverage. The per-step trigger record preserves both. In the language of D3 mode control: the hybrid does not replace the mode signal — it augments it with a provenance tag.

Third, the composition imposes no regression. Hybrid performance on `cascading_burst` is identical to RNOS-alone (7 executions); hybrid performance on `distributed_low_rate` is identical to CB-alone (10 executions). The safety-first merge does not weaken either sub-system's result on the scenario for which it was the stronger detector.

These results suggest a broader architectural principle for admissibility under uncertainty. A single control primitive defined over one instability geometry — cumulative execution history, local failure density, gradient, or latency — cannot be sufficient across all failure regimes that an execution system may encounter. Different geometries expose different observable signals; different signals require different detection structures. Rather than attempting to subsume all instability into a single composite score, a composed architecture assigns each detection responsibility to the primitive best suited to it and admits execution only when all active observers agree.

In D3 terms, this is an extension of the admissibility condition (Section 2.3). Rather than a single trust function $T$ governing admission, a composed architecture evaluates a set of admissibility observers $\{T_1, T_2, \ldots, T_k\}$, each defined over a distinct instability geometry, and requires:

$$A_t = 1 \iff \min_i\, T_i(\tau_t, s_t) \geq T_{\text{req}}$$

The safety-first merge is the operational realization of this condition: if any observer finds the system inadmissible, execution is refused, and the governing observer is recorded. The current two-component system — RNOS structural memory plus CB failure density — is the simplest instance. Persistence-aware observers, modeling stability streaks and chronic failure rate as identified observationally in Experiment 4, are natural additions within the same framework. The architectural lesson is not that more observers are always better, but that **execution control over diverse failure geometries appears to benefit from composed observers operating over distinct signals, with refusal triggered by the most severe valid assessment available**.

### 5.4 Tri-Modal Control Architecture

The cooperative control result (Section 5.3) demonstrates that two observers — RNOS and the adaptive CB — cover complementary failure geometries under a safety-first merge. The question this raises is whether the decomposition extends: whether execution instability separates into more than two orthogonal detectable geometries, and whether a third observer operating at a different temporal scale can cover failure regimes that both RNOS and CB miss, without interference with either.

To test this, a domain-agnostic PersistenceController was constructed and evaluated across database query execution and CI pipeline simulation domains, each implementing three scenarios designed to isolate distinct failure geometries. The central finding is not that persistence catches a third failure class in isolation — it is that both domains produce structurally isomorphic controller dominance patterns despite having no shared implementation components.

**Cross-domain convergence.** The following table combines results from both domains (metric: executions before first REFUSE termination; 20-step maximum):

| Domain | Scenario | RNOS | CB | Persist | Hybrid | Governing signal |
|---|---|---|---|---|---|---|
| DB | `cascading_query_explosion` | **8** | 20 | 20 | **8** | RNOS |
| DB | `lock_contention` | 20 | **6** | 11 | **6** | CB |
| DB | `slow_lock_drift` | 20 | 20 | **11** | **11** | Persistence |
| CI | `pipeline_fanout_explosion` | **6** | 20 | 20 | **6** | RNOS |
| CI | `flaky_test_storm` | 20 | **6** | 11 | **6** | CB |
| CI | `gradual_flakiness` | 20 | 20 | **11** | **11** | Persistence |

The dominance pattern is identical across both domains: RNOS governs structural growth scenarios, CB governs failure density scenarios, Persistence governs slow-drift scenarios. In all six cases, the hybrid controller matches the best individual controller exactly. The pattern is isomorphic across domains — the same three-way partitioning of controller dominance appears in each domain independently — and this isomorphism is the headline finding. If the decomposition into three failure geometries were an artifact of a particular implementation or domain design choice, the same partition would not be expected to reproduce across independently constructed DB and CI simulations with different state representations and different entropy signal definitions.

**What each observer detects.** The three observers are sensitive to structurally distinct failure signals. RNOS accumulates a composite entropy score over the full execution run — join depth and spawned cost in the DB domain; active job count, cumulative spawned jobs, and retry accumulation in the CI domain — that rises under structural expansion regardless of whether individual steps succeed or fail. The circuit breaker maintains a short sliding window of recent failure outcomes and trips when the failure rate within that window exceeds a threshold; it is insensitive to structural expansion and responds purely to failure concentration. Persistence operates over a longer window (10 steps) and computes a score over sustained failure rate and time spent above an entropy floor:

$$\text{persistence\_score} = 0.7 \times \text{failure\_rate} + 0.3 \times \text{time\_above\_entropy\_floor}$$

The weights are heuristic. The persistence observer trips when this score exceeds 0.50 over a full window, requiring both sustained failures and sustained structural pressure — not pure failure density (which CB handles) and not pure structural load (which RNOS handles). This combination is what isolates the slow-drift regime.

**Temporal scale separation.** The three observers operate at distinct temporal scales, and this separation is the core architectural property enabling non-interfering composition. RNOS evaluates structural entropy per step, accumulating monotonically from run start. CB evaluates failure density over a short sliding window (5 steps). Persistence evaluates sustained failure rate and entropy floor time over a longer fixed window (10 steps minimum) and requires the window to be full before any alert can fire.

The 10-step minimum fill requirement is a structural design choice, not a tuning parameter. Fast-diverging scenarios — structural fanout and burst failure density — terminate at steps 6–8 in the evaluated experiments, before the persistence window is full. Persistence therefore cannot produce an alert on these scenarios by construction: the run ends before the window fills. Slow-drift scenarios, by contrast, do not trigger RNOS or CB within the 20-step evaluation window, allowing the persistence window to fill and the sustained pattern to accumulate. The temporal separation prevents observer interference without requiring threshold coordination between the three observers. This property — that different failure geometries manifest instability at different temporal scales, and that observers designed at matching scales produce natural non-interference — is referred to here as **multi-scale control**.

**Composition property.** The safety-first merge from Section 5.3 extends directly to three observers: the maximum severity across RNOS, CB, and Persistence governs the merged decision. A `trigger_source` field records the governing observer at each step. In all six evaluated scenarios, the hybrid controller matches or improves upon the best individual controller, and no regression is observed. On structural and density scenarios, Persistence's window has not filled when the faster observer fires; the merge result is therefore determined entirely by RNOS or CB. On slow-drift scenarios, RNOS and CB remain below their respective thresholds throughout, and Persistence carries the governing signal. The formal admissibility condition from Section 5.3 extends directly to three observers:

$$A_t = 1 \iff \min\bigl(T_{\text{RNOS}}(\tau_t, s_t),\; T_{\text{CB}}(\tau_t, s_t),\; T_{\text{persist}}(\tau_t, s_t)\bigr) \geq T_{\text{req}}$$

Each observer evaluates a distinct instability geometry; refusal is issued when any observer finds conditions inadmissible.

**Limitations of this result.** All six scenarios are synthetic and deterministic. The two simulation domains share a common Python implementation layer and were designed by the same methodology; the cross-domain claim reflects different state representations and domain-specific entropy signals, not fully independent experimental codebases. Persistence scoring weights and the 10-step window are heuristic and were not calibrated against a broader workload distribution. The claim supported by these results is bounded: three orthogonal geometries — structural growth, failure density, sustained drift — are observable and separable in the evaluated regimes. Whether this decomposition is exhaustive, or whether additional orthogonal geometries exist that none of the three tested observers can detect, is an open question. The isomorphic pattern across two domains is structurally consistent and methodologically deliberate, but six scenarios across two domain simulations do not establish universality.

Execution control under uncertainty may require multiple admissibility observers operating across distinct temporal and structural scales, with refusal determined by the most severe valid signal available at each step.

### 5.5 Architectural Boundaries

Three boundaries are empirically identified and measured:

**No aggregate cost tracking.** The fragmentation attack demonstrates this with 96% pre-refusal execution. The gate evaluates each context independently; a cascade that is safe at the per-context level can be catastrophic at the aggregate level. Addressing this requires aggregate cost projection across planned siblings before admitting the first — a look-ahead property the current formulation lacks.

**No phase-level budget reservation.** The delayed amplification attack demonstrates this with 0% savings in 2/6 seeds. With no mechanism to reserve budget for known future phases, a cheap early sequence consumes the headroom that a late expensive phase requires. In the extreme case, the entire cascade completes without refusal. Addressing this requires per-phase reservation, which the single shared-counter budget model does not provide.

**Rollback-aware accounting.** The refund illusion attack demonstrates that released effect reservations extend the admitted-context sequence beyond what committed-effect accounting would allow. The effects gate provided partial mitigation in this specific experiment (network effects were blocked by sandbox policy); that mitigation is not guaranteed when the effects in question are policy-allowed.

Each gap is quantified: 96% pre-refusal execution, 0% savings (complete bypass), containment delayed from step 4 to step 8. These are not threshold-tuning issues. They represent structural properties of the admissibility formulation that require architectural changes — aggregate cost projection, phase-level reservation, committed-effect accounting — to address.

### 5.6 Theory and Implementation

The kernel specification defines 19 sections of formal semantics. RNOS-2.0 and RNOS-Runtime implement a subset: entropy budgeting, admissibility, mode-based graduated response, effects containment, and trust degradation (RNOS-Runtime). Unimplemented kernel features — full redundancy semantics, validation escalation (VSET/VCMP/VCHK), entropy field integration, the complete ISA — represent architectural extensions rather than current capabilities.

The relationship between theory and implementation is two-directional. The kernel spec provides the formal grounding: trust function axioms, safety guarantee conditions, mode control surface definitions. The implementations provide empirical feedback: the adversarial analysis identifies which architectural properties are exploitable and quantifies the impact, pointing to specific kernel spec sections (per-operation admissibility, absence of inter-context budget reservation, rollback semantics) that require extension. The gaps identified in Section 4.6 correspond to architectural additions, not parameter tuning.

### 5.7 Future Directions

The Terafab program — a radiation-hardened processor family targeting orbital AI compute — operates at the physical fault-correction layer: ECC/EDAC on all memory structures, TMR on critical paths, and continuous scrubbing prevent physical bit errors from propagating silently. D3/RNOS operates at the semantic execution integrity layer: it governs whether execution that has survived physical fault correction should continue given current entropy and trust state. The two layers are complementary — radiation hardening does not prevent retry storms or cascading semantic failures; entropy-bounded execution does not prevent physical bit corruption — and neither substitutes for the other. Proposed ISA bridge instructions (RADMON, THERMCHK, EBUDGET) would translate physical fault signals into entropy budget charges, enabling proactive mode transitions before semantic integrity is compromised, but these instructions are unimplemented proposals and no integration between D3/RNOS and any Terafab hardware has been validated.

Several open directions follow directly from the experimental results. The persistence signals identified in Experiment 4 — stability streak, rolling failure rate over a longer window, average recent latency — discriminate `smoldering_instability` cleanly but are not yet part of the entropy composition; integrating them would close the 0.195-unit detection gap without requiring threshold changes. Aggregate cost projection across planned sibling contexts, and phase-level budget reservation for known future phases, are architectural additions required to address the fragmentation gap (96% pre-refusal execution in Budget Fragmentation) and the delayed amplification bypass (complete bypass in 2/6 seeds) identified in Section 4.5. Finally, the adaptive CB has not been evaluated against the composite cascade scenario (Section 4.4); testing whether the CB achieves early containment under cross-phase budget depletion would establish whether RNOS's shared-budget mechanism provides an advantage that per-call failure density cannot replicate.

---

## 6. Limitations

### 6.1 Initial Budget Sensitivity

The adversarial analysis (Section 4.5) assumes a correctly-sized budget of 50 units in the composite cascade. This assumption is not validated. An over-provisioned budget extends the pre-refusal execution window for all attacks proportionally: Budget Fragmentation's 96% pre-refusal execution and Delayed Amplification's complete bypass in 2/6 seeds both become strictly worse with more headroom, since a larger budget allows more individually-affordable contexts to accumulate before any global signal fires. An under-provisioned budget increases premature refusal on legitimate workloads, rejecting execution at points where continuation is correct. No systematic sensitivity analysis has been conducted across budget sizes; the interaction between budget sizing and the identified structural gaps — aggregate cost projection and phase-level reservation — is an open question, since those gaps are themselves functions of how much headroom remains when the attack phase begins. This should be understood as a limitation of the current evaluation, not as evidence that the 50-unit budget is optimal or representative.

**Synthetic deterministic workloads.** All experiments use fixed failure schedules without stochastic variation. Results characterize behavior on the specific scenarios evaluated. Whether these profiles generalize to real workloads with non-deterministic failure timing, variable latency distributions, and concurrent interacting failure modes is untested.

**Threshold values are calibrated per experimental regime.** RNOS-Runtime discrimination experiments use thresholds of DEGRADE = 9.0 and REFUSE = 11.0, which differ from illustrative values cited in repository documentation (DEGRADE = 3.0, REFUSE = 6.0). The calibrated values account for a structural entropy floor produced by repeated tool use and cumulative cost; the illustrative values are explanatory defaults. This distinction does not affect the policy structure but should be kept in mind when comparing results across contexts.

**Entropy weights are hand-tuned.** Charge parameters and thresholds were set by design judgment and confirmed by a single calibration study over three adversarial scenarios. The calibration confirmed relative ordering and identified three load-bearing parameters, but the search space was limited to a univariate sweep followed by a joint grid search. Alternative weight assignments would produce different detection boundaries. The 0.195 entropy gap in Experiment 4 and the 8.8% savings in Budget Fragmentation are consequences of specific weight choices; they are not claimed to be worst-case bounds.

**White-box adversarial attacks.** Full knowledge of RNOS internals — cost model constants, gate evaluation logic, budget accounting mechanics, threshold values — was used to design each attack. A gray-box or black-box attacker faces additional uncertainty. The measured attack effectiveness quantifies what is achievable with deliberate optimization; it does not represent the expected impact of naturally-occurring workloads.

**Four attacks are not exhaustive.** The adversarial experiments cover four properties identified by inspection of the architecture. Sandbox-crossing surcharges, depth-dependent cost scaling, descendant count limits, and interactions among multiple attack patterns were not evaluated. Additional vulnerabilities may exist in compositions not examined here.

**Circuit breaker is a single baseline.** The adaptive CB with sliding-window failure rate, exponential backoff, and adaptive threshold is one instantiation of the circuit breaker pattern. Multi-signal breakers combining failure density, latency histograms, error type classification, and saturation metrics might perform differently. The finding that CB outperforms RNOS on Experiment 4 is specific to this configuration.

**The cooperative control experiment evaluates two synthetic scenarios.** The safety-first merge result (Section 5.3) is demonstrated on one cascading-burst scenario and one distributed-density scenario, both using fully deterministic step schedules. Generalization to adversarial, compositional, or stochastic failure regimes, to more than two composed sub-systems, and to scenarios where both sub-systems are simultaneously active has not been tested. The structural bound — hybrid cannot be worse than the better component — holds by construction; whether the empirical performance advantage is robust across a broader scenario space is an open question.

**The tri-modal results are preliminary.** The cross-domain isomorphism observed across database and CI pipeline simulation domains (Section 5.4) is structurally consistent but limited to two additional domains beyond the original RNOS-Runtime scenarios. The two simulation domains share a common implementation layer and methodology; the cross-domain claim reflects different state representations and entropy signal definitions, not fully independent experimental codebases. Persistence scoring weights (0.7 × failure_rate + 0.3 × time_above_entropy_floor) and the 10-step window are heuristic parameters confirmed adequate for the evaluated scenarios but not calibrated against a broader workload distribution. The claim that three orthogonal geometries — structural growth, failure density, sustained drift — cover the full space of execution instability failure modes cannot be established from six deterministic scenarios; the isomorphic pattern across two domains is suggestive but not conclusive. Additional failure geometries beyond the three tested may exist, and the persistence observer as designed would not detect them.

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

**Structured concurrency.** Structured concurrency frameworks (Swift's TaskGroup, Kotlin coroutines, Java Loom) impose lifecycle discipline on concurrent tasks, ensuring child tasks complete before parent scope exits. This provides structural containment analogous to D3's depth and fanout limits. D3 extends this with budget-governed admission and trust-gated continuation; structured concurrency provides lifecycle management but not admissibility control based on accumulated instability. A Swift `TaskGroup`, for example, enforces that all child tasks complete before the parent scope exits — but it has no mechanism to refuse spawning a new child based on the accumulated failure history or budget depletion of prior children in the same group. The lifecycle of the group is bounded; the admission decision for each new child is not conditioned on what the prior children cost. This is the specific gap D3 fills: the admissibility gate fires before each child is spawned, using the residual entropy budget and accumulated trust state from all prior execution in the run.

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

