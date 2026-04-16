# D3, D3-Q, and D3-QS: A Formal Reconstruction

## Introduction

D3 begins from a simple observation: some execution systems do not fail by crashing; they fail by continuing under degraded conditions. Retry storms, recursive fanout, and dependency cascades can keep a system active while uncertainty accumulates and effects become progressively less trustworthy. The core idea of D3 is therefore not to ask whether execution *can* continue, but whether it *should* continue.

The original D3 paper proposes a thresholded execution model in which continuation is permitted only while an execution state remains admissible according to two control variables:

- entropy, representing uncertainty accumulation;
- trust, representing the quality of state and dependencies.

When the admissibility condition fails, the system does not crash or silently degrade. It produces an explicit terminal outcome: refusal.

This document reconstructs that idea as a formal system, identifies its main theoretical weakness, and develops two refinements:

- **D3-Q**, which replaces cumulative entropy with unresolved entropy and adds bounded quarantine;
- **D3-QS**, which adds safe suspension so that recoverable stochastic noise need not imply almost-sure eventual termination.

## Original D3 Model

The paper’s conceptual model is stepwise and thresholded. At each time step the system evaluates entropy, trust, and admissibility. The original admissibility rule is

$$
A_t = 
\begin{cases}
1, & T_t \ge \tau_T \text{ and } H_t \le \tau_H, \\
0, & \text{otherwise.}
\end{cases}
$$

Execution semantics are

$$
\mathrm{continue}(t) \iff A_t = 1,
$$

$$
\mathrm{refuse}(t) \iff A_t = 0.
$$

The paper also records a run-level output tuple

$$
O = \{\hat y, c, I, H, m, v, T, R\},
$$

where the intended components are prediction, confidence, integrity, entropy, execution mode, validation level, trust, and redundancy/refusal status.

The empirical claim in the paper is that this rule yields strong containment on adversarial workloads: large reductions in context creation and effect commitment, with both early and late refusal modes depending on the failure structure.

### Minimal axioms for D3

Let \((\Omega, \mathcal F, (\mathcal F_t)_{t \ge 0}, \mathbb P)\) be a filtered probability space.

**Axiom D1.** For each \(t \ge 0\),

$$
H_t : \Omega \to [0,\infty), \qquad T_t : \Omega \to [0,1]
$$

are \(\mathcal F_t\)-measurable.

**Axiom D2.** There exist fixed thresholds \(h^\star \in (0,\infty)\) and \(t^\star \in (0,1)\) such that

$$
A_t := \mathbf 1\{H_t \le h^\star,\ T_t \ge t^\star\}.
$$

**Axiom D3.** The refusal time is

$$
\tau_R := \inf\{t \ge 0 : A_t = 0\},
$$

and refusal is absorbing:

$$
A_t = 0 \quad \forall t \ge \tau_R.
$$

These axioms define the boundary and the terminal semantics, but they do not specify how \(H_t\) or \(T_t\) evolve.

## Formalization

### A concrete stochastic model

Work in discrete time \(t = 0,1,2,\dots\). Let the observable execution signal be

$$
Y_t = (B_t, V_t),
$$

where:

- \(B_t \in \mathbb N\) is a branching/fanout/retry count;
- \(V_t \in \{0,1\}\) is a validation indicator.

Define a healthy reference law \(\mu\) and an adversarial/degraded law \(\nu\) by

$$
\mu(b,v) = \mathrm{Pois}(b; \lambda_0)\, \mathrm{Bern}(v; p_0),
$$

$$
\nu(b,v) = \mathrm{Pois}(b; \lambda_1)\, \mathrm{Bern}(v; p_1),
$$

with

$$
\lambda_1 > \lambda_0, \qquad p_1 < p_0.
$$

Thus degraded execution has higher branching pressure and lower validation success.

### Trust \(T_t\)

Let \(G\) denote the latent event that execution remains in the healthy regime. Define trust as the posterior probability of health:

$$
T_t := \mathbb P(G \mid Y_{1:t}).
$$

If \(T_0 = \pi_0\), then Bayes’ rule gives

$$
T_t = \frac{\pi_0 \prod_{k=1}^t \mu(Y_k)}{\pi_0 \prod_{k=1}^t \mu(Y_k) + (1-\pi_0) \prod_{k=1}^t \nu(Y_k)}.
$$

Define log-odds

$$
L_t := \log \frac{T_t}{1-T_t}.
$$

Then

$$
L_t = L_0 + \sum_{k=1}^t W_k,
$$

where

$$
W_k := \log \frac{\mu(Y_k)}{\nu(Y_k)}.
$$

For the Poisson-Bernoulli model,

$$
W_k = B_k \log \frac{\lambda_0}{\lambda_1} + (\lambda_1 - \lambda_0)
+ V_k \log \frac{p_0}{p_1}
+ (1 - V_k) \log \frac{1-p_0}{1-p_1}.
$$

### Entropy \(H_t\)

A direct completion of the original D3 idea is cumulative surprisal under the healthy model:

$$
Z_k := -\log \mu(Y_k),
$$

$$
H_t := H_0 + \sum_{k=1}^t Z_k.
$$

This treats entropy as uncertainty budget consumed by the run.

### Admissibility and stopping rule

With thresholds \(h^\star\) and \(t^\star\), admissibility is

$$
A_t = \mathbf 1\{H_t \le h^\star,\ T_t \ge t^\star\}.
$$

Equivalently, in log-odds coordinates,

$$
\ell^\star := \log \frac{t^\star}{1-t^\star},
$$

$$
\tau_R = \inf\{t \ge 0 : H_t > h^\star \text{ or } L_t < \ell^\star\}.
$$

This makes D3 a first-exit system on the state space \((H_t, T_t)\).

## Key Failure: Monotonic Entropy -> Guaranteed Termination

The cumulative entropy model creates a structural problem.

If the execution signal is healthy, i.e. \(Y_t \sim \mu\) i.i.d., then

$$
\frac{H_t}{t} \to \mathbb E_\mu[-\log \mu(Y_1)] = H(\mu) > 0
\qquad \text{a.s.}
$$

for any non-degenerate healthy model. Hence

$$
H_t \to +\infty \qquad \text{a.s.}
$$

and therefore

$$
\tau_H := \inf\{t \ge 0 : H_t > h^\star\} < \infty
\qquad \text{a.s.}
$$

for every finite threshold \(h^\star\).

Thus, under cumulative entropy, D3 does not merely terminate adversarial runs. It imposes a hard execution horizon on any sufficiently long run, including benign ones. The source of termination is not adversariality per se, but the monotone growth of \(H_t\).

## D3-Q

D3-Q is a refinement that changes two parts of the model:

1. cumulative entropy \(H_t\) is replaced by unresolved entropy \(U_t\);
2. immediate terminal refusal is replaced by bounded quarantine before final refusal.

### Hidden-state formulation

Let \(X_t \in \mathcal X\) be a finite hidden state and let

$$
\pi_t(x) := \mathbb P(X_t = x \mid \mathcal F_t)
$$

be the belief state. Let \(a_t \in \mathcal A\) be the action at time \(t\), and let \(O_{t+1} \in \mathcal O\) be the next observation.

Define the predictive belief

$$
\pi_{t+1 \mid t}(x') = \sum_{x \in \mathcal X} \pi_t(x) P_t(x' \mid x, a_t),
$$

where \(P_t\) is the controlled state transition kernel.

Let Shannon entropy of a belief \(\mu\) be

$$
\mathsf H(\mu) := -\sum_{x \in \mathcal X} \mu(x) \log \mu(x).
$$

### Rigorous definition of information gain \(G_t\)

Define information gain as conditional mutual information:

$$
G_t := I(X_{t+1}; O_{t+1} \mid \mathcal F_t, a_t).
$$

Equivalently,

$$
G_t = \mathsf H(\pi_{t+1 \mid t}) - \mathbb E[\mathsf H(\pi_{t+1}) \mid \mathcal F_t, a_t].
$$

In explicit kernel form,

$$
G_t = \sum_{x' \in \mathcal X} \sum_{o \in \mathcal O}
\pi_{t+1 \mid t}(x') K_t(o \mid x', a_t)
\log \frac{K_t(o \mid x', a_t)}{\sum_{y \in \mathcal X} \pi_{t+1 \mid t}(y) K_t(o \mid y, a_t)}.
$$

A realized credited gain can be defined conservatively by

$$
\widehat G_{t+1} := \big[\mathsf H(\pi_{t+1 \mid t}) - \mathsf H(\pi_{t+1})\big]_+.
$$

### Unresolved entropy

Replace cumulative entropy by

$$
U_{t+1} = \big[U_t + Z_{t+1} - \rho \widehat G_{t+1}\big]_+,
$$

where:

- \(Z_{t+1} \ge 0\) is a new uncertainty charge;
- \(\widehat G_{t+1} \ge 0\) is credited information gain;
- \(\rho > 0\) converts certified information into retired uncertainty.

Unlike \(H_t\), unresolved entropy can decrease if the system gains enough information.

### Trust in D3-Q

Let \(\mathcal X_G \subseteq \mathcal X\) be the set of healthy states. Define

$$
T_t := \pi_t(\mathcal X_G) = \sum_{x \in \mathcal X_G} \pi_t(x),
$$

and again let

$$
L_t := \log \frac{T_t}{1-T_t}.
$$

### Low-risk action

Let \(E_{t+1}^a\) be the random external effect of action \(a\), let \(c : \mathcal E \to [0,\infty)\) be a measurable harm functional, and let \(\mathcal E_{\mathrm{irr}}\) be the set of irreversible effects.

For \(\alpha \in (0,1)\), define conditional tail risk by

$$
\operatorname{CVaR}_\alpha(c(E_{t+1}^a) \mid \mathcal F_t)
:=
\inf_{m \in L^1(\mathcal F_t)}
\left(
 m + \frac{1}{1-\alpha} \mathbb E[(c(E_{t+1}^a) - m)_+ \mid \mathcal F_t]
\right).
$$

Then an action is low-risk at time \(t\) if

$$
\operatorname{CVaR}_\alpha(c(E_{t+1}^a) \mid \mathcal F_t) \le \varepsilon
$$

and

$$
\mathbb P(E_{t+1}^a \in \mathcal E_{\mathrm{irr}} \mid \mathcal F_t) = 0.
$$

### Mode law

Let \(M_t \in \{E, Q, R\}\) denote execute, quarantine, refuse. Let \(b_t \in \{0,1,\dots,B\}\) be a non-replenishing quarantine budget with

$$
b_0 = B, \qquad b_{t+1} = b_t - \mathbf 1\{M_t = Q\}.
$$

Fix thresholds

$$
t_{\mathrm{hi}} > t_{\mathrm{lo}}, \qquad u_{\mathrm{hi}} < u_{\mathrm{lo}}.
$$

The D3-Q mode law is

$$
M_t =
\begin{cases}
E, & T_t \ge t_{\mathrm{hi}},\ U_t \le u_{\mathrm{hi}}, \\
Q, & T_t \ge t_{\mathrm{lo}},\ U_t \le u_{\mathrm{lo}},\ b_t > 0,\ \text{and not } E, \\
R, & \text{otherwise.}
\end{cases}
$$

High-impact actions are allowed only in \(E\). In \(Q\), only low-risk actions are permitted.

### Minimal additional axioms for D3-Q

In addition to Axioms D1--D3:

**Axiom Q1.** \(U_t, Z_{t+1}, \widehat G_{t+1}\) are adapted and satisfy

$$
U_{t+1} = [U_t + Z_{t+1} - \rho \widehat G_{t+1}]_+.
$$

**Axiom Q2.** The quarantine budget \(b_t\) is non-replenishing and bounded:

$$
b_t \in \{0,1,\dots,B\}, \qquad b_{t+1} = b_t - \mathbf 1\{M_t = Q\}.
$$

**Axiom Q3.** Thresholds satisfy

$$
t_{\mathrm{hi}} > t_{\mathrm{lo}}, \qquad u_{\mathrm{hi}} < u_{\mathrm{lo}}.
$$

**Axiom Q4.** Quarantine actions are restricted to the low-risk class defined above.

**Axiom Q5.** Refusal is absorbing:

$$
M_t = R \quad \forall t \ge \tau_R,
$$

where

$$
\tau_R := \inf\{t \ge 0 : M_t = R\}.
$$

## Formal Results

### Lemma 1: bounded-drift hitting bound

**Lemma 1.** Let \(X_t\) be adapted. Fix \(x > X_0\) and define

$$
\tau_x := \inf\{t \ge 0 : X_t > x\}.
$$

Assume that on \(\{t < \tau_x\}\),

$$
0 \le X_{t+1} - X_t \le c,
$$

and

$$
\mathbb E[X_{t+1} - X_t \mid \mathcal F_t] \ge \eta > 0.
$$

Then

$$
\mathbb E[\tau_x] \le \frac{x - X_0 + c}{\eta}.
$$

**Proof sketch.** Write

$$
X_{n \wedge \tau_x} = X_0 + \sum_{t=0}^{n-1} (X_{t+1} - X_t) \mathbf 1\{t < \tau_x\}.
$$

Take expectations, use the drift lower bound to control \(\mathbb E[n \wedge \tau_x]\), and use the overshoot bound \(X_{n \wedge \tau_x} \le x + c\). Then let \(n \to \infty\).

### Theorem 1: finite-time refusal for D3 under persistent adversarial drift

**Theorem 1.** Suppose

$$
H_t = H_0 + \sum_{k=1}^t Z_k, \qquad
L_t = L_0 + \sum_{k=1}^t W_k,
$$

where \(Y_k \sim \nu\) i.i.d.,

$$
Z_k := -\log \mu(Y_k), \qquad W_k := \log \frac{\mu(Y_k)}{\nu(Y_k)}.
$$

Let

$$
\tau_R = \inf\{t \ge 0 : H_t > h^\star \text{ or } L_t < \ell^\star\},
$$

with

$$
\ell^\star := \log \frac{t^\star}{1-t^\star}.
$$

If

$$
\mathbb E_\nu |Z_1| < \infty, \qquad \mathbb E_\nu |W_1| < \infty,
$$

and

$$
\mathbb E_\nu Z_1 > 0, \qquad \mathbb E_\nu W_1 < 0,
$$

then

$$
\tau_R < \infty \qquad \text{a.s.}
$$

**Proof sketch.** By the strong law of large numbers,

$$
\frac{H_t}{t} \to \mathbb E_\nu Z_1 > 0, \qquad \frac{L_t}{t} \to \mathbb E_\nu W_1 < 0
\qquad \text{a.s.}
$$

Hence \(H_t \to +\infty\) and \(L_t \to -\infty\) almost surely, so at least one threshold is crossed in finite time.

### Theorem 2: expected execution-length bound for D3-Q

**Theorem 2.** Assume D3-Q and define

$$
\ell_{\mathrm{hi}} := \log \frac{t_{\mathrm{hi}}}{1-t_{\mathrm{hi}}}.
$$

Assume that on \(\{M_t = E\}\),

$$
0 \le U_{t+1} - U_t \le c_U,
$$

$$
\mathbb E[U_{t+1} - U_t \mid \mathcal F_t] \ge \eta_U > 0,
$$

$$
0 \le L_t - L_{t+1} \le c_L,
$$

$$
\mathbb E[L_t - L_{t+1} \mid \mathcal F_t] \ge \eta_L > 0.
$$

Assume further that every entry time \(\sigma\) into execute mode satisfies

$$
U_\sigma \le u_{\mathrm{hi}}, \qquad L_\sigma \le L_+ < \infty.
$$

Then

$$
\mathbb E[\tau_R] \le (B+1) K_E + B,
$$

where

$$
K_E := \min \left\{
\frac{u_{\mathrm{hi}} + c_U}{\eta_U},
\frac{L_+ - \ell_{\mathrm{hi}} + c_L}{\eta_L}
\right\}.
$$

**Proof sketch.** Each execute episode ends when either \(U_t\) crosses \(u_{\mathrm{hi}}\) or \(L_t\) crosses below \(\ell_{\mathrm{hi}}\). Apply Lemma 1 separately to these two coordinates. The expected length of each execute episode is at most \(K_E\). Since each execute episode after the first requires at least one quarantine step and the global quarantine budget is \(B\), there are at most \(B+1\) execute episodes and at most \(B\) quarantine steps.

### Proposition 1: successful quarantine can still force eventual refusal

**Proposition 1.** There exists a D3-Q model such that:

1. every quarantine step is low-risk;
2. every quarantine step restores execute mode in one step with probability \(1\);
3. nevertheless \(\tau_R < \infty\) almost surely.

**Construction.** Let \(U_t \equiv 0\). Choose constants

$$
T^E \in (t_{\mathrm{hi}}, 1), \qquad T^Q \in [t_{\mathrm{lo}}, t_{\mathrm{hi}}).
$$

Let \((\xi_t)_{t \ge 0}\) be i.i.d. Bernoulli\((p)\), \(p \in (0,1)\), and set

$$
T_t =
\begin{cases}
T^E, & \xi_t = 0, \\
T^Q, & \xi_t = 1.
\end{cases}
$$

Every time \(T_t = T^Q\), the controller uses a low-risk quarantine action that deterministically returns the system to \(T_{t+1} = T^E\). Each such event consumes one unit of budget. Refusal occurs at the \((B+1)\)-st disturbance.

**Proof sketch.** The number of disturbances up to time \(n\) is binomial with parameter \(p\), and an i.i.d. Bernoulli process produces infinitely many disturbances almost surely. Therefore the \((B+1)\)-st disturbance occurs at finite time almost surely.

### Proven, assumed, heuristic

**Proven.** Lemma 1, Theorem 1, Theorem 2, Proposition 1, and the D3-QS results below are valid under their explicit hypotheses.

**Assumed.** The stochastic laws \(\mu\), \(\nu\), the threshold values, bounded-increment conditions, drift conditions, and action-risk model are assumptions, not consequences of the framework.

**Heuristic.** Interpreting \(T_t\) as “trust,” \(H_t\) or \(U_t\) as “integrity budget,” and \(\widehat G_t\) as operational information gain is a modeling decision. Threshold calibration and estimation in real systems remain open.

## Failure Modes

### D3 failures

#### 1. Transient payment outage

A payment system experiences a short upstream issuer outage. Suppose healthy validation success is

$$
p_0 = 0.995,
$$

while during a brief outage it drops to

$$
p_1 = 0.1.
$$

If \(T_0 = 0.99\) and two consecutive failures occur, then under the simple Bernoulli trust model

$$
T_2 = \frac{T_0 (1-p_0)^2}{T_0 (1-p_0)^2 + (1-T_0) (1-p_1)^2}
\approx 0.003.
$$

Thus trust falls below any reasonable threshold almost immediately, and D3 refuses. A baseline system with capped backoff can survive the outage and complete the transaction once the dependency recovers.

#### 2. Legitimate branch-and-bound search

In a search or optimization workload, high branching can be a productive part of the algorithm rather than a pathology. If the healthy reference uses \(\lambda_0 = 1\) but the search repeatedly branches with \(B_t = 4\), then each such step contributes

$$
-\log \Pr_\mu(B_t = 4) = -\log \left(e^{-1} \frac{1^4}{4!}\right) = 1 + \log 24 \approx 4.18.
$$

If \(h^\star = 100\), then only about

$$
\frac{100}{4.18} \approx 24
$$

such steps are enough to exhaust the entropy budget. D3 therefore confuses necessary exploration with instability.

#### 3. Degraded-but-useful security monitoring

Suppose a monitoring system can continue in a degraded advisory mode even when part of its upstream evidence is unavailable. If attack prevalence is \(\pi = 0.05\), degraded recall is \(r = 0.70\), false-positive rate is \(f = 0.05\), false-negative cost is \(C_{\mathrm{FN}} = 100\), and false-positive cost is \(C_{\mathrm{FP}} = 1\), then the degraded baseline has expected loss

$$
L_{\mathrm{base}} = \pi (1-r) C_{\mathrm{FN}} + (1-\pi) f C_{\mathrm{FP}}
= 0.05 \cdot 0.30 \cdot 100 + 0.95 \cdot 0.05 \cdot 1
= 1.5475.
$$

Under hard refusal, recall falls to zero, so

$$
L_{\mathrm{D3}} = \pi C_{\mathrm{FN}} = 5.
$$

Refusal is therefore strictly worse than bounded degraded output.

### D3-Q failures

#### Quarantine-budget exhaustion under recoverable oscillations

D3-Q repairs cumulative entropy, but it introduces path-dependent budget depletion. In the construction of Proposition 1, every disturbance is recoverable and every quarantine action succeeds. Nevertheless, because the quarantine budget is finite and non-replenishing, repeated i.i.d. recoverable disturbances imply eventual refusal almost surely.

The failure is not unsafe execution; it is the loss of long-run robustness under persistent but individually harmless noise.

## D3-QS (Safe Suspension Model)

To prevent almost-sure termination from finite quarantine budget alone, augment D3-Q with a fourth mode:

- \(E\): execute;
- \(Q\): quarantine;
- \(S\): safe suspension;
- \(R\): refuse.

The key idea is that budget exhaustion should not immediately imply refusal whenever trust and unresolved entropy remain within safe low-level bounds. Instead, the system enters a zero-effect suspension mode and waits for evidence of recovery.

### Mode law

Let \(r_t \in \{0,1,\dots,m\}\) be a clean-probe counter. The mode law is

$$
M_t =
\begin{cases}
E, & T_t \ge t_{\mathrm{hi}},\ U_t \le u_{\mathrm{hi}}, \\
Q, & t_{\mathrm{lo}} \le T_t,\ U_t \le u_{\mathrm{lo}},\ b_t > 0,\ \text{and not } E, \\
S, & t_{\mathrm{lo}} \le T_t,\ U_t \le u_{\mathrm{lo}},\ b_t = 0,\ \text{and not } E, \\
R, & T_t < t_{\mathrm{lo}} \text{ or } U_t > u_{\mathrm{lo}}.
\end{cases}
$$

The budget and recovery counter update by

$$
b_{t+1} =
\begin{cases}
b_t - 1, & M_t = Q, \\
B, & M_t = S,\ r_{t+1} = m, \\
b_t, & \text{otherwise,}
\end{cases}
$$

$$
r_{t+1} =
\begin{cases}
\min\{m, r_t + 1\}, & M_t = S,\ C_{t+1} = 1, \\
0, & M_t = S,\ C_{t+1} = 0, \\
0, & M_t \ne S,
\end{cases}
$$

where \(C_{t+1} \in \{0,1\}\) is a clean-probe indicator.

In \(S\), only a zero-effect probe is allowed, satisfying

$$
\operatorname{CVaR}_\alpha(c(E_{t+1}^{a_t^0}) \mid \mathcal F_t) = 0,
$$

$$
\mathbb P(E_{t+1}^{a_t^0} \in \mathcal E_{\mathrm{irr}} \mid \mathcal F_t) = 0.
$$

Thus safety constraints remain bounded and strict.

## Theorem: Infinite Safe Execution Under Noise

**Theorem 3.** Assume the recoverable disturbance model

$$
U_t \equiv 0,
$$

$$
T_t \in \{T^E, T^Q\},
$$

with

$$
T^E \ge t_{\mathrm{hi}}, \qquad T^Q \in [t_{\mathrm{lo}}, t_{\mathrm{hi}}),
$$

and let \((\xi_t)_{t \ge 0}\) be i.i.d. Bernoulli\((p)\), \(p \in (0,1)\), with clean-probe indicator

$$
C_t = 1 - \xi_t.
$$

Then under D3-QS,

$$
\tau_R = \inf\{t \ge 0 : M_t = R\} = \infty
\qquad \text{a.s.}
$$

Moreover, every safe-suspension episode ends in finite time almost surely.

**Proof sketch.** Because \(U_t = 0 \le u_{\mathrm{lo}}\) and \(T_t \ge t_{\mathrm{lo}}\) always hold, refusal cannot be triggered by the low-level safety boundary. The only question is whether safe suspension can become absorbing. But a block of \(m\) consecutive clean probes has probability \((1-p)^m > 0\), and non-overlapping such blocks occur infinitely often almost surely by Borel--Cantelli. Therefore every suspension episode eventually accumulates \(m\) clean probes, restores budget, and exits suspension.

## New Failure Mode: Safe Livelock

D3-QS avoids almost-sure termination under i.i.d. recoverable noise, but it introduces a new pathology: safe livelock.

**Proposition 2.** There exist bounded threshold-respecting disturbance sequences for which D3-QS never refuses and never resumes normal execution.

**Construction.** Take

$$
U_t \equiv 0, \qquad T_t \equiv T^Q \in [t_{\mathrm{lo}}, t_{\mathrm{hi}}), \qquad b_0 = 0,
$$

and choose the clean-probe sequence

$$
C_t = 1,0,1,0,1,0,\dots
$$

with recovery threshold \(m = 2\). Then the clean counter alternates between \(1\) and \(0\), never reaches \(2\), and budget is never restored. The system remains in \(S\) forever: it is safe, non-terminal, and non-progressing.

A stochastic analogue is that the expected suspension duration until \(m\) consecutive clean probes is

$$
\mathbb E[\theta] = \frac{1 - q^m}{p q^m}, \qquad q := 1-p,
$$

which grows rapidly as \(p \uparrow 1\) or \(m\) increases. Thus safe suspension can dominate runtime even when refusal never occurs.

## Core Tradeoff (Safety vs Progress vs Robustness)

The three models expose a sharp design tradeoff.

### D3

- strongest termination guarantee under one-sided cumulative entropy;
- strongest containment of amplification;
- weakest tolerance for long benign execution and transient degradation.

### D3-Q

- removes the monotonic-entropy defect by using unresolved entropy;
- permits bounded low-risk recovery before final refusal;
- still fails under persistent recoverable noise because the quarantine budget is finite and non-replenishing.

### D3-QS

- preserves bounded safety constraints and low-risk recovery;
- makes infinite safe execution possible under stochastic recoverable noise;
- loses finite wall-clock termination guarantees and introduces safe livelock/starvation.

The core frontier is therefore:

- stronger refusal guarantees imply stronger containment but weaker robustness;
- stronger robustness to recoverable noise requires non-absorbing intermediate states, which weakens progress guarantees.

## Open Questions / Future Work

1. **Threshold calibration.** How should \(t^\star\), \(h^\star\), \(t_{\mathrm{hi}}\), \(t_{\mathrm{lo}}\), \(u_{\mathrm{hi}}\), \(u_{\mathrm{lo}}\), \(B\), and \(m\) be learned or adapted?
2. **Trust estimation.** In real systems, what observation model should replace the stylized Poisson-Bernoulli law?
3. **Operational information gain.** How should \(\widehat G_t\) be credited in systems with approximate tests, partial observability, or delayed validation?
4. **Action typing.** Can low-risk actions be verified compositionally, rather than by a single-step risk bound?
5. **Non-i.i.d. disturbances.** The present arguments use independent noise. Correlated or adversarial noise needs separate analysis.
6. **Performance guarantees.** What bounds are possible on utility, task completion, or regret under D3-like safety rules?
7. **Livelock control.** Can D3-QS be extended with fairness or progress certificates that prevent indefinite safe suspension without reintroducing almost-sure refusal under benign noise?

## Intuition & Plain-English Summary

D3 says: a system should not continue merely because it is able to continue. If uncertainty has piled up too far, or confidence in the system’s own state has fallen too low, the right outcome may be to stop explicitly.

That idea is attractive, but the naive formalization has a flaw. If “entropy” is just cumulative uncertainty spent over time, then every long run eventually looks bad, even a healthy one. The model guarantees termination because the budget only moves in one direction.

D3-Q fixes that by replacing cumulative entropy with unresolved entropy: if the system learns enough, uncertainty can go back down. It also adds quarantine, so the system can attempt bounded low-risk recovery instead of refusing immediately.

But finite quarantine budget creates a new problem. Under endless small disturbances, even perfectly successful recoveries consume budget, so eventual refusal can still happen almost surely.

D3-QS fixes that by adding safe suspension: when the system is not healthy enough to execute but still safe enough not to refuse, it can stop acting, probe carefully, and wait for evidence of recovery. This avoids almost-sure refusal under recoverable stochastic noise.

The final tradeoff is clear:

- **D3** prioritizes containment;
- **D3-Q** balances containment with bounded recovery;
- **D3-QS** prioritizes safety under noise, but can stall forever.

A publishable theory of refusal-based execution therefore needs to state explicitly which of the three it wants to guarantee: termination, progress, or robustness. No single threshold system gives all three for free.
