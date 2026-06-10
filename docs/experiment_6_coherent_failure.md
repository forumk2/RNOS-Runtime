# Experiment 6: Coherent-Failure Detection

**Date:** 2026-06-09  **Seeds:** 20  **Steps per run:** 25
**Threshold sets:** default (3.0/6.0) and canonical (7.5/10.0)
**Branch step b*:** 6  **T_correct:** 10.0  **T_wrong:** -10.0

## §0 Entropy Floor (post-refactor formula)

For a zero-failure accumulator run with alternating tool names (step_toward / add):

| Signal | Value | Formula |
|---|---|---|
| depth_score | 0 | depth=0 always |
| retry_score | 0 | retry_count=0 in Variant A |
| failure_score | 0 | no failures in last 5 steps |
| repeated_tool | 0 | alternating names |
| latency_score | 0.010 | 20 ms x 0.5/1000 |
| cost_score | 0.500 | base_cost saturates at cumulative>=20; spend_per_success=1.0 -> waste=0 |
| long_memory_score | 0 | EWMA=0, no failures |
| **Floor** | **0.510** | well below DEGRADE=3.0 (default) or 7.5 (canonical) |

With single repeated tool name: repeated_tool=2, floor=**2.510** — still below 3.0.

**Canonical threshold set:** `EXP2_POLICY` in `experiments/experiment_2.py` is
`degrade_entropy=7.5, refuse_entropy=10.0`.
The comment in `run_experiment_5.py` ('degrade=9.0, refuse=11.0') is a documentation
error; the Python object resolves to 7.5/10.0. This experiment uses both
default (3.0/6.0) and canonical (7.5/10.0).

## §7 Results Table

Metric: detection rate across N=20 seeds.
A run is WRONG if distance_to_correct(terminal) > 0.5 with all steps succeeded.

| Variant | Policy | Mode | Detector | Fires in [b*,t_term]? | Detection Rate | Mean Detect Step |
|---|---|---|---|---|---|---|
| A | default_3_6 | baseline | entropy | **NO** | 0.000 | — |
| A | default_3_6 | baseline | circuit_breaker | **NO** | 0.000 | — |
| A | default_3_6 | baseline | coherence_cf | **NO** | 0.000 | — |
| A | default_3_6 | baseline | lambda_proxy | **NO** | 0.000 | — |
| A | default_3_6 | rnos | entropy | **NO** | 0.000 | — |
| A | default_3_6 | rnos | circuit_breaker | **NO** | 0.000 | — |
| A | default_3_6 | rnos | coherence_cf | **NO** | 0.000 | — |
| A | default_3_6 | rnos | lambda_proxy | **NO** | 0.000 | — |
| A | default_3_6 | cb | entropy | **NO** | 0.000 | — |
| A | default_3_6 | cb | circuit_breaker | **NO** | 0.000 | — |
| A | default_3_6 | cb | coherence_cf | **NO** | 0.000 | — |
| A | default_3_6 | cb | lambda_proxy | **NO** | 0.000 | — |
| A | default_3_6 | hybrid | entropy | **NO** | 0.000 | — |
| A | default_3_6 | hybrid | circuit_breaker | **NO** | 0.000 | — |
| A | default_3_6 | hybrid | coherence_cf | **NO** | 0.000 | — |
| A | default_3_6 | hybrid | lambda_proxy | **NO** | 0.000 | — |
| A | canonical_7p5_10 | baseline | entropy | **NO** | 0.000 | — |
| A | canonical_7p5_10 | baseline | circuit_breaker | **NO** | 0.000 | — |
| A | canonical_7p5_10 | baseline | coherence_cf | **NO** | 0.000 | — |
| A | canonical_7p5_10 | baseline | lambda_proxy | **NO** | 0.000 | — |
| A | canonical_7p5_10 | rnos | entropy | **NO** | 0.000 | — |
| A | canonical_7p5_10 | rnos | circuit_breaker | **NO** | 0.000 | — |
| A | canonical_7p5_10 | rnos | coherence_cf | **NO** | 0.000 | — |
| A | canonical_7p5_10 | rnos | lambda_proxy | **NO** | 0.000 | — |
| A | canonical_7p5_10 | cb | entropy | **NO** | 0.000 | — |
| A | canonical_7p5_10 | cb | circuit_breaker | **NO** | 0.000 | — |
| A | canonical_7p5_10 | cb | coherence_cf | **NO** | 0.000 | — |
| A | canonical_7p5_10 | cb | lambda_proxy | **NO** | 0.000 | — |
| A | canonical_7p5_10 | hybrid | entropy | **NO** | 0.000 | — |
| A | canonical_7p5_10 | hybrid | circuit_breaker | **NO** | 0.000 | — |
| A | canonical_7p5_10 | hybrid | coherence_cf | **NO** | 0.000 | — |
| A | canonical_7p5_10 | hybrid | lambda_proxy | **NO** | 0.000 | — |
| B | default_3_6 | baseline | entropy | **NO** | 0.000 | — |
| B | default_3_6 | baseline | circuit_breaker | **NO** | 0.000 | — |
| B | default_3_6 | baseline | coherence_cf | **NO** | 0.000 | — |
| B | default_3_6 | baseline | lambda_proxy | **NO** | 0.000 | — |
| B | default_3_6 | rnos | entropy | **NO** | 0.000 | — |
| B | default_3_6 | rnos | circuit_breaker | **NO** | 0.000 | — |
| B | default_3_6 | rnos | coherence_cf | **NO** | 0.000 | — |
| B | default_3_6 | rnos | lambda_proxy | **NO** | 0.000 | — |
| B | default_3_6 | cb | entropy | **NO** | 0.000 | — |
| B | default_3_6 | cb | circuit_breaker | **NO** | 0.000 | — |
| B | default_3_6 | cb | coherence_cf | **NO** | 0.000 | — |
| B | default_3_6 | cb | lambda_proxy | **NO** | 0.000 | — |
| B | default_3_6 | hybrid | entropy | **NO** | 0.000 | — |
| B | default_3_6 | hybrid | circuit_breaker | **NO** | 0.000 | — |
| B | default_3_6 | hybrid | coherence_cf | **NO** | 0.000 | — |
| B | default_3_6 | hybrid | lambda_proxy | **NO** | 0.000 | — |
| B | canonical_7p5_10 | baseline | entropy | **NO** | 0.000 | — |
| B | canonical_7p5_10 | baseline | circuit_breaker | **NO** | 0.000 | — |
| B | canonical_7p5_10 | baseline | coherence_cf | **NO** | 0.000 | — |
| B | canonical_7p5_10 | baseline | lambda_proxy | **NO** | 0.000 | — |
| B | canonical_7p5_10 | rnos | entropy | **NO** | 0.000 | — |
| B | canonical_7p5_10 | rnos | circuit_breaker | **NO** | 0.000 | — |
| B | canonical_7p5_10 | rnos | coherence_cf | **NO** | 0.000 | — |
| B | canonical_7p5_10 | rnos | lambda_proxy | **NO** | 0.000 | — |
| B | canonical_7p5_10 | cb | entropy | **NO** | 0.000 | — |
| B | canonical_7p5_10 | cb | circuit_breaker | **NO** | 0.000 | — |
| B | canonical_7p5_10 | cb | coherence_cf | **NO** | 0.000 | — |
| B | canonical_7p5_10 | cb | lambda_proxy | **NO** | 0.000 | — |
| B | canonical_7p5_10 | hybrid | entropy | **NO** | 0.000 | — |
| B | canonical_7p5_10 | hybrid | circuit_breaker | **NO** | 0.000 | — |
| B | canonical_7p5_10 | hybrid | coherence_cf | **NO** | 0.000 | — |
| B | canonical_7p5_10 | hybrid | lambda_proxy | **NO** | 0.000 | — |

## §6 Decision Criteria Evaluation

### H0: In Variant A (pure confident-wrong), do all detectors stay healthy?

- [default_3_6 / baseline] wrong_terminal_rate=1.000  entropy_rate=0.000  cb_rate=0.000  coherence_rate=0.000  -> H0 **CONFIRMED**
- [default_3_6 / rnos] wrong_terminal_rate=1.000  entropy_rate=0.000  cb_rate=0.000  coherence_rate=0.000  -> H0 **CONFIRMED**
- [default_3_6 / cb] wrong_terminal_rate=1.000  entropy_rate=0.000  cb_rate=0.000  coherence_rate=0.000  -> H0 **CONFIRMED**
- [default_3_6 / hybrid] wrong_terminal_rate=1.000  entropy_rate=0.000  cb_rate=0.000  coherence_rate=0.000  -> H0 **CONFIRMED**
- [canonical_7p5_10 / baseline] wrong_terminal_rate=1.000  entropy_rate=0.000  cb_rate=0.000  coherence_rate=0.000  -> H0 **CONFIRMED**
- [canonical_7p5_10 / rnos] wrong_terminal_rate=1.000  entropy_rate=0.000  cb_rate=0.000  coherence_rate=0.000  -> H0 **CONFIRMED**
- [canonical_7p5_10 / cb] wrong_terminal_rate=1.000  entropy_rate=0.000  cb_rate=0.000  coherence_rate=0.000  -> H0 **CONFIRMED**
- [canonical_7p5_10 / hybrid] wrong_terminal_rate=1.000  entropy_rate=0.000  cb_rate=0.000  coherence_rate=0.000  -> H0 **CONFIRMED**

### H1: In Variant B, does coherence fire before entropy?

- [default_3_6 / baseline] coherence NEVER fires (H1 cannot be tested — see §8 for structural reason)
- [default_3_6 / rnos] coherence NEVER fires (H1 cannot be tested — see §8 for structural reason)
- [default_3_6 / cb] coherence NEVER fires (H1 cannot be tested — see §8 for structural reason)
- [default_3_6 / hybrid] coherence NEVER fires (H1 cannot be tested — see §8 for structural reason)
- [canonical_7p5_10 / baseline] coherence NEVER fires (H1 cannot be tested — see §8 for structural reason)
- [canonical_7p5_10 / rnos] coherence NEVER fires (H1 cannot be tested — see §8 for structural reason)
- [canonical_7p5_10 / cb] coherence NEVER fires (H1 cannot be tested — see §8 for structural reason)
- [canonical_7p5_10 / hybrid] coherence NEVER fires (H1 cannot be tested — see §8 for structural reason)

## §8 Structural Finding: Coherent-Failure Signature Cannot Fire Without Tool Failures

`coherence.py:_find_coherent_failure_run` requires ALL of:
  1. `r_t >= 0.75` (high synchrony) — satisfied when tools execute and succeed
  2. `H_t` strictly rising between consecutive steps
  3. `consecutive_failures` strictly rising between consecutive steps
  4. `phase != 'stable'` in at least one of the two steps

In Variant B (friction without tool failures), `consecutive_failures = 0` throughout.
Therefore condition 3 can never be satisfied, and the signature never fires.

This is a structural finding, not a threshold choice: **the coherent-failure
signature as coded in `coherence.py` requires actual tool failures to accumulate
alongside high synchrony.** It detects 'execution proceeding coherently WHILE
failures accumulate' — not a run that fails silently with every step succeeding.

The lambda proxy (`hybrid.py`) is similarly blind in both variants:
- With all tools succeeding and RNOS allowing: `r_t = 1.0`, `h_t = 0`
  -> `lambda_proxy = 1.0` -> always RESONANT.

**Implication:** To detect confident-wrongness, a goal-progress signal is needed
(§8 constructive: `goal_divergence` as a 7th entropy term, or CEVAK probe).

## Sanity Check: Oracle Independence

`distance_to_correct` is computed from `env.distance_to_correct()` and logged
to CSV only. It is never passed to `ActionRecord`, `calculate_entropy`,
`calculate_trust`, `evaluate_policy`, `AdaptiveCircuitBreaker`, or
`HybridController`. The oracle is provably independent of all detector inputs.

## Piggyback: EWMA Effectiveness (Triage Item #5)

distributed_low_rate pattern (F-F-S repeating, 30 steps) with alpha 0/0.10/0.30:

| alpha | Final EWMA score (x2.0) | First step score >= 1.0 |
|---|---|---|
| alpha_0.00 | 0.0000 | never |
| alpha_0.10 | 1.2085 | 11 |
| alpha_0.30 | 1.0867 | 2 |

α=0.0: EWMA frozen at 0 — the long-memory signal is completely disabled.
α=0.10 (current): EWMA accumulates; reaches meaningful signal after many steps.
α=0.30 (fast): reaches >= 1.0 earlier; more responsive but higher false-alarm risk.

## Piggyback: Combo-REFUSE False-Positive Rate (Triage Item #8)

Scenario: 5 consecutive failures (burst) -> 25 successes (recovery). Hybrid controller.

- Recovery DEGRADE count (steps 6-30): 0
- Recovery REFUSE count (steps 6-30): 0
- False-positive REFUSE during recovery: **NO**

The combo-REFUSE rule does not fire REFUSE during legitimate recovery under this scenario. The post-merge hybrid does not introduce FP-REFUSE on the burst+recovery pattern.

## Constructive Next Steps (§8 from spec)

Since H0 holds — trace-internal detectors are all blind to confident-wrong runs:

1. **External progress signal as a 7th entropy term.** Add `goal_divergence`:
   sustained increase in an external/estimated `distance_to_correct` (or a
   self-reported progress estimate). Changes entropy from turbulence-only to
   turbulence-plus-direction. Cost: requires a goal model.

2. **CEVAK probe.** Run CEVAK's output-distribution view over the Variant A trace
   and test whether it flags drift where the execution-trace detectors are silent.
   If yes: 'execution-trace detection has a hard blind spot; CEVAK covers it.'
