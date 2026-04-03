# Experiment 5: Hybrid RNOS + Circuit Breaker

**Seed:** 42  **Max steps:** 30  **CB:** AdaptiveCircuitBreaker(window=10, threshold=0.6)  **Policy:** EXP2_POLICY (degrade=9.0, refuse=11.0)

## Threshold Configuration

The RNOS controller in this experiment uses calibrated thresholds from the RNOS-Runtime discrimination suite:

- **DEGRADE:** 9.0
- **REFUSE:** 11.0

These differ from the illustrative thresholds shown in the repository README (DEGRADE = 3.0, REFUSE = 6.0). The higher values are used here because ConfigurableAPI scenarios produce a structural entropy floor of approximately 4.0 — from `repeated_tool` (2.0) and `cost_score` (2.0, saturating after ~7 steps) — which is present in every run regardless of failure pattern. The illustrative thresholds would cause DEGRADE to fire on this floor alone, obscuring meaningful discrimination between scenarios. The calibrated thresholds place DEGRADE and REFUSE above the structural floor, so decisions reflect actual instability signals rather than baseline execution cost.

This is an experimental configuration choice and does not alter the structure of the RNOS policy.

## Results Table

Metric: tool_executions (actual API calls before termination).

```
-------------------------------------------------------------------------------------------------
Scenario               | BASELINE       | RNOS           | CB             | HYBRID         | Best
-------------------------------------------------------------------------------------------------
cascading_burst        | 30 exec        | 7 exec         | 10 exec        | 7 exec         | RNOS = HYBRID
distributed_low_rate   | 30 exec        | 30 exec        | 10 exec        | 10 exec        | CB = HYBRID
-------------------------------------------------------------------------------------------------
```

## Key Findings

### cascading_burst

- Baseline completed 30 executions (no control).
- RNOS stopped at 7 executions (first intervention: step 7, type=degrade, final_state=refused).
- CB stopped at 10 executions (first intervention: step 11, type=blocked, final_state=cb_blocked).
- Hybrid stopped at 7 executions (first intervention: step 7, trigger_source=rnos, final_state=refused).
- **Best:** RNOS = Hybrid (7 executions). Hybrid matches best.

### distributed_low_rate

- Baseline completed 30 executions (no control).
- RNOS stopped at 30 executions (first intervention: step None, type=None, final_state=completed).
- CB stopped at 10 executions (first intervention: step 11, type=blocked, final_state=cb_blocked).
- Hybrid stopped at 10 executions (first intervention: step 11, trigger_source=cb, final_state=refused).
- **Best:** CB = Hybrid (10 executions). Hybrid matches best.

## Mechanism

### cascading_burst

RNOS detects this scenario via **retry_score** accumulation: each consecutive failure increments retry_count (weight 1.0/step, cap 4.0). Combined with failure_score and the repeated_tool/cost floor, entropy crosses the DEGRADE threshold (9.0) before the CB's 10-step window fills.
RNOS first intervened at step 7 (entropy → refuse threshold at step 8). CB required 10 executions to fill its window (first block at step 11). Hybrid caught at step 7 (trigger: rnos).

### distributed_low_rate

CB detects this scenario via its **sliding window failure rate**: the F-F-S pattern produces 7/10 = 0.70 failures in any full 10-step window, exceeding the 0.60 threshold. RNOS's entropy stays below 9.0 because retry_count resets every third step (on the S step), keeping retry_score ≤ 2.0, and failure_score peaks at ~1.95 (3/5 recent).
RNOS ran all 30 steps without intervention. CB tripped at step 11 after 10 executions. Hybrid matched CB: step 11 (trigger: cb).

## Limitations

- **Hybrid never strictly dominates both sub-systems simultaneously.** In each scenario it matches the better-performing sub-system (RNOS for cascading_burst, CB for distributed_low_rate) but does not improve on it. The safety-first merge cannot extract information beyond what either sub-system independently detects.

- **Policy dependency.** Results use EXP2_POLICY (degrade=9.0, refuse=11.0) which is calibrated for the ConfigurableAPI structural floor (repeated_tool=2.0 + cost_score=2.0 ≈ 4.0 base entropy). A lower RNOS threshold would cause RNOS to flag the distributed scenario via the entropy floor alone, obscuring the CB's comparative advantage.

- **Deterministic scenarios.** Both scenarios use fully explicit step schedules. Real-world distributions would require stochastic robustness testing across many seeds before making strong architectural claims.

- **CB parameter sensitivity.** The CB window_size=10 and threshold=0.60 are tuned to produce clear differentiation in these scenarios. A smaller window (e.g., 5) would cause CB to trip earlier on cascading_burst, potentially matching RNOS and reducing the RNOS advantage.

## Per-step Data

Per-step CSV files written to `results/experiment_5/`. Each file is named `{scenario}_{mode}.csv` and contains: `step, executed, success, latency_ms, entropy, trust, rnos_decision, cb_state, cb_failure_rate, hybrid_decision, hybrid_trigger_source`.
