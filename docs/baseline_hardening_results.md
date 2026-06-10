# Baseline Hardening Results

Run date: 2026-06-04  |  N=20 seeds  |  V=1.00 USD  |  cost=0.010/1k tokens  |  avg_tokens=500

## Full suite — net_value (μ ± 95% CI)

| Scenario | Type | Mode | N | Loop(μ) | Fail(μ) | Survival% [CI] | FPR% [CI] | NetVal(μ) | CI± |
|----------|------|------|---|---------|---------|----------------|-----------|-----------|-----|
| benign | benign | Baseline | 20 | 20.0 | 0.9 | 100% [84,100] | 0% [0,16] | +0.9000 | ±0.0000 |
| benign | benign | RNOS | 20 | 20.0 | 0.9 | 100% [84,100] | 0% [0,16] | +0.9000 | ±0.0000 |
| benign | benign | CircuitBreaker | 20 | 20.0 | 0.9 | 100% [84,100] | 0% [0,16] | +0.9000 | ±0.0000 |
| benign | benign | Hybrid | 20 | 20.0 | 0.9 | 100% [84,100] | 0% [0,16] | +0.9000 | ±0.0000 |
| cascading_burst | failure | Baseline | 20 | 20.0 | 18.0 | 0% [0,16] | n/a | -0.1000 | ±0.0000 |
| cascading_burst | failure | RNOS | 20 | 7.0 | 4.0 | 0% [0,16] | n/a | -0.0350 | ±0.0000 |
| cascading_burst | failure | CircuitBreaker | 20 | 20.0 | 10.0 | 0% [0,16] | n/a | -0.1000 | ±0.0000 |
| cascading_burst | failure | Hybrid | 20 | 7.0 | 4.0 | 0% [0,16] | n/a | -0.0350 | ±0.0000 |
| distributed_low_rate | failure | Baseline | 20 | 20.0 | 14.0 | 0% [0,16] | n/a | -0.1000 | ±0.0000 |
| distributed_low_rate | failure | RNOS | 20 | 20.0 | 14.0 | 0% [0,16] | n/a | -0.1000 | ±0.0000 |
| distributed_low_rate | failure | CircuitBreaker | 20 | 20.0 | 8.0 | 0% [0,16] | n/a | -0.1000 | ±0.0000 |
| distributed_low_rate | failure | Hybrid | 20 | 9.0 | 6.0 | 0% [0,16] | n/a | -0.0450 | ±0.0000 |
| fanout_cascade | failure | Baseline | 20 | 20.0 | 13.0 | 0% [0,16] | n/a | -0.1000 | ±0.0000 |
| fanout_cascade | failure | RNOS | 20 | 10.0 | 4.0 | 0% [0,16] | n/a | -0.0500 | ±0.0000 |
| fanout_cascade | failure | CircuitBreaker | 20 | 20.0 | 8.0 | 0% [0,16] | n/a | -0.1000 | ±0.0000 |
| fanout_cascade | failure | Hybrid | 20 | 10.0 | 4.0 | 0% [0,16] | n/a | -0.0500 | ±0.0000 |
| recoverable_burst | recoverable | Baseline | 20 | 20.0 | 8.3 | 100% [84,100] | 0% [0,16] | +0.9000 | ±0.0000 |
| recoverable_burst | recoverable | RNOS | 20 | 7.0 | 4.0 | 0% [0,16] | 100% [84,100] | -0.0350 | ±0.0000 |
| recoverable_burst | recoverable | CircuitBreaker | 20 | 20.0 | 8.2 | 10% [3,30] | 90% [70,97] | -0.0000 | ±0.1349 |
| recoverable_burst | recoverable | Hybrid | 20 | 7.0 | 4.0 | 0% [0,16] | 100% [84,100] | -0.0350 | ±0.0000 |

## Dominance verdict


=== DOMINANCE VERDICT ===
  benign                  hyb=+0.9000  rnos=+0.9000  cb=+0.9000  [surv=100% fpr=0%]
  cascading_burst         hyb=-0.0350  rnos=-0.0350  cb=-0.1000
  distributed_low_rate    hyb=-0.0450  rnos=-0.1000  cb=-0.1000
  fanout_cascade          hyb=-0.0500  rnos=-0.0500  cb=-0.1000
  recoverable_burst       hyb=-0.0350  rnos=-0.0350  cb=-0.0000  [surv=0% fpr=100%]

  Hybrid STRICTLY dominates RNOS on: distributed_low_rate
  Hybrid LOSES to CB on: recoverable_burst

  FALSE POSITIVE NOTE (speed/safety frontier, not a bug):
    recoverable_burst FPR=100% — early window is indistinguishable from cascading failure.
    A controller cannot distinguish recoverable_burst from cascading_burst
    in the shared opening window; refusal is unavoidable if it catches the burst.


## V/c sensitivity

V=1.00, c=avg_tokens*cost_per_1k/1000 = 500*0.01/1000
Completing a recoverable run is net-positive when goal_step < V/c = 200.0 steps.
Rerun with `--task-value X` to sweep this ratio without changing thresholds.
