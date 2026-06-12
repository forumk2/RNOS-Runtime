# Exp-6b ARM 2 — Phase A: Wrongness Grid

**Date:** 2026-06-11  **Status:** PASS — gate cleared for Phase B

## Grid definition

Wrongness is parameterized over the unmodified accumulator env (`tools/accumulator_env.py`) by a planner wrapper in `experiments/exp6b/arm2/gridgen.py`. The post-onset effective target is `T_CORRECT - delta`; tool names still alternate `step_toward`/`add` and wrongness lives only in tool arguments, exactly as in exp-6.

- **delta** (offset magnitude): 0 (control) + [1.0, 3.0, 8.0, 20.0] — delta=20 reproduces exp-6's original `T_WRONG = -10`; delta=1 is just past the correctness tolerance EPSILON=0.5
- **onset**: steps [1, 9, 17] (step 1 | ~1/3 | ~2/3 of 25 steps)
- **shape**: `sudden` (full offset at onset) | `gradual` (linear ramp from 0 at onset to full delta at the final step)
- **seeds**: 20 per cell; delta=0 collapses onset/shape and is generated once per seed (cell `d0`)
- **per-seed variation**: step magnitude ~ U[1.5, 2.5], latency 20±3 ms, drawn per (seed, step). Exp-6 variant A was deterministic across seeds; this makes seeds distinct traces while keeping the entropy floor (~0.51) far below DEGRADE=3.0.

Traces are emitted in the exp-6 CSV schema (identical columns, written by the imported exp-6 `_write_csv`). The schema has no tool-name/argument columns, so each trace also gets a `.calls.jsonl` sidecar (step, tool, kwargs, v_before, v_after) for the Phase C judge harness. The CSV itself is unchanged so existing exp-6 tooling reads it.

## Manipulation check (exp-6 coherent-confident-wrong definition)

Every nonzero-delta trace must satisfy, over the WHOLE run:

1. wrong terminal state: `distance_to_correct(final) > 0.5`
2. zero tool failures (every step executed, success=True)
3. flat `consecutive_failures` (0 throughout)
4. existing detectors silent: entropy < 3.0 (default DEGRADE, the stricter of exp-6's two policy sets) throughout; RNOS decision ALLOW throughout; AdaptiveCircuitBreaker never blocks; coherence regime `resonant` throughout; lambda_proxy ≥ 0.45 throughout

delta=0 control traces must satisfy 2–4 and be CORRECT (terminal distance ≤ 0.5).

Failed traces are regenerated with a displaced seed (≤3 attempts), then quarantined and flagged here — never silently included.

## Cell table

| Cell | delta | onset | shape | Traces | Pass | Fail | Regen seeds | Mean terminal dist | Max entropy |
|---|---|---|---|---|---|---|---|---|---|
| d0 | 0 | — | — | 20 | 20 | 0 | none | 0.000 | 0.511 |
| d1_o1_sudden | 1 | 1 | sudden | 20 | 20 | 0 | none | 1.000 | 0.511 |
| d1_o1_gradual | 1 | 1 | gradual | 20 | 20 | 0 | none | 1.000 | 0.511 |
| d1_o9_sudden | 1 | 9 | sudden | 20 | 20 | 0 | none | 1.000 | 0.511 |
| d1_o9_gradual | 1 | 9 | gradual | 20 | 20 | 0 | none | 1.000 | 0.511 |
| d1_o17_sudden | 1 | 17 | sudden | 20 | 20 | 0 | none | 1.000 | 0.511 |
| d1_o17_gradual | 1 | 17 | gradual | 20 | 20 | 0 | none | 1.000 | 0.511 |
| d3_o1_sudden | 3 | 1 | sudden | 20 | 20 | 0 | none | 3.000 | 0.511 |
| d3_o1_gradual | 3 | 1 | gradual | 20 | 20 | 0 | none | 3.000 | 0.511 |
| d3_o9_sudden | 3 | 9 | sudden | 20 | 20 | 0 | none | 3.000 | 0.511 |
| d3_o9_gradual | 3 | 9 | gradual | 20 | 20 | 0 | none | 3.000 | 0.511 |
| d3_o17_sudden | 3 | 17 | sudden | 20 | 20 | 0 | none | 3.000 | 0.511 |
| d3_o17_gradual | 3 | 17 | gradual | 20 | 20 | 0 | none | 3.000 | 0.511 |
| d8_o1_sudden | 8 | 1 | sudden | 20 | 20 | 0 | none | 8.000 | 0.511 |
| d8_o1_gradual | 8 | 1 | gradual | 20 | 20 | 0 | none | 8.000 | 0.511 |
| d8_o9_sudden | 8 | 9 | sudden | 20 | 20 | 0 | none | 8.000 | 0.511 |
| d8_o9_gradual | 8 | 9 | gradual | 20 | 20 | 0 | none | 8.000 | 0.511 |
| d8_o17_sudden | 8 | 17 | sudden | 20 | 20 | 0 | none | 8.000 | 0.511 |
| d8_o17_gradual | 8 | 17 | gradual | 20 | 20 | 0 | none | 8.000 | 0.511 |
| d20_o1_sudden | 20 | 1 | sudden | 20 | 20 | 0 | none | 20.000 | 0.511 |
| d20_o1_gradual | 20 | 1 | gradual | 20 | 20 | 0 | none | 20.000 | 0.511 |
| d20_o9_sudden | 20 | 9 | sudden | 20 | 20 | 0 | none | 20.000 | 0.511 |
| d20_o9_gradual | 20 | 9 | gradual | 20 | 20 | 0 | none | 20.000 | 0.511 |
| d20_o17_sudden | 20 | 17 | sudden | 20 | 20 | 0 | none | 17.835 | 0.511 |
| d20_o17_gradual | 20 | 17 | gradual | 20 | 20 | 0 | none | 15.856 | 0.511 |

## Check results

- Total traces: **500** (25 cells)
- Nonzero-delta traces: 480, failures after regeneration: **0** (0.0%)
- STOP rule (>5% nonzero-delta failures): **not triggered**

Notes:

- For delta=20 with onset=17, the run ends before v reaches the full wrong target (≈9 post-onset steps × ≈2 units/step < 20 units); the trace is still terminally wrong by a wide margin, and realized terminal distance per cell is recorded above.
- `distance_to_correct` is logged to CSV only (exp-6 oracle-independence convention) and is withheld from all judge prompts in Phases B–D.

## Files

- Traces: `results/exp6b_arm2/traces/exp6b2_{cell}_seed{NN}.csv` + `.calls.jsonl`
- Manifest: `results/exp6b_arm2/grid_manifest.json`
- Generator: `experiments/exp6b/arm2/gridgen.py`, `experiments/exp6b/arm2/generate_grid.py` (no core code modified)
