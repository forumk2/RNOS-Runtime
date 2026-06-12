# Exp-6b Phase 0: Feasibility Audit — CEVAK Bridge Over a Live Planner

**Date:** 2026-06-11  **Status:** STOP — interface gap found (see §4).
Phases 1 (pre-registration) and 2 (build/run) were **not** entered.

## §1 CEVAK ingestion interface

CEVAK has no ingestion point for logprobs or probability distributions of any
kind. Every entry path consumes either **discrete realized action labels** or
**pre-scored scalar floats**:

| Entry point | Signature | Consumes |
|---|---|---|
| `rnos_cevak/core/ade.py:75` | `ActionDistributionEntropy.record_action(action_type: str, task_class: str = "default")` | One realized action label per step |
| `rnos_cevak/core/ade.py:138` | `compute_ade(task_class) -> float` | Nothing external — KL(Q‖P) computed internally from Laplace-smoothed **categorical counts** of recorded labels (sliding window Q vs frozen baseline corpus P) |
| `rnos_cevak/core/cevak.py:153` | `CevakMonitor.record_calibration_vector(vec: CevakVector)` | Five pre-scored floats in [0,1] (C/E/V/A/K) |
| `rnos_cevak/core/cevak.py:197` | `CevakMonitor.evaluate(current: CevakVector, task_class) -> CevakResult` | Same five floats; the `ade` field is overwritten internally from the action window |
| `rnos_cevak/core/drift.py:71` | `classify_drift(*, dim_drifts: Sequence[float], ade_score: float, ...)` | Scalars only |
| `experiments/cevak_rnos_probe/cevak_eval.py:68` | `evaluate(trace: Trace) -> CevakResult` | `Step` objects: `action_type` label + five floats |
| `agent_runtime/cevak.py:67` | `compute_cevak(result: ExecutionResult, history)` | AST-similarity/success scalars (heuristic re-derivation; separate from `rnos_cevak` core) |

`grep -i logprob` over the repository returns **zero hits** (excluding the
Phase 0 smoke test added below). Token-level logprob vectors, top-k
distributions, and action-level distributions are not consumable anywhere.

## §2 Exp-6 frozen trace format

Confirmed: **no distributional data**, as expected.

- `results/experiment_6/exp6_{variant}_{mode}_{policy}_seed{NN}.csv` (320 files)
  — header: `step, mode, variant, policy_tag, seed, executed, success,
  distance_to_correct, v, entropy, trust, rnos_decision, cb_state, cb_reason,
  cb_failure_rate, r_t, H_t, Lambda_t, coherence_regime, lambda_proxy, phase,
  consecutive_failures, planner_latency_ms`. All scalar execution-trace
  signals.
- `logs/exp6_trace.jsonl` — per-step records:
  `{stage, tool, payload, depth, retry_count, metadata, entropy, trust,
  decision, reasons, constraints}` / `{stage, tool, ..., success}`. Tool name
  only; no distribution over alternatives.

A bridge therefore cannot be retrofitted onto frozen exp-6 traces; it would
require live planner runs (which is what exp-6b proposed).

## §3 LM Studio endpoint contract

Verified live on 2026-06-11. `scripts/exp6b_smoke_logprobs.py` POSTs to
`http://127.0.0.1:1234/v1/chat/completions` with `logprobs: true,
top_logprobs: 5` and asserts `choices[0].logprobs.content[0].top_logprobs`
has ≥ 2 entries. Result:

```
OK: logprobs present. ['A', 'I', ' A', 'Say', 'Please']
```

The endpoint honours the OpenAI-compatible logprobs contract. The endpoint is
**not** the bottleneck.

## §4 Verdict: STOP — three-layer gap

CEVAK's interface **cannot be fed from top-k logprobs without modification**.
Per the phase gate, the gap is reported here instead of being worked around.

**Layer 1 — No ingestion point.** ADE accepts only realized action labels
(§1). The only unmodified feeding path is to sample/argmax the planner's
action and call `record_action(label)` — which discards exactly the
distributional content the bridge exists to capture. There is no API that
accepts a per-step distribution, token-level or action-level.

**Layer 2 — Threshold semantics mismatch.** The "calibrated" numbers cited
for exp-6b (KL ceiling 2.0, mean 1.90) come from
`experiments/cevak_rnos_probe/README.md` (Results, bullet 2): KL of
1.87–2.06 nats (mean 1.90) on synthetic EVASION_PATTERN traces. That KL is
computed over **realized-action count distributions** with deliberately
disjoint vocabularies (baseline `{search, compute, plan}` vs runtime
`{fetch, write}`). It is not a logprob-derived quantity. Applying the
existing `ade_threshold=0.35` / `ade_max=2.0` to a KL computed from planner
logprob distributions would apply thresholds calibrated on a *different
random variable* — functionally equivalent to silent retuning, which exp-6b's
constraints forbid. The same holds for inventing a logprob→`CevakVector.confidence`
mapping: `dim_threshold=0.25` was calibrated on synthetic output scores, not
on any logprob-derived calibration signal.

**Layer 3 — Structural blindness on this environment.** The accumulator env's
action vocabulary is `{step_toward, add, scale}` (`tools/accumulator_env.py:35-36,100`),
and the exp-6 plan alternates `step_toward`/`add` in **both** correct and
confident-wrong runs — wrongness lives entirely in the tool *arguments*
(target/delta values), not in which tool is called. ADE's own docstring
(`rnos_cevak/core/ade.py:24-29`) states the limitation precisely: it
"detects shifts in *which* actions are taken, not *what* those actions do."
With identical realized-action mixes across arms, Q ≈ P, so ADE ≈ 0 in both
arms. The pre-registered criterion (ADE fires on ≥ 70% of arm-A runs) would
be structurally predetermined to fail — running it would produce a negative
result by construction, not by measurement. This is the same class of
structural finding as exp-6 §8 (coherence cannot fire without tool failures).

## §5 Options (decision required before any Phase 1)

None of these can be taken silently under exp-6b's constraints:

1. **Extend ADE with a distribution-ingestion API** (e.g.
   `record_action_distribution(probs: dict[str, float])` and a per-step
   KL against a baseline mixture). This modifies `rnos_cevak` core and
   requires fresh calibration of `ade_threshold`/`ade_max` on held-out data —
   i.e. a new pre-registration with an explicit calibration arm, not reuse of
   the 1.90/2.0 numbers.
2. **Re-scope exp-6b to realized-action ADE only** (no logprobs). Honest but,
   per Layer 3, near-certain negative by construction on this env; it answers
   a narrower question than the bridge was meant to test.
3. **Change the environment** so that confident-wrongness manifests as an
   action-mix shift (e.g. a wrong-goal planner that prefers `scale`). This
   tests CEVAK in its design regime but abandons exp-6's defining property
   (coherent failure indistinguishable at the action layer) — it would be a
   different experiment, not exp-6b.

## Files touched in Phase 0

- `scripts/exp6b_smoke_logprobs.py` (new, smoke test only)
- `docs/exp6b_phase0_feasibility.md` (this document)

No core code was modified. Phases 1 and 2 not entered.
