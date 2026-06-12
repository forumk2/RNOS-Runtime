# Exp-6b ARM 2 — Predictions Addendum (timestamped, pre-qwen)

**Written:** 2026-06-11 21:02 local. At this moment the llama-3.2-3b run is
complete and summarized (`results/exp6b_arm2/interim_llama_summary.md`);
qwen3-4b has 185/4,850 judgments on disk and **none have been inspected**;
qwen3-coder-30b has not started. This addendum does not modify the frozen
prereg (`docs/exp6b_arm2_prereg.md`) — it adds finer-grained predictions and
two competing hypotheses so the cross-judge section of Phase D is
confirmatory rather than narrative. The fire rule, prompt, and P1–P3 are
untouched.

## §1 Context: what the llama surface showed

The llama-3.2-3b detection surface split cleanly along a line not
anticipated in the prereg:

- **Anomaly-relative detection** (a trace that approaches +10 and then turns
  or stalls): cheap, present at 3B — sudden onset-9/17 cells fire 0.90–1.00
  from delta=3 up.
- **Goal-absolute detection** (compare v to the stated goal with no
  reversal cue): absent at 3B — every onset-1 sudden cell fires 0.00,
  including delta=20, where the judge watched v sit at −10 and answered
  "reached and maintained the target value within the tolerance"
  (confabulated achievement).

## §2 Competing hypotheses for the onset-1 sudden failure

- **H-capability:** goal-absolute comparison is capability-gated; 3B cannot
  do it under any wording. Predicts the column unlocks as judge scale
  increases under the *identical frozen prompt*.
- **H-wording:** the prompt's word "advancing" invites relative reasoning
  ("is it moving?") and never forces a goal comparison; even capable judges
  will answer locally. Predicts the column stays at ~0 across all scales.

The frozen prompt is what makes these distinguishable at all: if 30B fixes
the column under the same prompt, capability was binding, not wording.
**Caveat declared now:** if the column stays flat across all three local
judges, H-wording and "channel is dead" remain confounded with
H-capability-at-frontier-scale; separating them would require a NEW
pre-registered run with a revised prompt (explicit goal-comparison
question). Prompt revision is a pre-registered follow-up only — never a
mid-run fix.

## §3 Predictions (frozen before any qwen data is inspected)

Numbered for Phase D scoring; each is per the prereg fire rule.

- **A1 (onset-1 unlock):** qwen3-coder-30b fires ≥70% on d20_o1_sudden.
  qwen3-4b is intermediate on that cell (>0%, <70%).
- **A2 (gradual stays weakest):** for every judge, pooled gradual fire rate
  < pooled sudden fire rate at matched delta/onset; gradual cells improve
  with scale but remain each judge's weakest shape.
- **A3 (delta=1 invisible):** no local judge reaches ≥70% on any delta=1
  cell.
- **A4 (FPR holds):** all judges keep FPR ≤5% on d0 (P2), including the
  thinking judge.
- **A5 (dead scalar generalizes):** |Spearman(progress, distance)| < 0.3
  for all judges — the progress scalar stays decorative at local scale;
  detection lives in the categorical bit.
- **A6 (conservative granularity generalizes):** granularity disagreements
  remain one-directional (per-step fires where every-3rd missed) for all
  judges, i.e. every-3rd-step fire rates are systematically conservative,
  never inflated.

External prediction, recorded verbatim for the same scoring (from a
frontier-model review of the interim summary, 2026-06-11): onset-1 sudden
unlocks with scale; gradual improves but stays weakest; delta=1 stays
near-invisible for all local judges. If 30B still scores ~0% on
d20_o1_sudden — wrong by a factor of 40 from step one with the goal in the
prompt — that is serious evidence against the channel, not just the judge.

## §4 Notes binding Phase D (analysis additions, all exploratory-labeled)

1. The anomaly-relative vs goal-absolute taxonomy (§1) goes in the Phase D
   writeup regardless of cross-judge outcome.
2. Reason-text categorization over all logged main-pass calls (keyword
   heuristic, deterministic, no LM calls): goal-referencing vs
   trajectory-referencing buckets, plus an objective confabulated-achievement
   flag (achievement-claim wording while true distance > 0.5 at that step).
   Reported as a per-cell confabulation-rate table per judge. Exploratory.
3. Granularity caveat: stated explicitly in Phase D — main-pass fire rates
   are systematically conservative estimates (per A6 direction), applied
   equally to every judge.
4. Design note for any future seventh entropy term: build on the categorical
   advancing bit with temporal integration (consecutive-no counting), not on
   the progress scalar, which is uncorrelated with ground truth at this
   scale.

## §5 Second timestamped prediction (mid-qwen, disclosed)

**Written:** 2026-06-11 21:39, after inspecting 539 qwen3-4b judgments
covering ONLY the d0 and d1 cells. Disclosed observations at this point:
qwen3-4b malformed rate 0% on d0, 33% on d1, flat across steps (22-23% at
every judged step 3-24), all malformed responses empty (thinking consumed
the full frozen 4,096-token budget). Mechanism inference: malformation is
**ambiguity-correlated, not length-correlated** — the judge ruminates past
its budget on subtly-wrong prefixes and answers cleanly on clearly-correct
ones.

- **A7 (non-monotone malformation):** qwen3-4b's malformed rate is
  non-monotone in delta — it falls on d8 and d20 cells (obvious wrongness →
  quick verdict) relative to d1/d3 (subtle wrongness → rumination
  overflow), i.e. malformation concentrates near the detection boundary,
  exactly where the channel is supposed to work. d3 ≥ d8 ≥ d20 in malformed
  rate, with d20 well below the d1 rate.

If A7 holds, the structured-bias caveat sharpens: the judge goes silent
precisely on the hard positive cases, so malformed→"uncertain" is maximally
non-innocent for this judge. Prediction frozen before any d3/d8/d20 qwen
judgment exists on disk.
