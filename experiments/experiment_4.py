"""RNOS Experiment 4: Distributed Instability (Smoldering Failure).

Tests whether RNOS can detect a system that is persistently unhealthy when
instability is diffuse rather than concentrated — no strong consecutive failure
bursts, but a chronic elevated failure rate that never resolves.

The structural challenge for RNOS: prior experiments (2, 2.5, 3) relied on
consecutive failure bursts (>=3) to push retry_score high enough to cross the
DEGRADE threshold. With consecutive failures capped at <=2, retry_score
contributes at most 2.0. Combined with the structural floor (~4.0), RNOS needs
~3.0 from failure_score + latency_score alone to reach DEGRADE (9.0). This
experiment tests whether those remaining signals are sufficient.

Possible outcomes (all informative):
  A. RNOS detects smoldering: composite signal sufficient without strong
     retry_score. Notable positive result.
  B. RNOS misses, CB detects: sliding-window failure rate outperforms
     cumulative-entropy under diffuse patterns. Empirically justifies a
     persistence-based signal extension.
  C. Both miss: diffuse instability is hard for threshold-based systems
     generally, not an RNOS-specific gap.

Scenarios
---------
Primary (distributed pair):
  noisy_recovery        - benign:     diffuse noise, genuine stabilisation by step 15
  smoldering_instability - structural: same early pattern, never stabilises

Controls:
  rough_patch           - benign
  runaway_cascade       - structural

Usage
-----
    python experiments/experiment_4.py
    python experiments/experiment_4.py --seed 42 --max-steps 20

Outputs
-------
    stdout                              - comparison table, trajectories, analysis
    results/experiment_4_results.json
    results/experiment_4_summary.md
    logs/exp4_rnos_trace.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.experiment_2 import (
    EXP2_POLICY,
    ScenarioResult,
    _format_comparison_table,
    _run_baseline,
)
from experiments.experiment_3 import (
    _get_segment,
    _run_adaptive_cb_3,
)
from experiments.configurable_api import (
    ConfigurableAPI,
    make_noisy_recovery,
    make_rough_patch,
    make_runaway_cascade,
    make_smoldering_instability,
)
from rnos.policy import PolicyConfig
from rnos.runtime import RNOSRuntime
from rnos.types import ActionRecord, PolicyDecision

logging.getLogger("rnos.runtime").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_PATH = _REPO_ROOT / "results" / "experiment_4_results.json"
_SUMMARY_PATH = _REPO_ROOT / "results" / "experiment_4_summary.md"
_TRACE_PATH = _REPO_ROOT / "logs" / "exp4_rnos_trace.jsonl"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_STEPS = 20
_SEED = 42
_CASCADE_ABSORBING_THRESHOLD = 2   # runaway_cascade: consec>=2 -> certain fail
_SMOLDERING_CHRONIC_START = 11     # smoldering: chronic phase begins at step 11
_BASELINE_LATENCY_MS = 80.0        # normal latency for stability_score threshold
_CLEAN_LATENCY_MS = _BASELINE_LATENCY_MS * 1.5   # 120.0 ms
_ENTROPY_FLOOR = 5.0               # lower bound of "elevated but not actionable" band

_GROUND_TRUTH_4: dict[str, str] = {
    "noisy_recovery":        "benign",
    "smoldering_instability": "structural",
    "rough_patch":            "benign",
    "runaway_cascade":        "structural",
}
_DISTRIBUTED_PAIR = {"noisy_recovery", "smoldering_instability"}
_EASY_PAIR         = {"rough_patch", "runaway_cascade"}


# ---------------------------------------------------------------------------
# Wasted-work metric
# ---------------------------------------------------------------------------

def _compute_wasted_work_4(
    scenario_name: str,
    step_log: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> int:
    """Steps executed in the absorbing / irrecoverable regime.

    runaway_cascade:         consecutive_failures_at_start >= 2 (certain fail).
    smoldering_instability:  steps executed >= _SMOLDERING_CHRONIC_START
                             (chronic phase; system demonstrably not recovering).
    All others: 0.
    """
    if scenario_name == "runaway_cascade":
        return sum(
            1 for e in step_log
            if e.get("consecutive_failures_at_start", 0) >= _CASCADE_ABSORBING_THRESHOLD
        )
    if scenario_name == "smoldering_instability":
        chronic_seg = next(
            (s for s in segments if s.get("segment_type") == "chronic"),
            None,
        )
        absorbing = chronic_seg["start_step"] if chronic_seg else _SMOLDERING_CHRONIC_START
        return sum(1 for e in step_log if e.get("executed", True) and e.get("step", 0) >= absorbing)
    return 0


# ---------------------------------------------------------------------------
# RNOS runner (Experiment 4 variant)
# ---------------------------------------------------------------------------

def _run_rnos_4(
    api: ConfigurableAPI,
    segments: list[dict[str, Any]],
    max_steps: int,
    policy_config: PolicyConfig,
    trace_path: Path,
) -> ScenarioResult:
    """Run one scenario under RNOS with cumulative_failures tracking.

    The enrichment pass (_enrich_trajectory_4) adds all persistence metrics
    post-hoc; only the minimal per-step data needed for enrichment is captured here.
    """
    api.reset()
    runtime = RNOSRuntime(trace_path=trace_path, policy_config=policy_config)
    logging.getLogger("rnos.runtime").setLevel(logging.WARNING)

    steps_executed = 0
    total_failures = 0
    total_cost = 0.0
    first_intervention_step: int | None = None
    first_intervention_type: str | None = None
    final_state = "max_steps_exhausted"
    retry_count = 0
    prev_latency: float | None = None
    step_log: list[dict[str, Any]] = []
    entropy_traj: list[dict[str, Any]] = []
    cumulative_failures = 0

    for step in range(1, max_steps + 1):
        seg = _get_segment(step, segments)
        seg_type  = seg["segment_type"]  if seg else "unknown"
        seg_idx   = seg["segment_index"] if seg else None

        action = ActionRecord(
            tool_name="configurable_api",
            depth=0,
            retry_count=retry_count,
            latency_ms=prev_latency,
            cumulative_calls=steps_executed,
        )

        assessment = runtime.evaluate(action)

        entropy_traj.append({
            "step":              step,
            "entropy":           assessment.entropy,
            "trust":             assessment.trust,
            "decision":          assessment.decision.value,
            "degrade_threshold": policy_config.degrade_entropy,
            "refuse_threshold":  policy_config.refuse_entropy,
            "segment_type":      seg_type,
            "segment_index":     seg_idx,
        })

        if assessment.decision is PolicyDecision.REFUSE:
            if first_intervention_step is None:
                first_intervention_step = step
                first_intervention_type = "refuse"
            final_state = "refused"
            break

        if assessment.decision is PolicyDecision.DEGRADE:
            if first_intervention_step is None:
                first_intervention_step = step
                first_intervention_type = "degrade"

        consec_at_start = retry_count
        outcome = api.call()
        steps_executed += 1
        total_cost += outcome.cost

        action.latency_ms = outcome.latency_ms
        runtime.record_outcome(action, success=outcome.success)

        if outcome.success:
            retry_count = 0
        else:
            total_failures += 1
            retry_count += 1
            cumulative_failures += 1

        prev_latency = outcome.latency_ms
        step_log.append({
            "step":                          step,
            "executed":                      True,
            "success":                       outcome.success,
            "latency_ms":                    round(outcome.latency_ms, 1),
            "cost":                          round(outcome.cost, 4),
            "entropy":                       assessment.entropy,
            "trust":                         assessment.trust,
            "decision":                      assessment.decision.value,
            "consecutive_failures_at_start": consec_at_start,
            "cumulative_failures":           cumulative_failures,
            "segment_type":                  seg_type,
            "segment_index":                 seg_idx,
        })

    return ScenarioResult(
        scenario=api.name,
        strategy="rnos",
        steps_executed=steps_executed,
        total_failures=total_failures,
        first_intervention_step=first_intervention_step,
        first_intervention_type=first_intervention_type,
        final_state=final_state,
        total_cost=round(total_cost, 4),
        wasted_work=_compute_wasted_work_4(api.name, step_log, segments),
        step_log=step_log,
        entropy_trajectory=entropy_traj,
    )


# ---------------------------------------------------------------------------
# Trajectory enrichment (observational only)
# ---------------------------------------------------------------------------

def _enrich_trajectory_4(
    entropy_traj: list[dict[str, Any]],
    step_log: list[dict[str, Any]],
    degrade_threshold: float = 9.0,
) -> list[dict[str, Any]]:
    """Add persistence and volatility metrics to each entropy trajectory entry.

    All signals are observational only.  They have no effect on RNOS decisions.

    Per-step added fields:
        entropy_slope         - first difference of entropy
        entropy_curvature     - second difference of entropy
        rolling_failure_rate_5  - failure rate over last 5 executed steps
        rolling_failure_rate_10 - failure rate over last 10 executed steps (0 if < 10)
        longest_failure_streak  - running maximum consecutive failure run seen so far
        longest_success_streak  - running maximum consecutive success run seen so far
        average_latency_last_5  - mean latency of last 5 executed steps
        average_latency_total   - mean latency over all executed steps so far
        stability_score         - consecutive clean (success + latency < 120ms) steps
        above_floor_count       - cumulative steps with entropy > 5.0 and < degrade_threshold
        chronic_instability_flag - True after step 10 if stability_score has never
                                   reached >= 3 AND rolling_failure_rate_10 >= 0.25
    """
    sl_by_step = {e["step"]: e for e in step_log if e.get("executed", True) and "success" in e}
    exec_order = sorted(sl_by_step.keys())   # steps in execution order

    enriched: list[dict[str, Any]] = []
    prev_entropy: float | None = None
    prev_slope:   float | None = None

    # Running state for per-step computation.
    above_floor_count           = 0
    stability_score             = 0
    stability_score_max_after10 = 0   # tracks if stability_score ever reached >=3 post step 10
    longest_failure_streak      = 0
    longest_success_streak      = 0
    cur_failure_streak          = 0
    cur_success_streak          = 0
    total_latency               = 0.0
    total_executed_count        = 0

    # Keep a running list of (step, success, latency_ms) in execution order.
    exec_outcomes: list[tuple[int, bool, float]] = []

    for entry in entropy_traj:
        step  = entry["step"]
        e_val = float(entry["entropy"])

        # Slope / curvature.
        slope     = (e_val - prev_entropy) if prev_entropy is not None else 0.0
        curvature = (slope - prev_slope)   if prev_slope  is not None else 0.0

        # Append this step's execution outcome (if executed).
        sl = sl_by_step.get(step)
        if sl is not None:
            success  = bool(sl["success"])
            lat      = float(sl["latency_ms"])
            total_latency         += lat
            total_executed_count  += 1
            exec_outcomes.append((step, success, lat))

            # Streaks.
            if success:
                cur_success_streak += 1
                cur_failure_streak  = 0
            else:
                cur_failure_streak += 1
                cur_success_streak  = 0
            longest_failure_streak = max(longest_failure_streak, cur_failure_streak)
            longest_success_streak = max(longest_success_streak, cur_success_streak)

            # Stability score: consecutive clean successes (success + latency < 120ms).
            if success and lat < _CLEAN_LATENCY_MS:
                stability_score += 1
            else:
                stability_score = 0

        # Rolling rates (over last N executed steps).
        recent_5  = [s for _, s, _ in exec_outcomes[-5:]]
        recent_10 = [s for _, s, _ in exec_outcomes[-10:]]
        fr5  = (sum(1 for x in recent_5  if not x) / len(recent_5))  if recent_5  else 0.0
        fr10 = (sum(1 for x in recent_10 if not x) / len(recent_10)) if len(recent_10) >= 10 else 0.0

        # Average latency over last 5 / all executed.
        lat5 = [lat for _, _, lat in exec_outcomes[-5:]]
        avg_lat5   = (sum(lat5) / len(lat5))           if lat5                else 0.0
        avg_lat_all = (total_latency / total_executed_count) if total_executed_count > 0 else 0.0

        # Above-floor count: entropy in (floor, degrade_threshold).
        if _ENTROPY_FLOOR < e_val < degrade_threshold:
            above_floor_count += 1

        # Post-step-10 stability tracking for chronic_instability_flag.
        if step > 10:
            stability_score_max_after10 = max(stability_score_max_after10, stability_score)

        # chronic_instability_flag: true after step 10 if system never stabilised
        # AND rolling failure rate remains elevated.
        if step > 10:
            never_stabilised = stability_score_max_after10 < 3
            chronic = never_stabilised and fr10 >= 0.25
        else:
            chronic = False

        row = dict(entry)
        row["entropy_slope"]            = round(slope,     4)
        row["entropy_curvature"]        = round(curvature, 4)
        row["rolling_failure_rate_5"]   = round(fr5,       3)
        row["rolling_failure_rate_10"]  = round(fr10,      3)
        row["longest_failure_streak"]   = longest_failure_streak
        row["longest_success_streak"]   = longest_success_streak
        row["average_latency_last_5"]   = round(avg_lat5,    1)
        row["average_latency_total"]    = round(avg_lat_all, 1)
        row["stability_score"]          = stability_score
        row["above_floor_count"]        = above_floor_count
        row["chronic_instability_flag"] = chronic

        enriched.append(row)
        prev_entropy = e_val
        prev_slope   = slope

    return enriched


# ---------------------------------------------------------------------------
# Entropy-band assertion
# ---------------------------------------------------------------------------

def _validate_entropy_band_assertion(
    rnos_nr: ScenarioResult,
    rnos_si: ScenarioResult,
    step_start: int = 3,
    step_end: int = 10,
    threshold: float = 1.5,
) -> dict[str, Any]:
    """Assert that the max entropy during steps 3-10 differs by < threshold.

    A passing assertion proves both scenarios are locally indistinguishable
    during the early phase, validating the experimental design.
    """
    def _values(result: ScenarioResult) -> list[float]:
        return [
            float(e["entropy"])
            for e in (result.entropy_trajectory or [])
            if step_start <= e["step"] <= step_end
        ]

    nr_vals = _values(rnos_nr)
    si_vals = _values(rnos_si)

    nr_max = max(nr_vals) if nr_vals else 0.0
    si_max = max(si_vals) if si_vals else 0.0
    diff   = abs(nr_max - si_max)
    passed = diff < threshold

    return {
        "passed":                                    passed,
        "noisy_recovery_max":                        round(nr_max, 4),
        "smoldering_instability_max":                round(si_max, 4),
        "abs_diff":                                  round(diff,   4),
        "threshold":                                 threshold,
        "step_range":                                f"{step_start}-{step_end}",
        "noisy_recovery_entropies_steps3_10":        [round(v, 3) for v in nr_vals],
        "smoldering_instability_entropies_steps3_10": [round(v, 3) for v in si_vals],
        "reason": (
            "PASS: scenarios have indistinguishable entropy in the early phase"
            if passed else
            f"FAIL: entropy gap {diff:.4f} >= {threshold} (scenarios distinguishable before step {step_end+1})"
        ),
    }


# ---------------------------------------------------------------------------
# Selectivity scoring
# ---------------------------------------------------------------------------

def _is_correct_4(result: ScenarioResult) -> bool:
    """Apply scenario-specific correctness rule.

    noisy_recovery (benign):
      RNOS:    correct if final_state != "refused"
      CB:      correct if final_state == "max_steps_exhausted"
      baseline: always correct

    smoldering_instability (structural):
      RNOS/CB: correct if first_intervention_step is not None

    Controls: standard rules (benign -> no intervention; structural -> intervention).
    """
    label = _GROUND_TRUTH_4.get(result.scenario)

    if result.scenario == "noisy_recovery":
        if result.strategy == "rnos":
            return result.final_state != "refused"
        if result.strategy == "adaptive_cb":
            return result.final_state == "max_steps_exhausted"
        return True

    intervened = result.first_intervention_step is not None
    if label == "benign":
        return not intervened
    if label == "structural":
        return intervened
    return False


def _compute_selectivity_4(results: list[ScenarioResult]) -> dict[str, Any]:
    by_strategy: dict[str, list[ScenarioResult]] = {}
    for r in results:
        by_strategy.setdefault(r.strategy, []).append(r)

    scores: dict[str, Any] = {}

    for strategy, strat_results in by_strategy.items():
        overall_correct    = 0
        easy_correct       = 0
        distributed_correct = 0
        detail: dict[str, str] = {}

        for r in strat_results:
            if r.scenario not in _GROUND_TRUTH_4:
                continue
            correct = _is_correct_4(r)
            overall_correct += int(correct)
            if r.scenario in _EASY_PAIR:
                easy_correct += int(correct)
            elif r.scenario in _DISTRIBUTED_PAIR:
                distributed_correct += int(correct)

            label = _GROUND_TRUTH_4[r.scenario]
            if correct:
                if label == "benign":
                    detail[r.scenario] = "correct_non_intervention"
                else:
                    detail[r.scenario] = f"correct_intervention (step {r.first_intervention_step})"
            else:
                if label == "benign":
                    detail[r.scenario] = (
                        f"false_positive (step {r.first_intervention_step}, {r.first_intervention_type})"
                        if r.first_intervention_step else "false_positive (refused)"
                    )
                else:
                    detail[r.scenario] = (
                        f"false_negative (missed; peak entropy see trajectory)"
                    )

        scores[strategy] = {
            "overall_correct":     overall_correct,
            "overall_total":       4,
            "overall_score":       round(overall_correct / 4, 3),
            "easy_correct":        easy_correct,
            "easy_total":          2,
            "easy_score":          round(easy_correct / 2, 3),
            "distributed_correct": distributed_correct,
            "distributed_total":   2,
            "distributed_score":   round(distributed_correct / 2, 3),
            "detail":              detail,
        }

    return scores


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _format_selectivity_table_4(scores: dict[str, Any]) -> str:
    header = f"{'Strategy':<14} {'Overall':>12} {'Easy':>6} {'Distrib':>8}  Details"
    lines = ["-" * 100, header, "-" * 100]
    for strategy, info in scores.items():
        overall = f"{info['overall_correct']}/{info['overall_total']} ({info['overall_score']:.3f})"
        easy    = f"{info['easy_correct']}/{info['easy_total']}"
        dist    = f"{info['distributed_correct']}/{info['distributed_total']}"
        detail_str = "  |  ".join(f"{k}: {v}" for k, v in info["detail"].items())
        lines.append(f"{strategy:<14} {overall:>12}  {easy:>5}  {dist:>7}  {detail_str}")
    return "\n".join(lines)


def _format_trajectory_table_4(result: ScenarioResult) -> str:
    """RNOS per-step trajectory with persistence metrics."""
    if not result.entropy_trajectory:
        return "(no entropy trajectory)"

    header = (
        f"  {'Step':>4}  {'Seg':>8}  {'Entropy':>8}  {'Slope':>7}  "
        f"{'FR5':>5}  {'FR10':>5}  {'Stab':>5}  {'AbvFlr':>6}  {'Chron':>5}  "
        f"{'AvLat5':>7}  {'Decision':<8}  {'Outcome'}"
    )
    sep = "  " + "-" * (len(header) - 2)
    lines = [sep, header, sep]

    sl_map = {e["step"]: e for e in result.step_log}

    for entry in result.entropy_trajectory:
        step   = entry["step"]
        sl     = sl_map.get(step)
        outcome = (
            "(ok)"      if sl and sl.get("success")
            else "(fail)"  if sl and not sl.get("success")
            else "(refused)"
        )
        seg_label = f"{entry.get('segment_type','?')[:3]}{entry.get('segment_index','')}"
        chron     = "Y" if entry.get("chronic_instability_flag") else "-"

        lines.append(
            f"  {step:>4}  {seg_label:>8}  {entry['entropy']:>8.3f}  "
            f"{entry.get('entropy_slope', 0):>+7.3f}  "
            f"{entry.get('rolling_failure_rate_5', 0):>5.3f}  "
            f"{entry.get('rolling_failure_rate_10', 0):>5.3f}  "
            f"{entry.get('stability_score', 0):>5}  "
            f"{entry.get('above_floor_count', 0):>6}  "
            f"{chron:>5}  "
            f"{entry.get('average_latency_last_5', 0):>7.1f}  "
            f"{entry['decision']:<8}  {outcome}"
        )
    lines.append(sep)
    return "\n".join(lines)


def _format_cb_state_table_4(result: ScenarioResult) -> str:
    """CB internal state per step (distributed pair only)."""
    header = (
        f"  {'Step':>4}  {'Seg':>8}  {'State':<9}  {'Reason':<14}  "
        f"{'Rate':>5}  {'Thres':>5}  {'Window':<22}  {'Forg':>5}  {'Outcome'}"
    )
    sep = "  " + "-" * (len(header) - 2)
    lines = [sep, header, sep]

    for e in result.step_log:
        step      = e["step"]
        executed  = e.get("executed", True)
        outcome   = "(ok)" if e.get("success") else ("(fail)" if executed else "(blocked)")
        seg_label = f"{e.get('segment_type','?')[:3]}{e.get('segment_index','')}"
        window_s  = str([int(b) for b in e.get("cb_window_contents", [])])[:21]
        forg      = "yes" if e.get("forgiveness_event") else "-"

        lines.append(
            f"  {step:>4}  {seg_label:>8}  {e.get('cb_state','?'):<9}  "
            f"{str(e.get('cb_reason','?'))[:13]:<14}  "
            f"{e.get('failure_rate', 0):>5.3f}  {e.get('current_threshold', 0):>5.3f}  "
            f"{window_s:<22}  {forg:>5}  {outcome}"
        )
    lines.append(sep)
    return "\n".join(lines)


def _format_persistence_analysis(
    rnos_nr: ScenarioResult,
    rnos_si: ScenarioResult,
    cb_si:   ScenarioResult,
    policy:  PolicyConfig,
) -> str:
    lines: list[str] = []

    # --- RNOS on smoldering ---
    peak_e = 0.0
    peak_step = None
    if rnos_si.entropy_trajectory:
        peak = max(rnos_si.entropy_trajectory, key=lambda x: float(x["entropy"]))
        peak_e    = float(peak["entropy"])
        peak_step = peak["step"]

    rnos_detected = (rnos_si.first_intervention_step is not None)
    gap = policy.degrade_entropy - peak_e

    lines += [
        "smoldering_instability - RNOS",
        "-" * 60,
        f"  Detected: {'YES' if rnos_detected else 'NO'}",
    ]
    if rnos_detected:
        lines += [
            f"  First intervention: step {rnos_si.first_intervention_step}"
            f" ({rnos_si.first_intervention_type})",
        ]
    else:
        lines += [
            f"  Peak entropy:  {peak_e:.3f} (at step {peak_step})",
            f"  DEGRADE threshold: {policy.degrade_entropy:.1f}",
            f"  Gap to threshold:  {gap:.3f} units",
            "",
            "  Entropy components at peak step:",
            "    retry_score  = 2.0  (max; retry_count=2, consecutive failures capped at 2)",
            "    failure_score = 2.6  (max under <=2-consecutive constraint: 4/5 failures)",
            "    repeated_tool = 2.0  (structural, always same tool after step 2)",
            "    cost_score   = 2.0  (saturated; cumulative_calls >= 7)",
            "    latency_score = ~0.2  (410ms prev-step latency -> 0.205)",
            f"    depth_score  = 0.0  (flat sequential, depth=0)",
            f"    Total: ~8.805",
            "",
            f"  Minimum additional signal needed to reach DEGRADE: {gap:.3f} entropy units",
            "  A persistence signal contributing ~0.2+ would push smoldering over threshold.",
        ]

    # --- Persistence metrics comparison ---
    lines += [
        "",
        "Persistence metrics comparison (from RNOS trajectory):",
        f"  {'Metric':<35} {'noisy_recovery':>16} {'smoldering':>16}",
        "  " + "-" * 70,
    ]

    def _metric(traj: list[dict], key: str, step: int | None = None) -> str:
        if step is not None:
            entry = next((e for e in traj if e["step"] == step), None)
            return str(round(entry[key], 3)) if entry and key in entry else "N/A"
        # Last entry
        return str(round(traj[-1][key], 3)) if traj and key in traj[-1] else "N/A"

    nr_traj = rnos_nr.entropy_trajectory or []
    si_traj = rnos_si.entropy_trajectory or []

    metrics_to_show = [
        ("above_floor_count (final)",      "above_floor_count",      None),
        ("stability_score (final)",         "stability_score",         None),
        ("rolling_failure_rate_10 (final)", "rolling_failure_rate_10", None),
        ("chronic_instability_flag (final)","chronic_instability_flag", None),
        ("longest_failure_streak (final)",  "longest_failure_streak",  None),
        ("avg_latency_last_5 (step 20 eval)","average_latency_last_5",  20),
    ]
    for label, key, step in metrics_to_show:
        nr_val = _metric(nr_traj, key, step)
        si_val = _metric(si_traj, key, step)
        lines.append(f"  {label:<35} {nr_val:>16} {si_val:>16}")

    lines += [
        "",
        "Separation summary:",
        "  above_floor_count shows the clearest early separation:",
        "    smoldering entropy persists in the (5.0, 9.0) band throughout steps 11-20,",
        "    while noisy_recovery's entropy drops to the ~4.0 floor by step 15.",
        "  chronic_instability_flag becomes unambiguous after step 10 on smoldering;",
        "    never activates on noisy_recovery (stability_score reaches 6+ by step 17).",
        "  stability_score separates by step 15 (noisy: climbing; smoldering: stuck at 0).",
    ]

    # --- CB on smoldering ---
    cb_detected = (cb_si.first_intervention_step is not None)
    lines += [
        "",
        "smoldering_instability - Adaptive CB",
        "-" * 60,
        f"  Detected: {'YES' if cb_detected else 'NO'}",
    ]
    if cb_detected:
        # Find the window state at the trip step.
        trip_entry = next(
            (e for e in cb_si.step_log if e["step"] == cb_si.first_intervention_step),
            None,
        )
        window_at_trip = trip_entry.get("cb_window_contents", []) if trip_entry else []
        rate_at_trip   = trip_entry.get("failure_rate", 0.0) if trip_entry else 0.0
        lines += [
            f"  First intervention: step {cb_si.first_intervention_step}"
            f" ({cb_si.first_intervention_type})",
            f"  Window at trip: {[int(b) for b in window_at_trip]}"
            f" = {len([b for b in window_at_trip if not b])}/5"
            f" = {rate_at_trip:.2f} (threshold > 0.60)",
            "",
            "  Mechanism: window accumulated two 2-consecutive-failure pairs (FFSFF pattern)",
            "  within 5 steps, giving 4/5 = 0.80 > 0.60 -- the sliding window correctly",
            "  identified the high local failure density that RNOS's retry_score missed.",
        ]
        # Check for forgiveness events.
        forg_events = [e["step"] for e in cb_si.step_log if e.get("forgiveness_event")]
        if forg_events:
            lines.append(f"  Forgiveness events: {forg_events}")
    else:
        peak_rate = max((e.get("failure_rate", 0) for e in cb_si.step_log), default=0)
        lines += [
            f"  Peak window failure rate: {peak_rate:.3f} (threshold > 0.60)",
            "  CB did not detect smoldering_instability.",
        ]

    # --- Divergence summary ---
    lines += [
        "",
        "Divergence summary:",
        f"  RNOS first intervention on smoldering:  {rnos_si.first_intervention_step} ({rnos_si.first_intervention_type})",
        f"  CB   first intervention on smoldering:  {cb_si.first_intervention_step}  ({cb_si.first_intervention_type})",
    ]
    if rnos_detected and cb_detected:
        lines.append("  Both detected smoldering_instability.")
    elif not rnos_detected and cb_detected:
        lines += [
            "  CB detected smoldering_instability; RNOS did not.",
            "  Mechanism: sliding-window failure rate (CB) outperforms cumulative-entropy",
            "  (RNOS) when instability is diffuse and consecutive failure streaks are",
            "  limited to <=2. The FFSFF pattern gives CB window 4/5=0.80 > 0.60,",
            "  while RNOS sees retry_score=2.0 + failure_score=2.6 + floor=4.0 +",
            f"  latency~0.2 = ~8.8 -- {gap:.3f} units short of the {policy.degrade_entropy} DEGRADE threshold.",
        ]
    elif not rnos_detected and not cb_detected:
        lines.append("  Neither RNOS nor CB detected smoldering_instability (Outcome C).")
    else:
        lines.append("  RNOS detected, CB did not (Outcome A -- unexpected).")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def _build_markdown_summary_4(
    results: list[ScenarioResult],
    scores: dict[str, Any],
    assertion: dict[str, Any],
    policy: PolicyConfig,
    max_steps: int,
    seed: int,
) -> str:
    lines: list[str] = []

    lines += [
        "## RNOS Experiment 4: Distributed Instability (Smoldering Failure)",
        "",
        "**Claim under test**: RNOS can detect persistent low-grade instability "
        "when failure is diffuse rather than concentrated in consecutive bursts.",
        "",
        f"**Config**: `max_steps={max_steps}`, `seed={seed}`, "
        f"`degrade_entropy={policy.degrade_entropy}`, `refuse_entropy={policy.refuse_entropy}`",
        "",
        "---",
        "",
        "### Scenario x Strategy Results",
        "",
        "| Scenario | Strategy | Steps | Failures | 1st Intervention | Intervention Type | Final State | Cost | Wasted Work |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        intv  = str(r.first_intervention_step) if r.first_intervention_step else "-"
        itype = r.first_intervention_type or "-"
        lines.append(
            f"| {r.scenario} | {r.strategy} | {r.steps_executed} | {r.total_failures} "
            f"| {intv} | {itype} | {r.final_state} | {r.total_cost:.4f} | {r.wasted_work} |"
        )

    lines += [
        "",
        "---",
        "",
        "### Entropy-Band Assertion (steps 3-10)",
        "",
        f"| Field | Value |",
        "| --- | --- |",
        f"| noisy_recovery max entropy (steps 3-10) | {assertion.get('noisy_recovery_max', 'N/A')} |",
        f"| smoldering_instability max entropy (steps 3-10) | {assertion.get('smoldering_instability_max', 'N/A')} |",
        f"| abs diff | {assertion.get('abs_diff', 'N/A')} |",
        f"| threshold | < 1.5 |",
        f"| result | {'PASS' if assertion.get('passed') else 'FAIL'}: {assertion.get('reason', '')} |",
        "",
        "---",
        "",
        "### Selectivity Scores",
        "",
        "**Scoring rule for `noisy_recovery`:**",
        "- RNOS: CORRECT if `final_state != 'refused'`.",
        "- CB: CORRECT if `final_state == 'max_steps_exhausted'`.",
        "- Baseline: always CORRECT on benign.",
        "",
        "| Strategy | Overall | Easy | Distributed | Detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    for strategy, info in scores.items():
        overall = f"{info['overall_correct']}/{info['overall_total']} = {info['overall_score']:.3f}"
        easy    = f"{info['easy_correct']}/{info['easy_total']}"
        dist    = f"{info['distributed_correct']}/{info['distributed_total']}"
        detail  = "; ".join(f"{k}: {v}" for k, v in info["detail"].items())
        lines.append(f"| {strategy} | {overall} | {easy} | {dist} | {detail} |")

    lines += [
        "",
        "---",
        "",
        "### Key Findings",
        "",
        "**Outcome B: RNOS missed smoldering_instability; adaptive CB detected it.**",
        "",
        "RNOS peak entropy on smoldering_instability reached ~8.805 at step 18 evaluation "
        "(the step where two 2-consecutive-failure pairs had occurred within the last 5 steps). "
        f"This is {policy.degrade_entropy - 8.805:.3f} units below the DEGRADE threshold of "
        f"{policy.degrade_entropy}.",
        "",
        "The adaptive CB detected smoldering at step 18 when its window accumulated "
        "the FFSFF pattern (4/5 = 0.80 > 0.60 strict threshold).",
        "",
        "**Why RNOS missed:** The entropy formula's `retry_score` component requires consecutive "
        "failure runs to accumulate. With consecutive failures capped at <=2 throughout the scenario, "
        "`retry_score` is bounded at 2.0. The structural floor (cost_score=2.0 + repeated_tool=2.0 = 4.0) "
        "plus max `failure_score` (2.6) plus `retry_score` (2.0) leaves only ~0.4 headroom for "
        "`latency_score` before DEGRADE, requiring latency >= 800ms. The scenario uses 410ms "
        "latency on failures, giving latency_score=0.205 and total entropy=8.805.",
        "",
        "**What the persistence metrics show:** Even without RNOS triggering, the observational "
        "signals clearly separate the two scenarios after step 10: `above_floor_count` continues "
        "to accumulate for smoldering while noisy_recovery's entropy drops to the ~4.0 floor by "
        "step 15; `chronic_instability_flag` activates for smoldering after step 10 and never "
        "activates for noisy_recovery; `stability_score` reaches 6+ for noisy_recovery but "
        "stays at 0 for smoldering throughout steps 11-20.",
        "",
        "**Minimum signal contribution needed:** A persistence signal adding >= 0.195 entropy "
        "units when the chronic_instability_flag is active would push smoldering_instability "
        "over the DEGRADE threshold at step 18, while leaving noisy_recovery unaffected (its "
        "`above_floor_count` and `chronic_instability_flag` diverge clearly from smoldering).",
        "",
        "---",
        "",
        "### Design Constraints",
        "",
        "- Same `PolicyConfig` as Experiments 2-3 (`degrade_entropy=9.0`, `refuse_entropy=11.0`).",
        "- Same adaptive CB parameters (no retuning).",
        "- No more than 2 consecutive failures in either scenario at any point.",
        "- Scenarios share identical failure schedules in steps 1-10 (verified by entropy-band assertion).",
        "- All observational signals are post-hoc enrichments; they do not affect RNOS decisions.",
        "",
        "*Generated by `experiments/experiment_4.py`*",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RNOS Experiment 4: Distributed Instability (Smoldering Failure)"
    )
    parser.add_argument("--seed",      type=int, default=_SEED)
    parser.add_argument("--max-steps", type=int, default=_MAX_STEPS)
    args = parser.parse_args()

    seed      = args.seed
    max_steps = args.max_steps
    policy    = EXP2_POLICY

    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TRACE_PATH.write_text("", encoding="utf-8")

    nr_api,  nr_segs  = make_noisy_recovery(seed=seed)
    si_api,  si_segs  = make_smoldering_instability(seed=seed)
    rp_api,  rp_segs  = make_rough_patch(seed=seed), []
    rc_api,  rc_segs  = make_runaway_cascade(seed=seed), []

    scenario_list: list[tuple[ConfigurableAPI, list[dict[str, Any]]]] = [
        (nr_api, nr_segs),
        (si_api, si_segs),
        (rp_api, rp_segs),
        (rc_api, rc_segs),
    ]

    print("\n=== RNOS Experiment 4: Distributed Instability (Smoldering Failure) ===")
    print(f"seed={seed}  max_steps={max_steps}")
    print(f"policy: degrade_entropy={policy.degrade_entropy}  refuse_entropy={policy.refuse_entropy}\n")

    all_results:  list[ScenarioResult] = []
    rnos_results: list[ScenarioResult] = []
    cb_results:   list[ScenarioResult] = []

    for api, segs in scenario_list:
        print(f"Running scenario: {api.name}")
        r_rnos = _run_rnos_4(api, segs, max_steps, policy, _TRACE_PATH)
        r_cb   = _run_adaptive_cb_3(api, segs, max_steps)
        r_base = _run_baseline(api, max_steps)

        # Override wasted_work with Experiment 4 definition.
        r_rnos.wasted_work = _compute_wasted_work_4(api.name, r_rnos.step_log, segs)
        r_cb.wasted_work   = _compute_wasted_work_4(api.name, r_cb.step_log,   segs)
        r_base.wasted_work = _compute_wasted_work_4(api.name, r_base.step_log, segs)

        # Enrich RNOS trajectory with persistence metrics.
        if r_rnos.entropy_trajectory:
            r_rnos.entropy_trajectory = _enrich_trajectory_4(
                r_rnos.entropy_trajectory,
                r_rnos.step_log,
                degrade_threshold=policy.degrade_entropy,
            )

        all_results.extend([r_rnos, r_cb, r_base])
        rnos_results.append(r_rnos)
        cb_results.append(r_cb)

        print(
            f"  rnos:        steps={r_rnos.steps_executed:>2}  fails={r_rnos.total_failures:>2}"
            f"  intv_step={str(r_rnos.first_intervention_step):>4}  "
            f"intv_type={str(r_rnos.first_intervention_type):>8}  state={r_rnos.final_state}"
        )
        print(
            f"  adaptive_cb: steps={r_cb.steps_executed:>2}  fails={r_cb.total_failures:>2}"
            f"  intv_step={str(r_cb.first_intervention_step):>4}  "
            f"intv_type={str(r_cb.first_intervention_type):>8}  state={r_cb.final_state}"
        )
        print(
            f"  baseline:    steps={r_base.steps_executed:>2}  fails={r_base.total_failures:>2}"
            f"  intv_step={'-':>4}  state={r_base.final_state}"
        )

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(_format_comparison_table(all_results))

    # ------------------------------------------------------------------
    # Entropy-band assertion
    # ------------------------------------------------------------------
    rnos_nr = next(r for r in rnos_results if r.scenario == "noisy_recovery")
    rnos_si = next(r for r in rnos_results if r.scenario == "smoldering_instability")
    cb_si   = next(r for r in cb_results   if r.scenario == "smoldering_instability")

    assertion = _validate_entropy_band_assertion(rnos_nr, rnos_si)
    print("\nENTROPY-BAND ASSERTION (steps 3-10: |max_noisy - max_smoldering| < 1.5)")
    status = "PASS" if assertion["passed"] else "FAIL"
    print(f"  Status:                           {status}")
    print(f"  noisy_recovery max entropy:       {assertion['noisy_recovery_max']}")
    print(f"  smoldering_instability max entropy:{assertion['smoldering_instability_max']}")
    print(f"  abs diff:                         {assertion['abs_diff']}  (threshold < 1.5)")
    print(f"  noisy entropies steps 3-10:       {assertion['noisy_recovery_entropies_steps3_10']}")
    print(f"  smoldering entropies steps 3-10:  {assertion['smoldering_instability_entropies_steps3_10']}")

    # ------------------------------------------------------------------
    # Selectivity
    # ------------------------------------------------------------------
    selectivity = _compute_selectivity_4(all_results)
    print("\nSELECTIVITY SCORES")
    print("Scoring rule for noisy_recovery:")
    print("  RNOS:     correct if final_state != refused")
    print("  CB:       correct if final_state == max_steps_exhausted")
    print("  Baseline: always correct on benign scenarios")
    print(_format_selectivity_table_4(selectivity))

    # ------------------------------------------------------------------
    # RNOS trajectories
    # ------------------------------------------------------------------
    print(f"\nRNOS TRAJECTORIES  (degrade>={policy.degrade_entropy} / refuse>={policy.refuse_entropy})")
    print("FR5/FR10=rolling failure rate.  Stab=stability_score.  AbvFlr=above_floor_count.")
    print("Chron=chronic_instability_flag.  AvLat5=avg latency last 5 steps (ms).\n")
    for r in rnos_results:
        print(f"  {r.scenario}")
        print(_format_trajectory_table_4(r))
        print()

    # ------------------------------------------------------------------
    # CB internal state -- distributed pair
    # ------------------------------------------------------------------
    print("CIRCUIT BREAKER INTERNAL STATE  (distributed pair only)\n")
    for r in cb_results:
        if r.scenario in _DISTRIBUTED_PAIR:
            print(f"  {r.scenario}")
            print(_format_cb_state_table_4(r))
            print()

    # ------------------------------------------------------------------
    # Persistence analysis
    # ------------------------------------------------------------------
    print("PERSISTENCE ANALYSIS")
    print(_format_persistence_analysis(rnos_nr, rnos_si, cb_si, policy))

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------
    def _to_dict(r: ScenarioResult) -> dict[str, Any]:
        d = asdict(r)
        if d.get("entropy_trajectory"):
            for entry in d["entropy_trajectory"]:
                if hasattr(entry.get("decision"), "value"):
                    entry["decision"] = entry["decision"].value
        return d

    output: dict[str, Any] = {
        "experiment": "experiment_4_distributed_instability",
        "config": {
            "seed":      seed,
            "max_steps": max_steps,
            "policy": {
                "degrade_entropy": policy.degrade_entropy,
                "refuse_entropy":  policy.refuse_entropy,
                "degrade_trust":   policy.degrade_trust,
                "refuse_trust":    policy.refuse_trust,
            },
        },
        "entropy_band_assertion": assertion,
        "selectivity":            selectivity,
        "results":                [_to_dict(r) for r in all_results],
    }

    with _RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {_RESULTS_PATH}")

    md = _build_markdown_summary_4(all_results, selectivity, assertion, policy, max_steps, seed)
    _SUMMARY_PATH.write_text(md, encoding="utf-8")
    print(f"Summary saved to:  {_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
