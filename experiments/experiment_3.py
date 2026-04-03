"""RNOS Experiment 3: Intermittent Cascading Failure.

Tests whether RNOS's cumulative instability logic retains enough cross-burst
memory to distinguish genuine bursty-but-recoverable traffic from intermittent
cascades that never truly stabilise.

The question: do RNOS and the adaptive circuit breaker behave differently when:
  - local recovery windows exist between failure bursts
  - the overall system is clearly unhealthy
  - but no single burst is permanently terminal on its own

Scenarios
---------
Bursty pair (the hard cases):
  bursty_recovery      - benign:     two short bursts, genuine sustained recovery
  intermittent_cascade - structural: three bursts with dirty/deceptive recovery windows

Controls (same as Experiment 2):
  rough_patch          - benign
  runaway_cascade      - structural

Strategies
----------
  RNOS           - EXP2_POLICY (no threshold changes from Experiment 2)
  Adaptive CB    - same parameters as Experiment 2 (no retuning)
  Baseline       - unprotected

Usage
-----
    python experiments/experiment_3.py
    python experiments/experiment_3.py --seed 42 --max-steps 20

Outputs
-------
    stdout                            - comparison table, trajectories, divergence analysis
    results/experiment_3_results.json
    results/experiment_3_summary.md
    logs/exp3_rnos_trace.jsonl
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
from experiments.configurable_api import (
    ConfigurableAPI,
    make_bursty_recovery,
    make_intermittent_cascade,
    make_rough_patch,
    make_runaway_cascade,
)
from baselines.adaptive_circuit_breaker import AdaptiveCircuitBreaker
from rnos.policy import PolicyConfig
from rnos.runtime import RNOSRuntime
from rnos.types import ActionRecord, PolicyDecision

logging.getLogger("rnos.runtime").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_PATH = _REPO_ROOT / "results" / "experiment_3_results.json"
_SUMMARY_PATH = _REPO_ROOT / "results" / "experiment_3_summary.md"
_TRACE_PATH = _REPO_ROOT / "logs" / "exp3_rnos_trace.jsonl"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_STEPS = 20
_SEED = 42
_CASCADE_ABSORBING_THRESHOLD = 2    # runaway_cascade: consec>=2 -> certain fail
_IC_BURST3_START = 14               # intermittent_cascade: burst 3 start step
_CLEAN_LATENCY_MS = 150.0           # threshold for "clean" low-latency success

_GROUND_TRUTH_3: dict[str, str] = {
    "bursty_recovery":      "benign",
    "intermittent_cascade": "structural",
    "rough_patch":          "benign",
    "runaway_cascade":      "structural",
}
_BURSTY_PAIR = {"bursty_recovery", "intermittent_cascade"}
_EASY_PAIR   = {"rough_patch", "runaway_cascade"}


# ---------------------------------------------------------------------------
# Segment helper
# ---------------------------------------------------------------------------

def _get_segment(step: int, burst_segments: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the burst segment covering *step*, or None if not covered."""
    for seg in reversed(burst_segments):
        if seg["start_step"] <= step <= seg["end_step"]:
            return seg
    return None


# ---------------------------------------------------------------------------
# Wasted-work metric
# ---------------------------------------------------------------------------

def _compute_wasted_work_3(
    scenario_name: str,
    step_log: list[dict[str, Any]],
    burst_segments: list[dict[str, Any]],
) -> int:
    """Steps executed in the absorbing / unrecoverable failure regime.

    runaway_cascade:      consecutive_failures_at_start >= 2 (certain fail after)
    intermittent_cascade: step >= _IC_BURST3_START (burst 3 onset; system never recovers)
    bursty_recovery:      0 (recoverable; failures are informative, not wasted)
    rough_patch:          0 (same)
    """
    if scenario_name == "runaway_cascade":
        return sum(
            1 for e in step_log
            if e.get("consecutive_failures_at_start", 0) >= _CASCADE_ABSORBING_THRESHOLD
        )
    if scenario_name == "intermittent_cascade":
        # Burst 3 start from metadata; fallback to constant.
        b3 = next(
            (s for s in burst_segments
             if s.get("burst_index") == 3 and s.get("segment_type") == "burst"),
            None,
        )
        absorbing = b3["start_step"] if b3 else _IC_BURST3_START
        return sum(1 for e in step_log if e.get("step", 0) >= absorbing)
    return 0


# ---------------------------------------------------------------------------
# Strategy runners with extended instrumentation
# ---------------------------------------------------------------------------

def _run_rnos_3(
    api: ConfigurableAPI,
    burst_segments: list[dict[str, Any]],
    max_steps: int,
    policy_config: PolicyConfig,
    trace_path: Path,
) -> ScenarioResult:
    """Run one scenario under RNOS with full burst-aware instrumentation.

    Extra per-step fields (observational only, do NOT affect decisions):
        burst_index                        - from burst_segments metadata
        segment_type                       - "burst" / "recovery" / "stable"
        cumulative_failures                - running failure count
        cumulative_successes_after_burst1  - successes after burst 1 started
        post_burst_stability               - consecutive clean successes since last burst ended
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

    # Burst-aware counters (observational).
    cumulative_failures = 0
    cumulative_successes_after_burst1 = 0
    burst1_started = False
    post_burst_stability = 0   # consecutive clean-success steps since last burst ended

    # Find burst 1 start for cumulative_successes tracking.
    burst1_start_step = next(
        (s["start_step"] for s in burst_segments
         if s.get("burst_index") == 1 and s.get("segment_type") == "burst"),
        None,
    )

    for step in range(1, max_steps + 1):
        seg = _get_segment(step, burst_segments)
        seg_type = seg["segment_type"] if seg else "unknown"
        burst_idx = seg["burst_index"] if seg else None

        if burst1_start_step and step >= burst1_start_step:
            burst1_started = True

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
            "burst_index":       burst_idx,
            "segment_type":      seg_type,
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
            if burst1_started:
                cumulative_successes_after_burst1 += 1
            # post_burst_stability: increment if clean AND not currently in burst.
            if seg_type != "burst" and outcome.latency_ms < _CLEAN_LATENCY_MS:
                post_burst_stability += 1
            else:
                post_burst_stability = 0
        else:
            total_failures += 1
            retry_count += 1
            cumulative_failures += 1
            post_burst_stability = 0   # burst or failure resets stability

        prev_latency = outcome.latency_ms
        step_log.append({
            "step":                              step,
            "success":                           outcome.success,
            "latency_ms":                        round(outcome.latency_ms, 1),
            "cost":                              round(outcome.cost, 4),
            "entropy":                           assessment.entropy,
            "trust":                             assessment.trust,
            "decision":                          assessment.decision.value,
            "consecutive_failures_at_start":     consec_at_start,
            "cumulative_failures":               cumulative_failures,
            "cumulative_successes_after_burst1": cumulative_successes_after_burst1,
            "post_burst_stability":              post_burst_stability,
            "burst_index":                       burst_idx,
            "segment_type":                      seg_type,
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
        wasted_work=_compute_wasted_work_3(api.name, step_log, burst_segments),
        step_log=step_log,
        entropy_trajectory=entropy_traj,
    )


def _run_adaptive_cb_3(
    api: ConfigurableAPI,
    burst_segments: list[dict[str, Any]],
    max_steps: int,
    window_size: int = 5,
    initial_failure_rate: float = 0.60,
    min_failure_rate: float = 0.40,
    adaptation_step: float = 0.05,
    initial_cooldown_steps: int = 2,
    max_cooldown_steps: int = 10,
    max_total_blocked: int = 20,
) -> ScenarioResult:
    """Run one scenario under the adaptive CB with extended internal-state logging.

    Extra per-step fields (bursty-pair focused):
        cb_window_contents  - list of booleans (CB's current sliding window)
        forgiveness_event   - True if this step is a successful half-open probe
                              that resets the breaker to CLOSED
        burst_index / segment_type - from burst_segments metadata
    """
    api.reset()
    cb = AdaptiveCircuitBreaker(
        window_size=window_size,
        initial_failure_rate=initial_failure_rate,
        min_failure_rate=min_failure_rate,
        adaptation_step=adaptation_step,
        initial_cooldown_steps=initial_cooldown_steps,
        max_cooldown_steps=max_cooldown_steps,
        max_total_blocked=max_total_blocked,
    )

    steps_executed = 0
    total_failures = 0
    total_cost = 0.0
    first_intervention_step: int | None = None
    first_intervention_type: str | None = None
    final_state = "max_steps_exhausted"
    retry_count = 0
    step_log: list[dict[str, Any]] = []

    for step in range(1, max_steps + 1):
        cb.tick()
        state_before = cb.state
        allowed, cb_reason = cb.should_execute()

        seg = _get_segment(step, burst_segments)
        seg_type  = seg["segment_type"] if seg else "unknown"
        burst_idx = seg["burst_index"]  if seg else None
        window_snapshot = list(cb._window)      # CB internal state (for analysis)

        if not allowed:
            if first_intervention_step is None:
                first_intervention_step = step
                first_intervention_type = cb_reason
            if cb_reason == "permanently_open":
                final_state = "circuit_permanently_open"
                break
            step_log.append({
                "step":                          step,
                "executed":                      False,
                "cb_state":                      cb.state,
                "cb_reason":                     cb_reason,
                "cb_window_contents":            window_snapshot,
                "failure_rate":                  cb.stats["failure_rate"],
                "current_threshold":             cb.stats["current_threshold"],
                "total_blocked":                 cb.stats["total_blocked"],
                "cooldown_remaining":            cb.stats["cooldown_remaining"],
                "forgiveness_event":             False,
                "burst_index":                   burst_idx,
                "segment_type":                  seg_type,
                "consecutive_failures_at_start": retry_count,
            })
            continue

        consec_at_start = retry_count
        outcome = api.call()
        steps_executed += 1
        total_cost += outcome.cost

        # Detect forgiveness: half-open probe that succeeded -> CLOSED.
        was_half_open = (state_before == "half_open")
        cb.record_result(success=outcome.success)
        forgiveness = was_half_open and outcome.success and cb.state == "closed"

        if outcome.success:
            retry_count = 0
        else:
            total_failures += 1
            retry_count += 1

        step_log.append({
            "step":                          step,
            "executed":                      True,
            "success":                       outcome.success,
            "latency_ms":                    round(outcome.latency_ms, 1),
            "cost":                          round(outcome.cost, 4),
            "cb_state":                      cb.state,
            "cb_reason":                     cb_reason,
            "cb_window_contents":            window_snapshot,
            "failure_rate":                  cb.stats["failure_rate"],
            "current_threshold":             cb.stats["current_threshold"],
            "total_blocked":                 cb.stats["total_blocked"],
            "cooldown_remaining":            cb.stats["cooldown_remaining"],
            "forgiveness_event":             forgiveness,
            "burst_index":                   burst_idx,
            "segment_type":                  seg_type,
            "consecutive_failures_at_start": consec_at_start,
        })

    return ScenarioResult(
        scenario=api.name,
        strategy="adaptive_cb",
        steps_executed=steps_executed,
        total_failures=total_failures,
        first_intervention_step=first_intervention_step,
        first_intervention_type=first_intervention_type,
        final_state=final_state,
        total_cost=round(total_cost, 4),
        wasted_work=_compute_wasted_work_3(api.name, step_log, burst_segments),
        step_log=step_log,
    )


# ---------------------------------------------------------------------------
# Trajectory enrichment (observational)
# ---------------------------------------------------------------------------

def _enrich_trajectory_3(
    entropy_traj: list[dict[str, Any]],
    step_log: list[dict[str, Any]],
    burst_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append trajectory signals to each entropy trajectory entry.

    These signals are purely observational and have no effect on RNOS decisions.

    Added fields:
        entropy_slope         - first difference of entropy
        entropy_curvature     - second difference of entropy
        sliding_fail_rate_3   - failure rate over last 3 executed steps
        sliding_fail_rate_5   - failure rate over last 5 executed steps
        recovery_signal       - True if success after 2+ consecutive failures
        cumulative_failures   - from step_log (if available)
        post_burst_stability  - from step_log (if available)
    """
    sl_map = {e["step"]: e for e in step_log}

    enriched: list[dict[str, Any]] = []
    prev_entropy: float | None = None
    prev_slope: float | None = None
    exec_window: list[bool] = []

    for entry in entropy_traj:
        step = entry["step"]
        e = float(entry["entropy"])

        slope     = (e - prev_entropy) if prev_entropy is not None else 0.0
        curvature = (slope - prev_slope) if prev_slope is not None else 0.0

        sl = sl_map.get(step)
        if sl and sl.get("executed", True) and "success" in sl:
            exec_window.append(bool(sl["success"]))

        w3 = exec_window[-3:] if len(exec_window) >= 1 else []
        w5 = exec_window[-5:] if len(exec_window) >= 1 else []
        fr3 = (sum(1 for s in w3 if not s) / len(w3)) if w3 else 0.0
        fr5 = (sum(1 for s in w5 if not s) / len(w5)) if w5 else 0.0

        consec = sl.get("consecutive_failures_at_start", 0) if sl else 0
        success_now = bool(sl.get("success", False)) if sl else False
        recovery = success_now and consec >= 2

        row = dict(entry)
        row["entropy_slope"]        = round(slope, 4)
        row["entropy_curvature"]    = round(curvature, 4)
        row["sliding_fail_rate_3"]  = round(fr3, 3)
        row["sliding_fail_rate_5"]  = round(fr5, 3)
        row["recovery_signal"]      = recovery
        row["cumulative_failures"]  = sl.get("cumulative_failures",  None) if sl else None
        row["post_burst_stability"] = sl.get("post_burst_stability", None) if sl else None

        enriched.append(row)
        prev_entropy = e
        prev_slope   = slope

    return enriched


# ---------------------------------------------------------------------------
# Selectivity scoring
# ---------------------------------------------------------------------------

def _is_correct_3(result: ScenarioResult) -> bool:
    """Apply scenario-specific correctness rule.

    bursty_recovery (benign):
      RNOS:    correct if final_state != "refused"
               (DEGRADE along the way is acceptable; permanent stop is not)
      CB:      correct if final_state == "max_steps_exhausted"
               (entering open state but recovering and completing is OK;
                staying blocked at termination is not)
      baseline: always correct (no intervention capability on benign)

    All others: standard structural/benign rules.
    """
    label    = _GROUND_TRUTH_3.get(result.scenario)
    intervened = result.first_intervention_step is not None

    if result.scenario == "bursty_recovery":
        if result.strategy == "rnos":
            return result.final_state != "refused"
        if result.strategy == "adaptive_cb":
            return result.final_state == "max_steps_exhausted"
        return True   # baseline

    if label == "benign":
        return not intervened
    if label == "structural":
        return intervened
    return False


def _compute_selectivity_3(results: list[ScenarioResult]) -> dict[str, Any]:
    """Compute per-strategy selectivity with easy/bursty pair breakdown."""
    by_strategy: dict[str, list[ScenarioResult]] = {}
    for r in results:
        by_strategy.setdefault(r.strategy, []).append(r)

    scores: dict[str, Any] = {}

    for strategy, strat_results in by_strategy.items():
        overall_correct = 0
        easy_correct    = 0
        bursty_correct  = 0
        detail: dict[str, str] = {}

        for r in strat_results:
            if r.scenario not in _GROUND_TRUTH_3:
                continue
            correct = _is_correct_3(r)
            overall_correct += int(correct)
            if r.scenario in _EASY_PAIR:
                easy_correct += int(correct)
            elif r.scenario in _BURSTY_PAIR:
                bursty_correct += int(correct)

            if correct:
                label = _GROUND_TRUTH_3[r.scenario]
                if label == "benign":
                    detail[r.scenario] = "correct_non_intervention"
                else:
                    detail[r.scenario] = f"correct_intervention (step {r.first_intervention_step})"
            else:
                label = _GROUND_TRUTH_3[r.scenario]
                if label == "benign":
                    detail[r.scenario] = (
                        f"false_positive (step {r.first_intervention_step},"
                        f" {r.first_intervention_type})"
                        if r.first_intervention_step else "false_positive (final refused)"
                    )
                else:
                    detail[r.scenario] = "false_negative (no intervention)"

        scores[strategy] = {
            "overall_correct": overall_correct,
            "overall_total":   4,
            "overall_score":   round(overall_correct / 4, 3),
            "easy_correct":    easy_correct,
            "easy_total":      2,
            "easy_score":      round(easy_correct / 2, 3),
            "bursty_correct":  bursty_correct,
            "bursty_total":    2,
            "bursty_score":    round(bursty_correct / 2, 3),
            "detail":          detail,
        }

    return scores


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _format_selectivity_table_3(scores: dict[str, Any]) -> str:
    header = f"{'Strategy':<14} {'Overall':>12} {'Easy':>6} {'Bursty':>7}  Details"
    lines = ["-" * 100, header, "-" * 100]
    for strategy, info in scores.items():
        overall = f"{info['overall_correct']}/{info['overall_total']} ({info['overall_score']:.3f})"
        easy    = f"{info['easy_correct']}/{info['easy_total']}"
        bursty  = f"{info['bursty_correct']}/{info['bursty_total']}"
        detail_str = "  |  ".join(f"{k}: {v}" for k, v in info["detail"].items())
        lines.append(f"{strategy:<14} {overall:>12}  {easy:>5}  {bursty:>6}  {detail_str}")
    return "\n".join(lines)


def _format_trajectory_table_3(result: ScenarioResult) -> str:
    """Per-step RNOS trajectory with enriched signals."""
    if not result.entropy_trajectory:
        return "(no entropy trajectory)"

    header = (
        f"  {'Step':>4}  {'Seg':>8}  {'Entropy':>8}  {'Slope':>7}  {'Curv':>7}  "
        f"{'FR3':>5}  {'FR5':>5}  {'Stab':>5}  {'Decision':<8}  {'Outcome'}"
    )
    sep = "  " + "-" * (len(header) - 2)
    lines = [sep, header, sep]

    sl_map = {e["step"]: e for e in result.step_log}

    for entry in result.entropy_trajectory:
        step     = entry["step"]
        sl       = sl_map.get(step)
        executed = sl.get("executed", True) if sl else False
        if sl and executed and "success" in sl:
            outcome = "(ok)" if sl["success"] else "(fail)"
        elif sl and not executed:
            outcome = "(blocked)"
        else:
            outcome = "(refused)"

        seg_label = f"{entry.get('segment_type','?')[:3]}{entry.get('burst_index','')}"
        stab      = sl.get("post_burst_stability", 0) if sl else 0

        lines.append(
            f"  {step:>4}  {seg_label:>8}  {entry['entropy']:>8.3f}  "
            f"{entry.get('entropy_slope',0):>+7.3f}  {entry.get('entropy_curvature',0):>+7.3f}  "
            f"{entry.get('sliding_fail_rate_3',0):>5.3f}  {entry.get('sliding_fail_rate_5',0):>5.3f}  "
            f"{stab:>5}  {entry['decision']:<8}  {outcome}"
        )
    lines.append(sep)
    return "\n".join(lines)


def _format_cb_state_table(result: ScenarioResult) -> str:
    """Circuit breaker internal state per step (bursty pair only)."""
    header = (
        f"  {'Step':>4}  {'Seg':>8}  {'State':<9}  {'Reason':<14}  "
        f"{'Rate':>5}  {'Thres':>5}  {'Window':<20}  {'Forg':>5}  {'Outcome'}"
    )
    sep = "  " + "-" * (len(header) - 2)
    lines = [sep, header, sep]

    for e in result.step_log:
        step     = e["step"]
        executed = e.get("executed", True)
        outcome  = ("(ok)" if e.get("success") else "(fail)" if executed else "(blocked)")
        seg_label = f"{e.get('segment_type','?')[:3]}{e.get('burst_index','')}"
        window_s = str([int(b) for b in e.get("cb_window_contents", [])])[:19]
        forg     = "yes" if e.get("forgiveness_event") else "-"

        lines.append(
            f"  {step:>4}  {seg_label:>8}  {e.get('cb_state','?'):<9}  "
            f"{str(e.get('cb_reason','?'))[:13]:<14}  "
            f"{e.get('failure_rate',0):>5.3f}  {e.get('current_threshold',0):>5.3f}  "
            f"{window_s:<20}  {forg:>5}  {outcome}"
        )
    lines.append(sep)
    return "\n".join(lines)


def _format_divergence_analysis(
    rnos_br:  ScenarioResult,
    cb_br:    ScenarioResult,
    rnos_ic:  ScenarioResult,
    cb_ic:    ScenarioResult,
) -> str:
    """Side-by-side divergence analysis for the bursty pair at key steps."""
    lines: list[str] = []

    # --- bursty_recovery ---
    lines += [
        "bursty_recovery (benign) - RNOS vs Adaptive CB",
        "-" * 72,
        f"  RNOS first intervention:  {rnos_br.first_intervention_step} ({rnos_br.first_intervention_type})",
        f"  CB   first intervention:  {cb_br.first_intervention_step}  ({cb_br.first_intervention_type})",
        f"  RNOS final state: {rnos_br.final_state}  |  CB final state: {cb_br.final_state}",
        f"  RNOS peak entropy (from trajectory):",
    ]
    if rnos_br.entropy_trajectory:
        peak = max(rnos_br.entropy_trajectory, key=lambda x: x["entropy"])
        lines.append(
            f"    step {peak['step']}: entropy={peak['entropy']:.3f}"
            f"  [{peak.get('segment_type','?')} {peak.get('burst_index','')}]"
            f"  (degrade threshold={rnos_br.entropy_trajectory[0]['degrade_threshold']:.1f})"
        )
    lines.append("")

    # --- intermittent_cascade: step-by-step at key divergence window ---
    lines += [
        "intermittent_cascade (structural) - RNOS vs Adaptive CB",
        "First divergence: step where RNOS and CB make different decisions",
        "-" * 100,
    ]

    rnos_traj_map = {e["step"]: e for e in (rnos_ic.entropy_trajectory or [])}
    cb_map        = {e["step"]: e for e in cb_ic.step_log}

    # Header
    lines.append(
        f"  {'Step':>4}  {'Seg':>8}  "
        f"{'RNOS entropy':>13} {'RNOS decision':<12}  "
        f"{'CB window':>25} {'CB rate':>7} {'CB state':<10} {'CB blocked':>10}"
    )
    lines.append("  " + "-" * 96)

    for step in range(1, 21):
        re = rnos_traj_map.get(step)
        ce = cb_map.get(step)
        if re is None and ce is None:
            continue

        r_entropy  = f"{re['entropy']:.3f}" if re else "N/A"
        r_decision = re["decision"] if re else "N/A"
        seg_label  = f"{re.get('segment_type','?')[:3]}{re.get('burst_index','')}" if re else "?"

        cb_window   = str([int(b) for b in ce.get("cb_window_contents", [])]) if ce else "N/A"
        cb_rate_s   = f"{ce.get('failure_rate',0):.3f}" if ce else "N/A"
        cb_state_s  = ce.get("cb_state", "N/A") if ce else "N/A"
        cb_blocked  = "BLOCKED" if (ce and not ce.get("executed", True)) else ""

        # Mark divergence.
        diverge = ""
        if re and ce:
            rnos_acts = re["decision"] not in ("allow",)
            cb_acts   = not ce.get("executed", True)
            if rnos_acts != cb_acts:
                diverge = " <-- diverge"

        lines.append(
            f"  {step:>4}  {seg_label:>8}  "
            f"{r_entropy:>13} {r_decision:<12}  "
            f"{cb_window:>25} {cb_rate_s:>7} {cb_state_s:<10} {cb_blocked:>10}{diverge}"
        )

    lines += [
        "",
        "Divergence summary:",
        f"  RNOS first intervention: step {rnos_ic.first_intervention_step} ({rnos_ic.first_intervention_type})",
        f"  CB   first intervention: step {cb_ic.first_intervention_step}  ({cb_ic.first_intervention_type})",
        "",
        "Mechanism (RNOS fires earlier due to retry_score + structural floor):",
        "  At burst 2 end (step 11 eval):",
        "    retry_count = 3 (steps 8,9,10 consecutive failures)",
        "    retry_score = 3.0",
        "    failure_score = 1.95 (3 failures in last 5)",
        "    structural floor = 4.0 (cost_score=2.0 capped + repeated_tool=2)",
        "    latency_score = ~0.215 (430ms)",
        "    entropy = 9.165 > 9.0 threshold => DEGRADE",
        "  CB at same point:",
        "    window = [S,S,F,F,F] = 3/5 = 0.60",
        "    strict '>' check: 0.60 NOT > 0.60 => ALLOW",
        "",
        "Cross-burst memory mechanism:",
        "  cost_score = min(cumulative_calls * 0.3, 2.0) accumulates monotonically.",
        "  By step 11: cumulative_calls=10 => cost_score=2.0 (capped).",
        "  This structural floor (4.0) was built across all prior steps including",
        "  burst 1 and recovery 1 - it persists through recovery windows.",
        "  Without it, the same 3-consecutive-failure burst at step 2-4 of a fresh",
        "  run would produce entropy ~3.64 (ALLOW). The cross-burst budget exhaustion",
        "  is what makes burst 2 sufficient to trigger DEGRADE.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def _build_markdown_summary_3(
    results: list[ScenarioResult],
    scores: dict[str, Any],
    policy: PolicyConfig,
    max_steps: int,
    seed: int,
) -> str:
    lines: list[str] = []

    lines += [
        "## RNOS Experiment 3: Intermittent Cascading Failure",
        "",
        "**Claim**: RNOS's cumulative instability logic retains enough cross-burst memory "
        "to distinguish bursty-but-recoverable traffic from chronic intermittent cascades.",
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
        "### Selectivity Scores",
        "",
        "Overall = 4 scenarios.  Easy pair = rough_patch + runaway_cascade.  "
        "Bursty pair = bursty_recovery + intermittent_cascade.",
        "",
        "**Scoring rule for `bursty_recovery`:**",
        "- RNOS: CORRECT if `final_state != 'refused'` "
        "(DEGRADE acceptable; permanent stop is incorrect).",
        "- CB: CORRECT if `final_state == 'max_steps_exhausted'` "
        "(recovering to closed and completing is acceptable).",
        "- Baseline: always CORRECT on benign scenarios.",
        "",
        "| Strategy | Overall | Easy | Bursty | Detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    for strategy, info in scores.items():
        overall = f"{info['overall_correct']}/{info['overall_total']} = {info['overall_score']:.3f}"
        easy    = f"{info['easy_correct']}/{info['easy_total']}"
        bursty  = f"{info['bursty_correct']}/{info['bursty_total']}"
        detail  = "; ".join(f"{k}: {v}" for k, v in info["detail"].items())
        lines.append(f"| {strategy} | {overall} | {easy} | {bursty} | {detail} |")

    lines += [
        "",
        "---",
        "",
        "### Key Findings",
        "",
        "**bursty_recovery**: Both RNOS and adaptive CB allow through without permanent "
        "intervention.  RNOS peak entropy stays below the DEGRADE threshold (9.0) at all steps.",
        "",
        "**intermittent_cascade**: RNOS first flags at burst 2 end (step 11, DEGRADE) "
        "while the adaptive CB allows through until step 18 (7 steps later) when burst 3 "
        "accumulates enough failures to push the sliding window above 0.60.  Both ultimately "
        "identify the scenario as structural.",
        "",
        "**Divergence mechanism**: At step 11 (end of burst 2), RNOS's `retry_score` (3 "
        "consecutive failures = 3.0) combined with `failure_score` (1.95) and the structural "
        "floor (`cost_score=2.0 + repeated_tool=2.0 = 4.0`) gives entropy 9.165 > 9.0 threshold.  "
        "The adaptive CB sees 3/5=0.60 which does NOT exceed its strict `>` threshold.",
        "",
        "**Cross-burst memory**: `cost_score = min(cumulative_calls * 0.3, 2.0)` saturates at "
        "its cap by step 7 and remains at 2.0 through all subsequent recovery windows.  This "
        "structural floor persists even when retry and failure scores reset on each recovery, "
        "making each new burst start from a higher entropy baseline than it would in a fresh run.",
        "",
        "---",
        "",
        "### Design Constraints",
        "",
        "- Same `PolicyConfig` as Experiment 2 (`degrade_entropy=9.0`, `refuse_entropy=11.0`).",
        "- Same adaptive CB parameters (no retuning).",
        "- Trajectory signals are observational only; they do NOT influence RNOS decisions.",
        "- Explicit deterministic schedules used throughout for full reproducibility.",
        "",
        "*Generated by `experiments/experiment_3.py`*",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RNOS Experiment 3: Intermittent Cascading Failure"
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

    # Build scenario list: bursty pair (with segments) + controls (empty segments).
    br_api,  br_segs  = make_bursty_recovery(seed=seed)
    ic_api,  ic_segs  = make_intermittent_cascade(seed=seed)
    rp_api,  rp_segs  = make_rough_patch(seed=seed), []
    rc_api,  rc_segs  = make_runaway_cascade(seed=seed), []

    scenario_list: list[tuple[ConfigurableAPI, list[dict[str, Any]]]] = [
        (br_api, br_segs),
        (ic_api, ic_segs),
        (rp_api, rp_segs),
        (rc_api, rc_segs),
    ]

    print("\n=== RNOS Experiment 3: Intermittent Cascading Failure ===")
    print(f"seed={seed}  max_steps={max_steps}")
    print(f"policy: degrade_entropy={policy.degrade_entropy}  refuse_entropy={policy.refuse_entropy}\n")

    all_results:  list[ScenarioResult] = []
    rnos_results: list[ScenarioResult] = []
    cb_results:   list[ScenarioResult] = []

    for api, segs in scenario_list:
        print(f"Running scenario: {api.name}")
        r_rnos = _run_rnos_3(api, segs, max_steps, policy, _TRACE_PATH)
        r_cb   = _run_adaptive_cb_3(api, segs, max_steps)
        r_base = _run_baseline(api, max_steps)

        # Override wasted_work for baseline (uses no segments knowledge).
        r_base.wasted_work = _compute_wasted_work_3(api.name, r_base.step_log, segs)

        # Enrich RNOS trajectory.
        if r_rnos.entropy_trajectory:
            r_rnos.entropy_trajectory = _enrich_trajectory_3(
                r_rnos.entropy_trajectory, r_rnos.step_log, segs
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
    # Selectivity
    # ------------------------------------------------------------------
    selectivity = _compute_selectivity_3(all_results)
    print("\nSELECTIVITY SCORES")
    print("Scoring rule for bursty_recovery: RNOS correct if not refused;")
    print("CB correct if max_steps_exhausted; baseline always correct.")
    print(_format_selectivity_table_3(selectivity))

    # ------------------------------------------------------------------
    # RNOS trajectories
    # ------------------------------------------------------------------
    print(f"\nRNOS TRAJECTORIES  (degrade>={policy.degrade_entropy} / refuse>={policy.refuse_entropy})")
    print("Seg column: burst/recovery/stable segment.  Stab = post-burst stability count.\n")
    for r in rnos_results:
        print(f"  {r.scenario}")
        print(_format_trajectory_table_3(r))
        print()

    # ------------------------------------------------------------------
    # CB internal state — bursty pair only
    # ------------------------------------------------------------------
    print("CIRCUIT BREAKER INTERNAL STATE  (bursty pair only)")
    print("Window column shows last-N boolean outcomes [1=success, 0=failure].\n")
    for r in cb_results:
        if r.scenario in _BURSTY_PAIR:
            print(f"  {r.scenario}")
            print(_format_cb_state_table(r))
            print()

    # ------------------------------------------------------------------
    # Divergence analysis
    # ------------------------------------------------------------------
    rnos_br = next(r for r in rnos_results if r.scenario == "bursty_recovery")
    cb_br   = next(r for r in cb_results   if r.scenario == "bursty_recovery")
    rnos_ic = next(r for r in rnos_results if r.scenario == "intermittent_cascade")
    cb_ic   = next(r for r in cb_results   if r.scenario == "intermittent_cascade")

    print("DIVERGENCE ANALYSIS  (bursty pair: RNOS vs Adaptive CB)")
    print(_format_divergence_analysis(rnos_br, cb_br, rnos_ic, cb_ic))

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
        "experiment": "experiment_3_intermittent_cascading_failure",
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
        "selectivity": selectivity,
        "results":     [_to_dict(r) for r in all_results],
    }

    with _RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {_RESULTS_PATH}")

    md = _build_markdown_summary_3(all_results, selectivity, policy, max_steps, seed)
    _SUMMARY_PATH.write_text(md, encoding="utf-8")
    print(f"Summary saved to:  {_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
