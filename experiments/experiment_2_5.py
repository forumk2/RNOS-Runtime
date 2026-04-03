"""RNOS Experiment 2.5: Trajectory-Aware Discrimination.

Tests the claim that RNOS can discriminate between two failure trajectories
that are *indistinguishable through step 6* and diverge only at step 7+.

The key constraint: both matched scenarios share identical step schedules
through step 6 (same outcomes, same latency), so RNOS cannot gain any
entropy-magnitude advantage from early-step signal. Discrimination must
come from trajectory change *after* the divergence point.

Four scenarios x three strategies = 12 combinations.

Pair classification
-------------------
Easy pairs (unambiguous from step 1):
    rough_patch      - benign  (same as Experiment 2)
    runaway_cascade  - structural  (same as Experiment 2)

Hard pairs (indistinguishable through step 6):
    matched_recovery - benign  (diverges at step 7 toward recovery)
    matched_collapse - structural  (diverges at step 7 toward absorbing failure)

Usage
-----
    python experiments/experiment_2_5.py
    python experiments/experiment_2_5.py --seed 42 --max-steps 20

Outputs
-------
    stdout                          - comparison table + trajectory tables
    results/experiment_2_5_results.json
    results/experiment_2_5_summary.md
    logs/exp2_5_rnos_trace.jsonl
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

# Re-use all strategy runners, formatters and data structures from Experiment 2.
from experiments.experiment_2 import (
    EXP2_POLICY,
    ScenarioResult,
    _format_comparison_table,
    _run_adaptive_cb,
    _run_baseline,
    _run_rnos,
)
from experiments.configurable_api import (
    ConfigurableAPI,
    make_matched_collapse,
    make_matched_recovery,
    make_rough_patch,
    make_runaway_cascade,
)
from rnos.policy import PolicyConfig

# Silence RNOS runtime logger for batch runs.
logging.getLogger("rnos.runtime").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_PATH = _REPO_ROOT / "results" / "experiment_2_5_results.json"
_SUMMARY_PATH = _REPO_ROOT / "results" / "experiment_2_5_summary.md"
_TRACE_PATH = _REPO_ROOT / "logs" / "exp2_5_rnos_trace.jsonl"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_STEPS = 20
_SEED = 42

# Absorbing thresholds
_CASCADE_ABSORBING_THRESHOLD = 2      # runaway_cascade: consec >= 2 -> certain fail
_MATCHED_COLLAPSE_DIVERGENCE_STEP = 7 # matched_collapse: steps >= 7 are wasted

# Ground truth for all four scenarios.
_GROUND_TRUTH_25: dict[str, str] = {
    "matched_recovery": "benign",
    "matched_collapse": "structural",
    "rough_patch":      "benign",
    "runaway_cascade":  "structural",
}

# Pair classification for the easy/hard breakdown.
_EASY_PAIR = {"rough_patch", "runaway_cascade"}
_HARD_PAIR = {"matched_recovery", "matched_collapse"}


# ---------------------------------------------------------------------------
# Wasted-work metric (Experiment 2.5 variant)
# ---------------------------------------------------------------------------

def _compute_wasted_work_25(scenario_name: str, step_log: list[dict[str, Any]]) -> int:
    """Steps executed in absorbing / post-divergence failure regime.

    runaway_cascade:  consecutive_failures_at_start >= 2 (certain fail).
    matched_collapse: step >= _MATCHED_COLLAPSE_DIVERGENCE_STEP (absorbing from step 7).
    All others: 0 (failures are informative, not wasted).
    """
    if "cascade" in scenario_name:
        return sum(
            1 for entry in step_log
            if entry.get("consecutive_failures_at_start", 0) >= _CASCADE_ABSORBING_THRESHOLD
        )
    if scenario_name == "matched_collapse":
        return sum(
            1 for entry in step_log
            if entry.get("step", 0) >= _MATCHED_COLLAPSE_DIVERGENCE_STEP
        )
    return 0


# ---------------------------------------------------------------------------
# Trajectory enrichment (observational only; does NOT influence RNOS decisions)
# ---------------------------------------------------------------------------

def _enrich_trajectory(
    entropy_traj: list[dict[str, Any]],
    step_log: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append trajectory signals to each entropy trajectory entry.

    These signals are purely observational and logged AFTER the run.
    They have no effect on RNOS decisions.

    Added fields per entry:
        entropy_slope     - first difference of entropy (step N vs N-1)
        entropy_curvature - second difference of entropy (slope N vs slope N-1)
        sliding_fail_rate_3 - failure rate over last 3 *executed* steps (0.0-1.0)
        recovery_signal   - True if this step succeeded after >= 2 consecutive failures
    """
    # Build a step -> success lookup from the step_log.
    step_success: dict[int, bool | None] = {}
    step_consec: dict[int, int] = {}
    for entry in step_log:
        s = entry["step"]
        step_success[s] = entry["success"]
        step_consec[s] = entry.get("consecutive_failures_at_start", 0)

    enriched: list[dict[str, Any]] = []
    prev_entropy: float | None = None
    prev_slope: float | None = None

    # Sliding window: list of success booleans in execution order.
    exec_window: list[bool] = []

    for i, entry in enumerate(entropy_traj):
        step = entry["step"]
        e = float(entry["entropy"])

        # Slope (first difference).
        slope = (e - prev_entropy) if prev_entropy is not None else 0.0

        # Curvature (second difference).
        curvature = (slope - prev_slope) if prev_slope is not None else 0.0

        # Update exec window with this step's outcome (if it was executed).
        if step in step_success and step_success[step] is not None:
            exec_window.append(bool(step_success[step]))

        # Sliding failure rate over last 3 executed steps.
        window3 = exec_window[-3:] if len(exec_window) >= 3 else exec_window
        fail_rate_3 = (sum(1 for s in window3 if not s) / len(window3)) if window3 else 0.0

        # Recovery signal: succeeded this step after >= 2 prior consecutive failures.
        prior_consec = step_consec.get(step, 0)
        succeeded_now = step_success.get(step, False)
        recovery = bool(succeeded_now and prior_consec >= 2)

        row = dict(entry)
        row["entropy_slope"] = round(slope, 4)
        row["entropy_curvature"] = round(curvature, 4)
        row["sliding_fail_rate_3"] = round(fail_rate_3, 3)
        row["recovery_signal"] = recovery

        enriched.append(row)
        prev_entropy = e
        prev_slope = slope

    return enriched


# ---------------------------------------------------------------------------
# Step-6 entropy assertion
# ---------------------------------------------------------------------------

def _validate_entropy_assertion(
    rnos_results: list[ScenarioResult],
) -> dict[str, Any]:
    """Assert that |entropy(recovery, step6) - entropy(collapse, step6)| < 0.5.

    Returns a dict with assertion result and the actual values.
    """
    mr = next((r for r in rnos_results if r.scenario == "matched_recovery"), None)
    mc = next((r for r in rnos_results if r.scenario == "matched_collapse"), None)

    if mr is None or mc is None or not mr.entropy_trajectory or not mc.entropy_trajectory:
        return {
            "passed": False,
            "reason": "entropy trajectory not available for one or both matched scenarios",
        }

    def _entropy_at_step(traj: list[dict[str, Any]], step: int) -> float | None:
        entry = next((e for e in traj if e["step"] == step), None)
        return float(entry["entropy"]) if entry else None

    e_mr = _entropy_at_step(mr.entropy_trajectory, 6)
    e_mc = _entropy_at_step(mc.entropy_trajectory, 6)

    if e_mr is None or e_mc is None:
        return {
            "passed": False,
            "reason": f"step 6 missing from trajectory (recovery={e_mr}, collapse={e_mc})",
        }

    diff = abs(e_mr - e_mc)
    passed = diff < 0.5
    return {
        "passed": passed,
        "matched_recovery_entropy_step6": round(e_mr, 4),
        "matched_collapse_entropy_step6": round(e_mc, 4),
        "abs_diff": round(diff, 4),
        "threshold": 0.5,
        "reason": "PASS: scenarios are entropy-matched through step 6" if passed
                  else f"FAIL: entropy gap {diff:.4f} >= 0.5 (scenarios are distinguishable before divergence point)",
    }


# ---------------------------------------------------------------------------
# Selectivity scoring with easy/hard pair breakdown
# ---------------------------------------------------------------------------

def _compute_selectivity_25(results: list[ScenarioResult]) -> dict[str, Any]:
    """Compute per-strategy selectivity scores with easy/hard pair breakdown.

    Overall score = correct / 4 (all four scored scenarios).
    Easy pair score = correct / 2 (rough_patch + runaway_cascade).
    Hard pair score = correct / 2 (matched_recovery + matched_collapse).
    """
    by_strategy: dict[str, list[ScenarioResult]] = {}
    for r in results:
        by_strategy.setdefault(r.strategy, []).append(r)

    scores: dict[str, Any] = {}

    for strategy, strat_results in by_strategy.items():
        overall_correct = 0
        easy_correct = 0
        hard_correct = 0
        detail: dict[str, str] = {}

        for r in strat_results:
            if r.scenario not in _GROUND_TRUTH_25:
                continue
            label = _GROUND_TRUTH_25[r.scenario]
            intervened = r.first_intervention_step is not None

            correct_call = (label == "benign" and not intervened) or \
                           (label == "structural" and intervened)
            overall_correct += int(correct_call)

            if r.scenario in _EASY_PAIR:
                easy_correct += int(correct_call)
            elif r.scenario in _HARD_PAIR:
                hard_correct += int(correct_call)

            if label == "benign" and not intervened:
                detail[r.scenario] = "correct_non_intervention"
            elif label == "structural" and intervened:
                detail[r.scenario] = "correct_intervention"
            elif label == "benign" and intervened:
                detail[r.scenario] = (
                    f"false_positive (step {r.first_intervention_step},"
                    f" {r.first_intervention_type})"
                )
            else:
                detail[r.scenario] = "false_negative (no intervention)"

        scores[strategy] = {
            "overall_correct": overall_correct,
            "overall_total": 4,
            "overall_score": round(overall_correct / 4, 3),
            "easy_correct": easy_correct,
            "easy_total": 2,
            "easy_score": round(easy_correct / 2, 3),
            "hard_correct": hard_correct,
            "hard_total": 2,
            "hard_score": round(hard_correct / 2, 3),
            "detail": detail,
        }

    return scores


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _format_selectivity_table_25(scores: dict[str, Any]) -> str:
    header = (
        f"{'Strategy':<14} {'Overall':>9} {'Easy':>6} {'Hard':>6}  Details"
    )
    lines = ["-" * 90, header, "-" * 90]
    for strategy, info in scores.items():
        overall = f"{info['overall_correct']}/{info['overall_total']} ({info['overall_score']:.3f})"
        easy = f"{info['easy_correct']}/{info['easy_total']}"
        hard = f"{info['hard_correct']}/{info['hard_total']}"
        detail_str = "  |  ".join(f"{k}: {v}" for k, v in info["detail"].items())
        lines.append(
            f"{strategy:<14} {overall:>9}  {easy:>5}  {hard:>5}  {detail_str}"
        )
    return "\n".join(lines)


def _format_trajectory_table(result: ScenarioResult) -> str:
    """Rich per-step table including entropy signals and trajectory metrics."""
    if not result.entropy_trajectory:
        return "(no entropy trajectory)"

    header = (
        f"  {'Step':>4}  {'Entropy':>8}  {'Slope':>7}  {'Curv':>7}  "
        f"{'FailR3':>6}  {'Recov':>5}  {'Decision':<8}  {'Outcome'}"
    )
    sep = "  " + "-" * (len(header) - 2)
    lines = [sep, header, sep]

    # Build a step -> outcome lookup from the step_log.
    sl_map = {entry["step"]: entry for entry in result.step_log}

    for entry in result.entropy_trajectory:
        step = entry["step"]
        sl = sl_map.get(step)
        outcome = ("(ok)" if sl and sl["success"]
                   else "(fail)" if sl else "(refused)")
        slope = entry.get("entropy_slope", 0.0)
        curv = entry.get("entropy_curvature", 0.0)
        fr3 = entry.get("sliding_fail_rate_3", 0.0)
        rec = "yes" if entry.get("recovery_signal") else "-"
        lines.append(
            f"  {step:>4}  {entry['entropy']:>8.3f}  {slope:>+7.3f}  {curv:>+7.3f}  "
            f"{fr3:>6.3f}  {rec:>5}  {entry['decision']:<8}  {outcome}"
        )
    lines.append(sep)
    return "\n".join(lines)


def _format_divergence_analysis(
    mr_result: ScenarioResult,
    mc_result: ScenarioResult,
) -> str:
    """Side-by-side comparison of matched pair at divergence window (steps 5-10)."""
    if not mr_result.entropy_trajectory or not mc_result.entropy_trajectory:
        return "(trajectory not available)"

    mr_map = {e["step"]: e for e in mr_result.entropy_trajectory}
    mc_map = {e["step"]: e for e in mc_result.entropy_trajectory}

    header = (
        f"{'Step':>4}  "
        f"{'RECOVERY entropy':>16} {'slope':>7} {'fail_r3':>7} {'decision':<9}  "
        f"{'COLLAPSE entropy':>16} {'slope':>7} {'fail_r3':>7} {'decision':<9}"
    )
    sep = "-" * len(header)
    lines = [
        "Divergence analysis: matched pair steps 5-10",
        "(identical through step 6; diverge from step 7)",
        sep,
        header,
        sep,
    ]

    for step in range(5, 11):
        mr_e = mr_map.get(step)
        mc_e = mc_map.get(step)

        def _fmt(e: dict[str, Any] | None) -> str:
            if e is None:
                return f"{'N/A':>16} {'N/A':>7} {'N/A':>7} {'N/A':<9}"
            return (
                f"{e['entropy']:>16.3f} {e.get('entropy_slope', 0):>+7.3f} "
                f"{e.get('sliding_fail_rate_3', 0):>7.3f} {e['decision']:<9}"
            )

        marker = " <-- diverges" if step == _MATCHED_COLLAPSE_DIVERGENCE_STEP else ""
        lines.append(f"{step:>4}  {_fmt(mr_e)}  {_fmt(mc_e)}{marker}")

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def _build_markdown_summary_25(
    results: list[ScenarioResult],
    scores: dict[str, Any],
    assertion: dict[str, Any],
    policy: PolicyConfig,
    max_steps: int,
    seed: int,
) -> str:
    lines: list[str] = []

    lines += [
        "## RNOS Experiment 2.5: Trajectory-Aware Discrimination",
        "",
        "**Claim**: RNOS can discriminate between two failure trajectories that are "
        "indistinguishable through step 6 and diverge only at step 7+.",
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
        intv = str(r.first_intervention_step) if r.first_intervention_step else "-"
        itype = r.first_intervention_type or "-"
        lines.append(
            f"| {r.scenario} | {r.strategy} | {r.steps_executed} | {r.total_failures} "
            f"| {intv} | {itype} | {r.final_state} | {r.total_cost:.4f} | {r.wasted_work} |"
        )

    lines += [
        "",
        "---",
        "",
        "### Step-6 Entropy Assertion",
        "",
        f"| Field | Value |",
        "| --- | --- |",
        f"| matched_recovery entropy (step 6) | {assertion.get('matched_recovery_entropy_step6', 'N/A')} |",
        f"| matched_collapse entropy (step 6) | {assertion.get('matched_collapse_entropy_step6', 'N/A')} |",
        f"| abs diff | {assertion.get('abs_diff', 'N/A')} |",
        f"| threshold | < 0.5 |",
        f"| result | {'PASS' if assertion.get('passed') else 'FAIL'}: {assertion.get('reason', '')} |",
        "",
        "---",
        "",
        "### Selectivity Scores (with easy/hard pair breakdown)",
        "",
        "Overall = 4 scenarios. Easy pair = rough_patch + runaway_cascade. "
        "Hard pair = matched_recovery + matched_collapse.",
        "",
        "| Strategy | Overall | Easy | Hard | Detail |",
        "| --- | --- | --- | --- | --- |",
    ]

    for strategy, info in scores.items():
        overall = f"{info['overall_correct']}/{info['overall_total']} = {info['overall_score']:.3f}"
        easy = f"{info['easy_correct']}/{info['easy_total']}"
        hard = f"{info['hard_correct']}/{info['hard_total']}"
        detail = "; ".join(f"{k}: {v}" for k, v in info["detail"].items())
        lines.append(f"| {strategy} | {overall} | {easy} | {hard} | {detail} |")

    lines += [
        "",
        "---",
        "",
        "### Key Design Constraints",
        "",
        "- Matched pair shares identical steps 1-6: [T, T, T(500ms), F, F, F].",
        f"  Divergence occurs at step {_MATCHED_COLLAPSE_DIVERGENCE_STEP}: recovery succeeds, collapse deepens.",
        "- Trajectory signals (slope, curvature, sliding_fail_rate_3, recovery_signal) are "
        "  logged as observational metadata ONLY. They do NOT influence RNOS decisions.",
        "- Same `PolicyConfig` and CB parameters as Experiment 2 — no re-tuning.",
        "- Trust gates remain disabled (`degrade_trust=-0.1`, `refuse_trust=-0.1`).",
        f"- `wasted_work` for matched_collapse = steps executed at step >= {_MATCHED_COLLAPSE_DIVERGENCE_STEP}.",
        "",
        "*Generated by `experiments/experiment_2_5.py`*",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RNOS Experiment 2.5: Trajectory-Aware Discrimination"
    )
    parser.add_argument("--seed", type=int, default=_SEED)
    parser.add_argument("--max-steps", type=int, default=_MAX_STEPS)
    args = parser.parse_args()

    seed = args.seed
    max_steps = args.max_steps
    policy = EXP2_POLICY

    # Ensure output dirs exist.
    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TRACE_PATH.write_text("", encoding="utf-8")

    scenarios: list[ConfigurableAPI] = [
        make_matched_recovery(seed=seed),
        make_matched_collapse(seed=seed),
        make_rough_patch(seed=seed),
        make_runaway_cascade(seed=seed),
    ]

    print("\n=== RNOS Experiment 2.5: Trajectory-Aware Discrimination ===")
    print(f"seed={seed}  max_steps={max_steps}")
    print(f"policy: degrade_entropy={policy.degrade_entropy}  refuse_entropy={policy.refuse_entropy}\n")

    all_results: list[ScenarioResult] = []
    rnos_results: list[ScenarioResult] = []

    for api in scenarios:
        print(f"Running scenario: {api.name}")
        r_rnos = _run_rnos(api, max_steps, policy, _TRACE_PATH)
        r_cb = _run_adaptive_cb(api, max_steps)
        r_base = _run_baseline(api, max_steps)

        # Override wasted_work with the Experiment 2.5 definition.
        r_rnos.wasted_work = _compute_wasted_work_25(api.name, r_rnos.step_log)
        r_cb.wasted_work = _compute_wasted_work_25(api.name, r_cb.step_log)
        r_base.wasted_work = _compute_wasted_work_25(api.name, r_base.step_log)

        # Enrich RNOS entropy trajectory with observational signals.
        if r_rnos.entropy_trajectory:
            r_rnos.entropy_trajectory = _enrich_trajectory(
                r_rnos.entropy_trajectory, r_rnos.step_log
            )

        all_results.extend([r_rnos, r_cb, r_base])
        rnos_results.append(r_rnos)

        print(
            f"  rnos:        steps={r_rnos.steps_executed:>2}  fails={r_rnos.total_failures:>2}"
            f"  intv_step={str(r_rnos.first_intervention_step):>4}  state={r_rnos.final_state}"
        )
        print(
            f"  adaptive_cb: steps={r_cb.steps_executed:>2}  fails={r_cb.total_failures:>2}"
            f"  intv_step={str(r_cb.first_intervention_step):>4}  state={r_cb.final_state}"
        )
        print(
            f"  baseline:    steps={r_base.steps_executed:>2}  fails={r_base.total_failures:>2}"
            f"  intv_step={'-':>4}  state={r_base.final_state}"
        )

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("COMPARISON TABLE")
    print("=" * 72)
    print(_format_comparison_table(all_results))

    # ------------------------------------------------------------------
    # Step-6 entropy assertion
    # ------------------------------------------------------------------
    assertion = _validate_entropy_assertion(rnos_results)
    print("\nSTEP-6 ENTROPY ASSERTION (matched pair must be indistinguishable through step 6)")
    status = "PASS" if assertion["passed"] else "FAIL"
    print(f"  Status: {status}")
    print(f"  matched_recovery entropy (step 6): {assertion.get('matched_recovery_entropy_step6', 'N/A')}")
    print(f"  matched_collapse entropy (step 6): {assertion.get('matched_collapse_entropy_step6', 'N/A')}")
    print(f"  abs diff: {assertion.get('abs_diff', 'N/A')}  (threshold: < 0.5)")
    print(f"  {assertion.get('reason', '')}")

    # ------------------------------------------------------------------
    # Selectivity scores
    # ------------------------------------------------------------------
    selectivity = _compute_selectivity_25(all_results)
    print("\nSELECTIVITY SCORES (easy pair: rough_patch + runaway_cascade / hard pair: matched_*)")
    print(_format_selectivity_table_25(selectivity))

    # ------------------------------------------------------------------
    # RNOS entropy trajectories with enriched signals
    # ------------------------------------------------------------------
    print("\nRNOS ENTROPY TRAJECTORIES WITH TRAJECTORY SIGNALS")
    print("  (slope/curvature/fail_rate3/recovery are observational only)")
    print(f"  degrade >= {policy.degrade_entropy}  /  refuse >= {policy.refuse_entropy}\n")
    for r in rnos_results:
        print(f"  {r.scenario}")
        print(_format_trajectory_table(r))
        print()

    # ------------------------------------------------------------------
    # Divergence analysis for matched pair
    # ------------------------------------------------------------------
    mr = next((r for r in rnos_results if r.scenario == "matched_recovery"), None)
    mc = next((r for r in rnos_results if r.scenario == "matched_collapse"), None)
    if mr and mc:
        print("DIVERGENCE ANALYSIS  (RNOS, matched pair)")
        print(_format_divergence_analysis(mr, mc))

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    def _result_to_dict(r: ScenarioResult) -> dict[str, Any]:
        d = asdict(r)
        # Convert PolicyDecision enums inside entropy_trajectory to strings.
        if d.get("entropy_trajectory"):
            for entry in d["entropy_trajectory"]:
                if hasattr(entry.get("decision"), "value"):
                    entry["decision"] = entry["decision"].value
        return d

    output: dict[str, Any] = {
        "experiment": "experiment_2_5_trajectory_aware_discrimination",
        "config": {
            "seed": seed,
            "max_steps": max_steps,
            "policy": {
                "degrade_entropy": policy.degrade_entropy,
                "refuse_entropy": policy.refuse_entropy,
                "degrade_trust": policy.degrade_trust,
                "refuse_trust": policy.refuse_trust,
            },
        },
        "step6_entropy_assertion": assertion,
        "selectivity": selectivity,
        "results": [_result_to_dict(r) for r in all_results],
    }

    with _RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {_RESULTS_PATH}")

    # ------------------------------------------------------------------
    # Save markdown summary
    # ------------------------------------------------------------------
    md = _build_markdown_summary_25(all_results, selectivity, assertion, policy, max_steps, seed)
    _SUMMARY_PATH.write_text(md, encoding="utf-8")
    print(f"Summary saved to:  {_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
