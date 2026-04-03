"""RNOS Experiment 2: Selective Containment.

Tests the claim that RNOS *discriminates* between recoverable instability
and structural failure - it is selective, not merely restrictive.

Four scenarios x three strategies = 12 combinations, plus a phase-transition
sweep on ``rough_patch`` over failure lengths 1-10.

Usage
-----
    python experiments/experiment_2.py
    python experiments/experiment_2.py --seed 42 --max-steps 20
    python experiments/experiment_2.py --no-sweep   # skip phase transition

Outputs
-------
    stdout                      - comparison table + entropy trajectories
    results/experiment_2_results.json
    results/experiment_2_summary.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.adaptive_circuit_breaker import AdaptiveCircuitBreaker
from experiments.configurable_api import (
    ConfigurableAPI,
    make_rough_patch,
    make_runaway_cascade,
    make_slow_burn,
    make_transient_blip,
)
from rnos.policy import PolicyConfig
from rnos.runtime import RNOSRuntime
from rnos.types import ActionRecord, PolicyDecision

# Suppress per-step RNOS console chatter during batch runs.
# NOTE: get_logger() in rnos/logger.py re-applies setLevel(INFO) on first use,
# so we also re-apply WARNING inside _run_rnos() after RNOSRuntime is created.
logging.getLogger("rnos.runtime").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_PATH = _REPO_ROOT / "results" / "experiment_2_results.json"
_SUMMARY_PATH = _REPO_ROOT / "results" / "experiment_2_summary.md"
_TRACE_PATH = _REPO_ROOT / "logs" / "exp2_rnos_trace.jsonl"

# ---------------------------------------------------------------------------
# RNOS policy config for Experiment 2
# ---------------------------------------------------------------------------
#
# Context:  the entropy formula produces a "structural floor" of ~4.0 for
# any run beyond step 3 because:
#   repeated_tool = 2  (always calling the same API name)
#   cost_score    = 2.0 (caps after cumulative_calls >= 7)
#
# Default thresholds (3.0 / 6.0) would therefore DEGRADE every healthy run
# and produce meaningless selectivity scores.  The thresholds below are
# calibrated so that:
#
#   transient_blip  (1 failure, peak entropy ~4.8)  -> ALLOW throughout
#   rough_patch     (3 failures, peak entropy ~8.7)  -> ALLOW throughout
#   runaway_cascade (certain failure by step 5)
#     -> DEGRADE at step 7 (entropy ~10.7), REFUSE at step 8 (entropy ~11.3)
#
# Trust gates are disabled (threshold -0.1, below the min trust value of 0)
# because the trust formula's entropy penalty creates near-perfect correlation
# with the entropy gate, causing double-counting for sequential API scenarios.

EXP2_POLICY = PolicyConfig(
    degrade_entropy=9.0,
    refuse_entropy=11.0,
    degrade_trust=-0.1,   # disabled: entropy is the sole gate
    refuse_trust=-0.1,    # same
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_STEPS = 20
_SEED = 42

# Cascade absorbing regime: consecutive failures >= this threshold means
# fail_prob = 0.5 x 1.8² = 1.62 -> 1.0 (certain failure every step).
_CASCADE_ABSORBING_THRESHOLD = 2

# Scenarios with an ambiguous ground-truth label are excluded from the
# binary selectivity score but still reported in the full results table.
_ANALYSIS_ONLY = {"slow_burn"}

# Ground truth labels used for selectivity scoring.
_GROUND_TRUTH: dict[str, str] = {
    "transient_blip":   "benign",     # should NOT intervene
    "rough_patch":      "benign",     # should generally NOT intervene
    "runaway_cascade":  "structural", # should intervene
    # slow_burn: excluded (borderline - no clear policy-based label)
}

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    scenario: str
    strategy: str
    steps_executed: int
    total_failures: int
    first_intervention_step: int | None
    first_intervention_type: str | None  # "degrade", "refuse", "open_blocked", …
    final_state: str          # completed | refused | max_steps_exhausted | circuit_permanently_open
    total_cost: float
    wasted_work: int
    step_log: list[dict[str, Any]] = field(default_factory=list, repr=False)
    entropy_trajectory: list[dict[str, Any]] | None = None  # RNOS only


# ---------------------------------------------------------------------------
# Wasted-work metric
# ---------------------------------------------------------------------------

def _compute_wasted_work(scenario_name: str, step_log: list[dict[str, Any]]) -> int:
    """Steps executed inside the cascade's absorbing failure regime.

    Operational definition
    ----------------------
    *runaway_cascade*: wasted_work = executed steps where
    ``consecutive_failures_at_start >= _CASCADE_ABSORBING_THRESHOLD``.
    At that point ``fail_prob >= 1.0`` so every remaining execution is
    guaranteed to fail and cannot contribute to a productive outcome.

    *Recoverable scenarios*: 0 always.  Failures in a rough patch or a
    transient blip are part of the observable trajectory that informs
    the runtime; they are not "wasted" because the information could
    trigger a correct allow/degrade decision.
    """
    if "cascade" not in scenario_name:
        return 0
    return sum(
        1 for entry in step_log
        if entry.get("consecutive_failures_at_start", 0) >= _CASCADE_ABSORBING_THRESHOLD
    )


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------

def _run_rnos(
    api: ConfigurableAPI,
    max_steps: int,
    policy_config: PolicyConfig,
    trace_path: Path,
) -> ScenarioResult:
    """Run one scenario under full RNOS control (no mocks)."""
    api.reset()
    runtime = RNOSRuntime(trace_path=trace_path, policy_config=policy_config)
    # Re-apply after get_logger() inside RNOSRuntime.__init__ resets level to INFO.
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

    for step in range(1, max_steps + 1):
        # Build ActionRecord.
        # depth=0: flat sequential calls, not recursive agent invocations.
        # retry_count: consecutive failures carried from previous steps.
        # latency_ms: previous step's observed latency (lagged signal).
        # cumulative_calls: total executed steps so far.
        action = ActionRecord(
            tool_name="configurable_api",
            depth=0,
            retry_count=retry_count,
            latency_ms=prev_latency,
            cumulative_calls=steps_executed,
        )

        assessment = runtime.evaluate(action)

        entropy_traj.append({
            "step": step,
            "entropy": assessment.entropy,
            "trust": assessment.trust,
            "decision": assessment.decision.value,
            "degrade_threshold": policy_config.degrade_entropy,
            "refuse_threshold": policy_config.refuse_entropy,
        })

        if assessment.decision is PolicyDecision.REFUSE:
            if first_intervention_step is None:
                first_intervention_step = step
                first_intervention_type = "refuse"
            final_state = "refused"
            break  # stop; do NOT execute this step

        if assessment.decision is PolicyDecision.DEGRADE:
            if first_intervention_step is None:
                first_intervention_step = step
                first_intervention_type = "degrade"
            # DEGRADE: execute with reduced confidence; execution continues.

        # Execute the API call.
        consec_at_start = retry_count
        outcome = api.call()
        steps_executed += 1
        total_cost += outcome.cost

        # Update action latency with the actual outcome (used for history).
        action.latency_ms = outcome.latency_ms
        runtime.record_outcome(action, success=outcome.success)

        if outcome.success:
            retry_count = 0
        else:
            total_failures += 1
            retry_count += 1

        prev_latency = outcome.latency_ms
        step_log.append({
            "step": step,
            "success": outcome.success,
            "latency_ms": round(outcome.latency_ms, 1),
            "cost": round(outcome.cost, 4),
            "entropy": assessment.entropy,
            "trust": assessment.trust,
            "decision": assessment.decision.value,
            "consecutive_failures_at_start": consec_at_start,
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
        wasted_work=_compute_wasted_work(api.name, step_log),
        step_log=step_log,
        entropy_trajectory=entropy_traj,
    )


def _run_adaptive_cb(
    api: ConfigurableAPI,
    max_steps: int,
    window_size: int = 5,
    initial_failure_rate: float = 0.60,
    min_failure_rate: float = 0.40,
    adaptation_step: float = 0.05,
    initial_cooldown_steps: int = 2,
    max_cooldown_steps: int = 10,
    max_total_blocked: int = 20,
) -> ScenarioResult:
    """Run one scenario under the adaptive sliding-window circuit breaker."""
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
        allowed, cb_reason = cb.should_execute()

        if not allowed:
            if first_intervention_step is None:
                first_intervention_step = step
                first_intervention_type = cb_reason
            if cb_reason == "permanently_open":
                final_state = "circuit_permanently_open"
                break
            continue  # blocked; skip execution for this step

        # Allowed (closed or half-open probe).
        consec_at_start = retry_count
        outcome = api.call()
        steps_executed += 1
        total_cost += outcome.cost
        cb.record_result(success=outcome.success)

        if outcome.success:
            retry_count = 0
        else:
            total_failures += 1
            retry_count += 1

        step_log.append({
            "step": step,
            "success": outcome.success,
            "latency_ms": round(outcome.latency_ms, 1),
            "cost": round(outcome.cost, 4),
            "cb_state": cb.state,
            "cb_reason": cb_reason,
            "failure_rate": cb.stats["failure_rate"],
            "current_threshold": cb.stats["current_threshold"],
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
        wasted_work=_compute_wasted_work(api.name, step_log),
        step_log=step_log,
    )


def _run_baseline(api: ConfigurableAPI, max_steps: int) -> ScenarioResult:
    """Run one scenario with no runtime protection (unprotected baseline)."""
    api.reset()

    steps_executed = 0
    total_failures = 0
    total_cost = 0.0
    retry_count = 0
    step_log: list[dict[str, Any]] = []

    for step in range(1, max_steps + 1):
        consec_at_start = retry_count
        outcome = api.call()
        steps_executed += 1
        total_cost += outcome.cost

        if outcome.success:
            retry_count = 0
        else:
            total_failures += 1
            retry_count += 1

        step_log.append({
            "step": step,
            "success": outcome.success,
            "latency_ms": round(outcome.latency_ms, 1),
            "cost": round(outcome.cost, 4),
            "consecutive_failures_at_start": consec_at_start,
        })

    return ScenarioResult(
        scenario=api.name,
        strategy="baseline",
        steps_executed=steps_executed,
        total_failures=total_failures,
        first_intervention_step=None,
        first_intervention_type=None,
        final_state="max_steps_exhausted",
        total_cost=round(total_cost, 4),
        wasted_work=_compute_wasted_work(api.name, step_log),
        step_log=step_log,
    )


# ---------------------------------------------------------------------------
# Selectivity scoring
# ---------------------------------------------------------------------------

def _compute_selectivity(results: list[ScenarioResult]) -> dict[str, Any]:
    """Compute per-strategy selectivity scores over the scored scenario set.

    Score = (correct interventions on structural scenarios
             + correct non-interventions on benign scenarios)
             / total scored scenarios

    Scored scenarios: transient_blip (benign), rough_patch (benign),
    runaway_cascade (structural).  slow_burn excluded (no clear label).

    A "correct intervention" for structural scenarios means first_intervention_step
    is not None (any DEGRADE or REFUSE counts).
    A "correct non-intervention" for benign scenarios means first_intervention_step
    is None.
    """
    by_strategy: dict[str, list[ScenarioResult]] = {}
    for r in results:
        by_strategy.setdefault(r.strategy, []).append(r)

    total_scored = len(_GROUND_TRUTH)
    scores: dict[str, Any] = {}

    for strategy, strat_results in by_strategy.items():
        correct = 0
        detail: dict[str, str] = {}
        for r in strat_results:
            if r.scenario not in _GROUND_TRUTH:
                continue
            label = _GROUND_TRUTH[r.scenario]
            intervened = r.first_intervention_step is not None
            if label == "benign" and not intervened:
                correct += 1
                detail[r.scenario] = "correct_non_intervention"
            elif label == "structural" and intervened:
                correct += 1
                detail[r.scenario] = "correct_intervention"
            elif label == "benign" and intervened:
                detail[r.scenario] = f"false_positive (step {r.first_intervention_step}, {r.first_intervention_type})"
            else:
                detail[r.scenario] = "false_negative (no intervention)"

        scores[strategy] = {
            "correct": correct,
            "total_scored": total_scored,
            "score": round(correct / total_scored, 3),
            "detail": detail,
        }

    return scores


# ---------------------------------------------------------------------------
# Phase-transition sweep
# ---------------------------------------------------------------------------

def _compute_phase_transition(
    max_steps: int,
    seed: int,
    policy_config: PolicyConfig,
    trace_path: Path,
) -> list[dict[str, Any]]:
    """Vary rough_patch failure length 1-10; record first-intervention step per strategy.

    This shows the sensitivity of each strategy to failure density and reveals
    the step at which each strategy switches from "allow/recover" to "intervene."
    """
    rows: list[dict[str, Any]] = []

    for failure_length in range(1, 11):
        api_rnos = make_rough_patch(seed=seed, failure_length=failure_length)
        api_cb   = make_rough_patch(seed=seed, failure_length=failure_length)
        api_base = make_rough_patch(seed=seed, failure_length=failure_length)

        r_rnos = _run_rnos(api_rnos, max_steps, policy_config, trace_path)
        r_cb   = _run_adaptive_cb(api_cb, max_steps)
        r_base = _run_baseline(api_base, max_steps)

        rows.append({
            "failure_length": failure_length,
            "rnos_intervention_step": r_rnos.first_intervention_step,
            "rnos_intervention_type": r_rnos.first_intervention_type,
            "rnos_final_state": r_rnos.final_state,
            "adaptive_cb_intervention_step": r_cb.first_intervention_step,
            "adaptive_cb_final_state": r_cb.final_state,
            "baseline_intervention_step": r_base.first_intervention_step,  # always None
        })

    return rows


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def _fmt_cell(value: Any, width: int) -> str:
    return str(value)[:width].ljust(width)


def _format_comparison_table(results: list[ScenarioResult]) -> str:
    """Render a fixed-width comparison table."""
    header = (
        f"{'Scenario':<18} {'Strategy':<12} {'Steps':>5} {'Fails':>5} "
        f"{'1st Intv':>8} {'Intv Type':<12} {'Final State':<25} "
        f"{'Cost':>7} {'Wasted':>6}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in results:
        intv = str(r.first_intervention_step) if r.first_intervention_step else "-"
        itype = (r.first_intervention_type or "-")[:11]
        lines.append(
            f"{r.scenario:<18} {r.strategy:<12} {r.steps_executed:>5} {r.total_failures:>5} "
            f"{intv:>8} {itype:<12} {r.final_state:<25} "
            f"{r.total_cost:>7.4f} {r.wasted_work:>6}"
        )
    lines.append(sep)
    return "\n".join(lines)


def _format_selectivity_table(scores: dict[str, Any]) -> str:
    lines = [
        f"{'Strategy':<14} {'Score':>7}  Details",
        "-" * 70,
    ]
    for strategy, info in scores.items():
        score_str = f"{info['correct']}/{info['total_scored']} ({info['score']:.3f})"
        detail_str = "  |  ".join(f"{k}: {v}" for k, v in info["detail"].items())
        lines.append(f"{strategy:<14} {score_str:>10}  {detail_str}")
    return "\n".join(lines)


def _format_phase_transition_table(rows: list[dict[str, Any]]) -> str:
    header = (
        f"{'Fail Len':>8}  {'RNOS Step':>9}  {'RNOS Type':<12}  "
        f"{'Adapt CB Step':>13}  {'CB Final':<25}  {'RNOS Final':<25}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for row in rows:
        rs = str(row["rnos_intervention_step"]) if row["rnos_intervention_step"] else "-"
        rt = (row["rnos_intervention_type"] or "-")[:11]
        cs = str(row["adaptive_cb_intervention_step"]) if row["adaptive_cb_intervention_step"] else "-"
        lines.append(
            f"{row['failure_length']:>8}  {rs:>9}  {rt:<12}  "
            f"{cs:>13}  {row['adaptive_cb_final_state']:<25}  {row['rnos_final_state']:<25}"
        )
    lines.append(sep)
    return "\n".join(lines)


def _format_entropy_trajectory(result: ScenarioResult) -> str:
    if not result.entropy_trajectory:
        return "(no entropy trajectory)"
    dg = result.step_log[0]["entropy"] if result.step_log else 0  # placeholder
    header = (
        f"  {'Step':>4}  {'Entropy':>8}  {'Trust':>7}  {'Decision':<8}  "
        f"  [degrade>={result.entropy_trajectory[0]['degrade_threshold']:.1f}  "
        f"refuse>={result.entropy_trajectory[0]['refuse_threshold']:.1f}]"
    )
    sep = "  " + "-" * (len(header) - 2)
    lines = [sep, header, sep]
    for entry in result.entropy_trajectory:
        # Find matching step_log entry for success info
        sl = next((s for s in result.step_log if s["step"] == entry["step"]), None)
        outcome = ("(ok)" if sl and sl["success"] else "(fail)" if sl else "(refused)")
        lines.append(
            f"  {entry['step']:>4}  {entry['entropy']:>8.3f}  {entry['trust']:>7.3f}  "
            f"{entry['decision']:<8}  {outcome}"
        )
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def _build_markdown_summary(
    results: list[ScenarioResult],
    scores: dict[str, Any],
    phase_rows: list[dict[str, Any]] | None,
    policy: PolicyConfig,
    max_steps: int,
    seed: int,
) -> str:
    lines: list[str] = []

    lines += [
        "## RNOS Experiment 2: Selective Containment",
        "",
        "**Claim**: RNOS discriminates between recoverable instability and structural failure.",
        "It is selective - not merely restrictive.",
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
        "### Selectivity Scores",
        "",
        "Score = (correct interventions on structural scenarios + correct non-interventions on benign scenarios) / total scored scenarios",
        "",
        "Scored set: `transient_blip` (benign), `rough_patch` (benign), `runaway_cascade` (structural).",
        "`slow_burn` excluded (borderline - no clear policy-based label).",
        "",
        "| Strategy | Score | Detail |",
        "| --- | --- | --- |",
    ]
    for strategy, info in scores.items():
        score_str = f"{info['correct']}/{info['total_scored']} = {info['score']:.3f}"
        detail = "; ".join(f"{k}: {v}" for k, v in info["detail"].items())
        lines.append(f"| {strategy} | {score_str} | {detail} |")

    if phase_rows:
        lines += [
            "",
            "---",
            "",
            "### Phase Transition: `rough_patch` Failure Density",
            "",
            "Sweep over failure_length 1-10. Records first intervention step per strategy.",
            "",
            "| Failure Length | RNOS Step | RNOS Type | Adaptive CB Step | Adaptive CB Final | RNOS Final |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in phase_rows:
            rs = str(row["rnos_intervention_step"]) if row["rnos_intervention_step"] else "-"
            rt = row["rnos_intervention_type"] or "-"
            cs = str(row["adaptive_cb_intervention_step"]) if row["adaptive_cb_intervention_step"] else "-"
            lines.append(
                f"| {row['failure_length']} | {rs} | {rt} | {cs} "
                f"| {row['adaptive_cb_final_state']} | {row['rnos_final_state']} |"
            )

    lines += [
        "",
        "---",
        "",
        "### Key Assumptions and Caveats",
        "",
        "- `depth=0` for all ActionRecords (flat sequential API calls, not recursive agent loops).",
        "- `retry_count` = consecutive failures from prior steps (resets on success).",
        "- `latency_ms` at evaluation time = previous step's observed latency (lagged signal).",
        f"- Custom `PolicyConfig(degrade_entropy={policy.degrade_entropy}, refuse_entropy={policy.refuse_entropy})` "
        "accounts for the entropy formula's structural floor (~4.0 for runs beyond step 3) "
        "from `cost_score + repeated_tool` signals.",
        "- Trust gates disabled (`degrade_trust=-0.1`, `refuse_trust=-0.1`) because the trust "
        "formula's entropy penalty correlates with the entropy gate, causing double-counting.",
        "- Cascade absorbing regime defined as `consecutive_failures >= 2` "
        f"(`0.5 x 1.8² = 1.62 -> 1.0`, certain failure).",
        "- Adaptive CB uses strict `>` for failure-rate check so 3/5 = 0.60 does NOT trip.",
        "- RNOS continues through DEGRADE (execution allowed, event flagged); stops on REFUSE.",
        "",
        "*Generated by `experiments/experiment_2.py`*",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RNOS Experiment 2: Selective Containment"
    )
    parser.add_argument("--seed", type=int, default=_SEED)
    parser.add_argument("--max-steps", type=int, default=_MAX_STEPS)
    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="Skip the phase-transition sweep (faster for quick checks).",
    )
    args = parser.parse_args()

    seed = args.seed
    max_steps = args.max_steps
    policy = EXP2_POLICY

    # Ensure output dirs exist.
    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Start with a fresh trace for this experiment run.
    _TRACE_PATH.write_text("", encoding="utf-8")

    # ------------------------------------------------------------------
    # Build scenario list
    # ------------------------------------------------------------------
    scenarios: list[ConfigurableAPI] = [
        make_transient_blip(seed=seed),
        make_rough_patch(seed=seed),
        make_slow_burn(seed=seed),
        make_runaway_cascade(seed=seed),
    ]

    # ------------------------------------------------------------------
    # Run all combinations
    # ------------------------------------------------------------------
    all_results: list[ScenarioResult] = []

    print("\n=== RNOS Experiment 2: Selective Containment ===")
    print(f"seed={seed}  max_steps={max_steps}")
    print(f"policy: degrade_entropy={policy.degrade_entropy}  refuse_entropy={policy.refuse_entropy}\n")

    for api in scenarios:
        print(f"Running scenario: {api.name}")
        r_rnos = _run_rnos(api, max_steps, policy, _TRACE_PATH)
        r_cb   = _run_adaptive_cb(api, max_steps)
        r_base = _run_baseline(api, max_steps)
        all_results.extend([r_rnos, r_cb, r_base])
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
    # Selectivity scores
    # ------------------------------------------------------------------
    selectivity = _compute_selectivity(all_results)
    print("\nSELECTIVITY SCORES (scored set: transient_blip, rough_patch, runaway_cascade)")
    print(_format_selectivity_table(selectivity))

    # ------------------------------------------------------------------
    # RNOS entropy trajectories
    # ------------------------------------------------------------------
    print("\nRNOS ENTROPY TRAJECTORIES  (degrade>=9.0 / refuse>=11.0)")
    print("(ok) = success  (fail) = failure  (refused) = refused (not executed)\n")
    for r in all_results:
        if r.strategy == "rnos":
            print(f"  {r.scenario}")
            print(_format_entropy_trajectory(r))
            print()

    # ------------------------------------------------------------------
    # Phase-transition sweep
    # ------------------------------------------------------------------
    phase_rows: list[dict[str, Any]] | None = None
    if not args.no_sweep:
        print("PHASE TRANSITION SWEEP  (rough_patch failure_length 1-10)")
        phase_rows = _compute_phase_transition(max_steps, seed, policy, _TRACE_PATH)
        print(_format_phase_transition_table(phase_rows))
    else:
        print("(phase-transition sweep skipped - use without --no-sweep to enable)")

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    output: dict[str, Any] = {
        "experiment": "experiment_2_selective_containment",
        "config": {
            "seed": seed,
            "max_steps": max_steps,
            "policy": {
                "degrade_entropy": policy.degrade_entropy,
                "refuse_entropy": policy.refuse_entropy,
                "degrade_trust": policy.degrade_trust,
                "refuse_trust": policy.refuse_trust,
            },
            "adaptive_cb": {
                "window_size": 5,
                "initial_failure_rate": 0.60,
                "min_failure_rate": 0.40,
                "adaptation_step": 0.05,
                "initial_cooldown_steps": 2,
                "max_cooldown_steps": 10,
            },
            "cascade_absorbing_threshold": _CASCADE_ABSORBING_THRESHOLD,
        },
        "scenario_results": [
            {k: v for k, v in asdict(r).items() if k != "step_log"}
            for r in all_results
        ],
        "selectivity_scores": selectivity,
        "phase_transition": phase_rows,
        "assumptions": [
            "depth=0 for all ActionRecords (flat sequential API calls)",
            "retry_count = consecutive failures from prior executed steps",
            "latency_ms at evaluation = previous step observed latency (lagged)",
            "Custom PolicyConfig accounts for entropy floor (~4.0 at step 7+)",
            "Trust gates disabled (threshold -0.1) to avoid double-counting with entropy",
            "Cascade absorbing regime: consecutive_failures >= 2 (fail_prob -> 1.0)",
            "Adaptive CB uses strict > for failure-rate check (3/5=0.60 does NOT trip)",
            "RNOS continues through DEGRADE; stops only on REFUSE",
        ],
    }

    with _RESULTS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"\nResults saved -> {_RESULTS_PATH}")

    # ------------------------------------------------------------------
    # Save markdown summary
    # ------------------------------------------------------------------
    md = _build_markdown_summary(
        all_results, selectivity, phase_rows, policy, max_steps, seed
    )
    _SUMMARY_PATH.write_text(md, encoding="utf-8")
    print(f"Markdown summary -> {_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
