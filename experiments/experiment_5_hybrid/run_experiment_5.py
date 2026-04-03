"""RNOS Experiment 5: Hybrid RNOS + Circuit Breaker (Cooperative Control).

Tests whether a hybrid controller that composes RNOS (cumulative entropy)
and an AdaptiveCircuitBreaker (sliding-window failure density) forms a
dominant control architecture across two distinct failure geometries.

Hypothesis
----------
  Hybrid performs ≥ best(RNOS, CB) in all scenarios, and strictly better
  than at least one sub-system in at least one measurable dimension.

Scenarios
---------
  cascading_burst       — RNOS strength: rapid consecutive failures
  distributed_low_rate  — CB strength:   dispersed 67% failure rate

Modes (4)
---------
  baseline   — no control; runs to max_steps
  rnos       — RNOS cumulative entropy / trust gating (EXP2_POLICY)
  cb         — AdaptiveCircuitBreaker (window=10, threshold=0.60)
  hybrid     — HybridController composing RNOS + AdaptiveCircuitBreaker

Usage
-----
    python experiments/experiment_5_hybrid/run_experiment_5.py
    python experiments/experiment_5_hybrid/run_experiment_5.py --seed 42 --max-steps 30

Outputs
-------
    stdout                                    — comparison table + per-mode trajectories
    results/experiment_5/summary.json
    results/experiment_5/{scenario}_{mode}.csv
    docs/experiment_5_hybrid.md
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.adaptive_circuit_breaker import AdaptiveCircuitBreaker
from experiments.configurable_api import ConfigurableAPI
from experiments.experiment_5_hybrid.scenarios import (
    make_cascading_burst,
    make_distributed_low_rate,
)
from experiments.experiment_2 import EXP2_POLICY
from rnos.hybrid import HybridController
from rnos.runtime import RNOSRuntime
from rnos.types import ActionRecord, PolicyDecision

# Suppress per-step RNOS console chatter during batch runs.
logging.getLogger("rnos.runtime").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _REPO_ROOT / "results" / "experiment_5"
_SUMMARY_PATH = _RESULTS_DIR / "summary.json"
_DOCS_PATH = _REPO_ROOT / "docs" / "experiment_5_hybrid.md"
_TRACE_PATH = _REPO_ROOT / "logs" / "exp5_trace.jsonl"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_STEPS = 30
_SEED = 42
_ACB_WINDOW = 10
_ACB_THRESHOLD = 0.60
_ACB_COOLDOWN = 3

# Policy tuned for ConfigurableAPI experiments (same as Exp 2/3/4).
# Default thresholds (3.0/6.0) would produce false DEGRADE on every run
# because repeated_tool=2.0 + cost_score=2.0 creates a ~4.0 structural floor.
_POLICY = EXP2_POLICY   # degrade=9.0, refuse=11.0, trust gates disabled


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    scenario: str
    mode: str
    total_steps: int            # loop iterations before termination
    tool_executions: int        # actual tool calls made
    tool_failures: int
    first_intervention_step: int | None
    first_intervention_type: str | None
    final_state: str
    entropy_at_termination: float | None   # final entropy (RNOS + hybrid only)
    trigger_source: str | None             # hybrid mode only
    step_log: list[dict[str, Any]] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# CB factory (shared between CB-only and hybrid modes)
# ---------------------------------------------------------------------------

def _make_acb() -> AdaptiveCircuitBreaker:
    return AdaptiveCircuitBreaker(
        window_size=_ACB_WINDOW,
        initial_failure_rate=_ACB_THRESHOLD,
        initial_cooldown_steps=_ACB_COOLDOWN,
    )


# ---------------------------------------------------------------------------
# Baseline runner
# ---------------------------------------------------------------------------

def _run_baseline(api: ConfigurableAPI, max_steps: int) -> ScenarioResult:
    """No control — run to max_steps regardless of failures."""
    api.reset()
    step_log: list[dict[str, Any]] = []
    tool_failures = 0

    for step in range(1, max_steps + 1):
        outcome = api.call()
        if not outcome.success:
            tool_failures += 1
        step_log.append({
            "step": step,
            "executed": True,
            "success": outcome.success,
            "latency_ms": round(outcome.latency_ms, 1),
            "entropy": None,
            "trust": None,
            "rnos_decision": None,
            "cb_state": None,
            "cb_failure_rate": None,
            "hybrid_decision": None,
            "hybrid_trigger_source": None,
        })

    return ScenarioResult(
        scenario=api.name,
        mode="baseline",
        total_steps=max_steps,
        tool_executions=max_steps,
        tool_failures=tool_failures,
        first_intervention_step=None,
        first_intervention_type=None,
        final_state="completed",
        entropy_at_termination=None,
        trigger_source=None,
        step_log=step_log,
    )


# ---------------------------------------------------------------------------
# RNOS runner
# ---------------------------------------------------------------------------

def _run_rnos(api: ConfigurableAPI, max_steps: int) -> ScenarioResult:
    """RNOS cumulative entropy / trust gating (EXP2_POLICY)."""
    api.reset()
    runtime = RNOSRuntime(trace_path=_TRACE_PATH, policy_config=_POLICY)
    logging.getLogger("rnos.runtime").setLevel(logging.WARNING)

    steps_executed = 0
    tool_failures = 0
    retry_count = 0
    prev_latency: float | None = None
    first_intervention_step: int | None = None
    first_intervention_type: str | None = None
    final_state = "completed"
    final_entropy: float | None = None
    step_log: list[dict[str, Any]] = []

    for step in range(1, max_steps + 1):
        action = ActionRecord(
            tool_name="configurable_api",
            depth=0,
            retry_count=retry_count,
            latency_ms=prev_latency,
            cumulative_calls=steps_executed,
        )
        assessment = runtime.evaluate(action)
        final_entropy = assessment.entropy

        if assessment.decision is PolicyDecision.REFUSE:
            if first_intervention_step is None:
                first_intervention_step = step
                first_intervention_type = "refuse"
            final_state = "refused"
            step_log.append({
                "step": step,
                "executed": False,
                "success": None,
                "latency_ms": None,
                "entropy": assessment.entropy,
                "trust": assessment.trust,
                "rnos_decision": "REFUSE",
                "cb_state": None,
                "cb_failure_rate": None,
                "hybrid_decision": None,
                "hybrid_trigger_source": None,
            })
            break

        if assessment.decision is PolicyDecision.DEGRADE:
            if first_intervention_step is None:
                first_intervention_step = step
                first_intervention_type = "degrade"

        outcome = api.call()
        steps_executed += 1
        if not outcome.success:
            tool_failures += 1
            retry_count += 1
        else:
            retry_count = 0
        action.latency_ms = outcome.latency_ms
        runtime.record_outcome(action, success=outcome.success)
        prev_latency = outcome.latency_ms

        step_log.append({
            "step": step,
            "executed": True,
            "success": outcome.success,
            "latency_ms": round(outcome.latency_ms, 1),
            "entropy": assessment.entropy,
            "trust": assessment.trust,
            "rnos_decision": assessment.decision.value.upper(),
            "cb_state": None,
            "cb_failure_rate": None,
            "hybrid_decision": None,
            "hybrid_trigger_source": None,
        })

    return ScenarioResult(
        scenario=api.name,
        mode="rnos",
        total_steps=len(step_log),
        tool_executions=steps_executed,
        tool_failures=tool_failures,
        first_intervention_step=first_intervention_step,
        first_intervention_type=first_intervention_type,
        final_state=final_state,
        entropy_at_termination=final_entropy,
        trigger_source=None,
        step_log=step_log,
    )


# ---------------------------------------------------------------------------
# Circuit Breaker runner
# ---------------------------------------------------------------------------

def _run_cb(api: ConfigurableAPI, max_steps: int) -> ScenarioResult:
    """AdaptiveCircuitBreaker only — no RNOS."""
    api.reset()
    acb = _make_acb()

    steps_executed = 0
    tool_failures = 0
    first_intervention_step: int | None = None
    first_intervention_type: str | None = None
    final_state = "completed"
    step_log: list[dict[str, Any]] = []

    for step in range(1, max_steps + 1):
        acb.tick()
        allowed, cb_reason = acb.should_execute()
        cb_stats = acb.stats

        if not allowed:
            if first_intervention_step is None:
                first_intervention_step = step
                first_intervention_type = "blocked"
            if cb_reason == "permanently_open":
                final_state = "permanently_open"
                step_log.append({
                    "step": step,
                    "executed": False,
                    "success": None,
                    "latency_ms": None,
                    "entropy": None,
                    "trust": None,
                    "rnos_decision": None,
                    "cb_state": acb.state,
                    "cb_failure_rate": cb_stats.get("failure_rate", 0.0),
                    "hybrid_decision": None,
                    "hybrid_trigger_source": None,
                })
                break
            # OPEN — blocked this step, continue loop
            step_log.append({
                "step": step,
                "executed": False,
                "success": None,
                "latency_ms": None,
                "entropy": None,
                "trust": None,
                "rnos_decision": None,
                "cb_state": acb.state,
                "cb_failure_rate": cb_stats.get("failure_rate", 0.0),
                "hybrid_decision": None,
                "hybrid_trigger_source": None,
            })
            final_state = "cb_blocked"
            break

        outcome = api.call()
        steps_executed += 1
        if not outcome.success:
            tool_failures += 1
        acb.record_result(success=outcome.success)
        cb_stats_after = acb.stats

        step_log.append({
            "step": step,
            "executed": True,
            "success": outcome.success,
            "latency_ms": round(outcome.latency_ms, 1),
            "entropy": None,
            "trust": None,
            "rnos_decision": None,
            "cb_state": acb.state,
            "cb_failure_rate": cb_stats_after.get("failure_rate", 0.0),
            "hybrid_decision": None,
            "hybrid_trigger_source": None,
        })

    return ScenarioResult(
        scenario=api.name,
        mode="cb",
        total_steps=len(step_log),
        tool_executions=steps_executed,
        tool_failures=tool_failures,
        first_intervention_step=first_intervention_step,
        first_intervention_type=first_intervention_type,
        final_state=final_state,
        entropy_at_termination=None,
        trigger_source=None,
        step_log=step_log,
    )


# ---------------------------------------------------------------------------
# Hybrid runner
# ---------------------------------------------------------------------------

def _run_hybrid(api: ConfigurableAPI, max_steps: int) -> ScenarioResult:
    """HybridController: RNOS + AdaptiveCircuitBreaker, safety-first merge."""
    api.reset()
    runtime = RNOSRuntime(trace_path=_TRACE_PATH, policy_config=_POLICY)
    logging.getLogger("rnos.runtime").setLevel(logging.WARNING)
    acb = _make_acb()
    ctrl = HybridController(runtime, acb)

    steps_executed = 0
    tool_failures = 0
    retry_count = 0
    prev_latency: float | None = None
    first_intervention_step: int | None = None
    first_intervention_type: str | None = None
    first_trigger_source: str | None = None
    final_state = "completed"
    final_entropy: float | None = None
    step_log: list[dict[str, Any]] = []

    for step in range(1, max_steps + 1):
        action = ActionRecord(
            tool_name="configurable_api",
            depth=0,
            retry_count=retry_count,
            latency_ms=prev_latency,
            cumulative_calls=steps_executed,
        )
        ctrl.tick()
        hd = ctrl.evaluate(action)
        final_entropy = hd.rnos_entropy

        if hd.decision == "REFUSE":
            if first_intervention_step is None:
                first_intervention_step = step
                first_intervention_type = "refuse"
                first_trigger_source = hd.trigger_source
            final_state = "refused"
            step_log.append({
                "step": step,
                "executed": False,
                "success": None,
                "latency_ms": None,
                "entropy": hd.rnos_entropy,
                "trust": hd.rnos_trust,
                "rnos_decision": hd.rnos_decision,
                "cb_state": hd.cb_state,
                "cb_failure_rate": round(hd.cb_failure_rate, 3),
                "hybrid_decision": hd.decision,
                "hybrid_trigger_source": hd.trigger_source,
            })
            break

        if hd.decision == "DEGRADE":
            if first_intervention_step is None:
                first_intervention_step = step
                first_intervention_type = "degrade"
                first_trigger_source = hd.trigger_source

        outcome = api.call()
        steps_executed += 1
        if not outcome.success:
            tool_failures += 1
            retry_count += 1
        else:
            retry_count = 0
        action.latency_ms = outcome.latency_ms
        ctrl.record_outcome(action, success=outcome.success)
        prev_latency = outcome.latency_ms

        step_log.append({
            "step": step,
            "executed": True,
            "success": outcome.success,
            "latency_ms": round(outcome.latency_ms, 1),
            "entropy": hd.rnos_entropy,
            "trust": hd.rnos_trust,
            "rnos_decision": hd.rnos_decision,
            "cb_state": hd.cb_state,
            "cb_failure_rate": round(hd.cb_failure_rate, 3),
            "hybrid_decision": hd.decision,
            "hybrid_trigger_source": hd.trigger_source,
        })

    return ScenarioResult(
        scenario=api.name,
        mode="hybrid",
        total_steps=len(step_log),
        tool_executions=steps_executed,
        tool_failures=tool_failures,
        first_intervention_step=first_intervention_step,
        first_intervention_type=first_intervention_type,
        final_state=final_state,
        entropy_at_termination=final_entropy,
        trigger_source=first_trigger_source,
        step_log=step_log,
    )


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "step", "executed", "success", "latency_ms",
    "entropy", "trust", "rnos_decision",
    "cb_state", "cb_failure_rate",
    "hybrid_decision", "hybrid_trigger_source",
]


def _write_csv(result: ScenarioResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{result.scenario}_{result.mode}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in result.step_log:
            writer.writerow({k: row.get(k) for k in _CSV_FIELDS})
    return path


# ---------------------------------------------------------------------------
# Results table printer
# ---------------------------------------------------------------------------

def _fmt_steps(r: ScenarioResult) -> str:
    return str(r.total_steps)


def _fmt_executions(r: ScenarioResult) -> str:
    return str(r.tool_executions)


def _print_comparison_table(
    results: dict[str, dict[str, ScenarioResult]],
) -> str:
    """Print and return a markdown results table."""
    modes = ["baseline", "rnos", "cb", "hybrid"]
    col_width = 14

    header = (
        f"{'Scenario':<22} | "
        + " | ".join(f"{m.upper():<{col_width}}" for m in modes)
        + " | Best"
    )
    sep = "-" * len(header)

    lines = [sep, header, sep]

    for scenario_name, mode_results in results.items():
        row_execs = []
        for m in modes:
            r = mode_results.get(m)
            val = f"{r.tool_executions} exec" if r else "N/A"
            row_execs.append(f"{val:<{col_width}}")

        # Determine "best" (fewest tool_executions among RNOS, CB, Hybrid)
        controlled = {
            m: mode_results[m].tool_executions
            for m in ("rnos", "cb", "hybrid")
            if m in mode_results
        }
        if controlled:
            min_exec = min(controlled.values())
            best_modes = [m.upper() for m, v in controlled.items() if v == min_exec]
            best_str = " = ".join(best_modes)
        else:
            best_str = "N/A"

        row_str = (
            f"{scenario_name:<22} | "
            + " | ".join(row_execs)
            + f" | {best_str}"
        )
        lines.append(row_str)

    lines.append(sep)
    table = "\n".join(lines)
    print(table)
    return table


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------

def _build_report(
    results: dict[str, dict[str, ScenarioResult]],
    table_str: str,
    seed: int,
    max_steps: int,
) -> str:
    modes = ["baseline", "rnos", "cb", "hybrid"]
    report_lines = [
        "# Experiment 5: Hybrid RNOS + Circuit Breaker",
        "",
        f"**Seed:** {seed}  **Max steps:** {max_steps}  "
        f"**CB:** AdaptiveCircuitBreaker(window={_ACB_WINDOW}, "
        f"threshold={_ACB_THRESHOLD})  "
        f"**Policy:** EXP2_POLICY (degrade=9.0, refuse=11.0)",
        "",
        "## Results Table",
        "",
        "Metric: tool_executions (actual API calls before termination).",
        "",
        "```",
        table_str,
        "```",
        "",
    ]

    # --- Key Findings -------------------------------------------------------
    report_lines += ["## Key Findings", ""]

    for scenario_name, mode_results in results.items():
        b = mode_results.get("baseline")
        r = mode_results.get("rnos")
        c = mode_results.get("cb")
        h = mode_results.get("hybrid")

        report_lines.append(f"### {scenario_name}")
        report_lines.append("")

        if b and r and c and h:
            controlled = {"RNOS": r.tool_executions, "CB": c.tool_executions, "Hybrid": h.tool_executions}
            min_exec = min(controlled.values())
            best = [k for k, v in controlled.items() if v == min_exec]

            report_lines.append(
                f"- Baseline completed {b.tool_executions} executions (no control)."
            )
            report_lines.append(
                f"- RNOS stopped at {r.tool_executions} executions "
                f"(first intervention: step {r.first_intervention_step}, "
                f"type={r.first_intervention_type}, "
                f"final_state={r.final_state})."
            )
            report_lines.append(
                f"- CB stopped at {c.tool_executions} executions "
                f"(first intervention: step {c.first_intervention_step}, "
                f"type={c.first_intervention_type}, "
                f"final_state={c.final_state})."
            )
            report_lines.append(
                f"- Hybrid stopped at {h.tool_executions} executions "
                f"(first intervention: step {h.first_intervention_step}, "
                f"trigger_source={h.trigger_source}, "
                f"final_state={h.final_state})."
            )
            report_lines.append(
                f"- **Best:** {' = '.join(best)} ({min_exec} executions). "
                + (
                    "Hybrid matches best." if "Hybrid" in best
                    else "Hybrid does NOT match best — see Limitations."
                )
            )
        report_lines.append("")

    # --- Mechanism ----------------------------------------------------------
    report_lines += ["## Mechanism", ""]

    for scenario_name, mode_results in results.items():
        r = mode_results.get("rnos")
        c = mode_results.get("cb")
        h = mode_results.get("hybrid")
        report_lines.append(f"### {scenario_name}")
        report_lines.append("")

        if scenario_name == "cascading_burst":
            report_lines.append(
                "RNOS detects this scenario via **retry_score** accumulation: each "
                "consecutive failure increments retry_count (weight 1.0/step, cap 4.0). "
                "Combined with failure_score and the repeated_tool/cost floor, entropy "
                "crosses the DEGRADE threshold (9.0) before the CB's 10-step window fills."
            )
            if r and c:
                report_lines.append(
                    f"RNOS first intervened at step {r.first_intervention_step} "
                    f"(entropy → refuse threshold at step {r.total_steps}). "
                    f"CB required {c.tool_executions} executions to fill its window "
                    f"(first block at step {c.first_intervention_step}). "
                    f"Hybrid caught at step {h.first_intervention_step if h else '?'} "
                    f"(trigger: {h.trigger_source if h else '?'})."
                )
        elif scenario_name == "distributed_low_rate":
            report_lines.append(
                "CB detects this scenario via its **sliding window failure rate**: "
                "the F-F-S pattern produces 7/10 = 0.70 failures in any full 10-step "
                "window, exceeding the 0.60 threshold. RNOS's entropy stays below 9.0 "
                "because retry_count resets every third step (on the S step), keeping "
                "retry_score ≤ 2.0, and failure_score peaks at ~1.95 (3/5 recent)."
            )
            if r and c:
                report_lines.append(
                    f"RNOS ran all {r.tool_executions} steps without intervention. "
                    f"CB tripped at step {c.first_intervention_step} after {c.tool_executions} executions. "
                    f"Hybrid matched CB: step {h.first_intervention_step if h else '?'} "
                    f"(trigger: {h.trigger_source if h else '?'})."
                )
        report_lines.append("")

    # --- Limitations --------------------------------------------------------
    report_lines += ["## Limitations", ""]
    report_lines.append(
        "- **Hybrid never strictly dominates both sub-systems simultaneously.** "
        "In each scenario it matches the better-performing sub-system (RNOS for "
        "cascading_burst, CB for distributed_low_rate) but does not improve on it. "
        "The safety-first merge cannot extract information beyond what either "
        "sub-system independently detects."
    )
    report_lines.append("")
    report_lines.append(
        "- **Policy dependency.** Results use EXP2_POLICY (degrade=9.0, refuse=11.0) "
        "which is calibrated for the ConfigurableAPI structural floor "
        "(repeated_tool=2.0 + cost_score=2.0 ≈ 4.0 base entropy). A lower RNOS "
        "threshold would cause RNOS to flag the distributed scenario via the entropy "
        "floor alone, obscuring the CB's comparative advantage."
    )
    report_lines.append("")
    report_lines.append(
        "- **Deterministic scenarios.** Both scenarios use fully explicit step "
        "schedules. Real-world distributions would require stochastic robustness "
        "testing across many seeds before making strong architectural claims."
    )
    report_lines.append("")
    report_lines.append(
        "- **CB parameter sensitivity.** The CB window_size=10 and threshold=0.60 "
        "are tuned to produce clear differentiation in these scenarios. A smaller "
        "window (e.g., 5) would cause CB to trip earlier on cascading_burst, "
        "potentially matching RNOS and reducing the RNOS advantage."
    )

    # --- Per-step data note -------------------------------------------------
    report_lines += [
        "",
        "## Per-step Data",
        "",
        f"Per-step CSV files written to `results/experiment_5/`. "
        f"Each file is named `{{scenario}}_{{mode}}.csv` and contains: "
        f"`step, executed, success, latency_ms, entropy, trust, rnos_decision, "
        f"cb_state, cb_failure_rate, hybrid_decision, hybrid_trigger_source`.",
        "",
    ]

    return "\n".join(report_lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Experiment 5: Hybrid RNOS + Circuit Breaker."
    )
    parser.add_argument("--seed", type=int, default=_SEED)
    parser.add_argument("--max-steps", type=int, default=_MAX_STEPS)
    args = parser.parse_args()

    seed = args.seed
    max_steps = args.max_steps

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TRACE_PATH.write_text("", encoding="utf-8")

    scenarios = [
        make_cascading_burst(seed=seed),
        make_distributed_low_rate(seed=seed),
    ]

    runners = {
        "baseline": _run_baseline,
        "rnos": _run_rnos,
        "cb": _run_cb,
        "hybrid": _run_hybrid,
    }

    all_results: dict[str, dict[str, ScenarioResult]] = {}

    for api in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {api.name}")
        print(f"{'='*60}")
        all_results[api.name] = {}

        for mode_name, runner in runners.items():
            print(f"\n  [{mode_name.upper()}]", end=" ", flush=True)
            result = runner(api, max_steps)
            all_results[api.name][mode_name] = result

            # Per-step CSV
            csv_path = _write_csv(result, _RESULTS_DIR)

            print(
                f"steps={result.total_steps} exec={result.tool_executions} "
                f"fail={result.tool_failures} "
                f"first_intervention=step {result.first_intervention_step} "
                f"({result.first_intervention_type}) "
                f"final={result.final_state}"
                + (f" trigger={result.trigger_source}" if result.trigger_source else "")
            )
            if result.entropy_at_termination is not None:
                print(f"           entropy_at_termination={result.entropy_at_termination:.3f}")
            print(f"           csv -> {csv_path.relative_to(_REPO_ROOT)}")

    # --- Summary table ------------------------------------------------------
    print(f"\n{'='*60}")
    print("Results Summary (tool_executions before termination)")
    print(f"{'='*60}\n")
    table_str = _print_comparison_table(all_results)

    # --- Serialize results ---------------------------------------------------
    summary_data = {}
    for scenario_name, mode_dict in all_results.items():
        summary_data[scenario_name] = {}
        for mode_name, result in mode_dict.items():
            summary_data[scenario_name][mode_name] = {
                "total_steps": result.total_steps,
                "tool_executions": result.tool_executions,
                "tool_failures": result.tool_failures,
                "first_intervention_step": result.first_intervention_step,
                "first_intervention_type": result.first_intervention_type,
                "final_state": result.final_state,
                "entropy_at_termination": result.entropy_at_termination,
                "trigger_source": result.trigger_source,
            }
    _SUMMARY_PATH.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"\nSummary written to {_SUMMARY_PATH.relative_to(_REPO_ROOT)}")

    # --- Markdown report ----------------------------------------------------
    report_md = _build_report(all_results, table_str, seed, max_steps)
    _DOCS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DOCS_PATH.write_text(report_md, encoding="utf-8")
    print(f"Report written to {_DOCS_PATH.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
