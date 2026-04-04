"""CI control experiment runner.

Runs four control modes (baseline, rnos, cb, hybrid) over two scenarios
(pipeline_fanout_explosion, flaky_test_storm) and prints per-step output
plus a summary results table.

Usage
-----
    python -m experiments.ci_control.run_ci_experiment
    python experiments/ci_control/run_ci_experiment.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.ci_control.controllers import (
    HybridCIController,
    RNOSCIController,
    SlidingWindowCBController,
)
from experiments.ci_control.pipeline_model import Decision, PipelineState
from experiments.ci_control.scenarios import (
    make_flaky_test_storm,
    make_pipeline_fanout_explosion,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_STEPS = 20
CB_WINDOW = 5
CB_THRESHOLD = 0.60
RNOS_DEGRADE = 8.0
RNOS_REFUSE = 10.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    scenario: str
    mode: str
    executions: int
    first_intervention_step: int | None
    final_state: str    # "completed" | "refused" | "degraded"


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def _run_baseline(scenario_name: str, states: list[PipelineState]) -> ScenarioResult:
    print(f"\n  [baseline] {scenario_name}")
    for s in states:
        print(
            f"    step {s.step:02d} | active={s.active_jobs:4d}"
            f" spawned={s.total_jobs_spawned:5d}"
            f" retries={s.retry_count:3d}"
            f" success={str(s.success):<5}"
            f" -> ALLOW (no control)"
        )
    return ScenarioResult(
        scenario=scenario_name,
        mode="baseline",
        executions=len(states),
        first_intervention_step=None,
        final_state="completed",
    )


def _run_rnos(scenario_name: str, states: list[PipelineState]) -> ScenarioResult:
    print(f"\n  [rnos] {scenario_name}")
    ctrl = RNOSCIController(degrade_threshold=RNOS_DEGRADE, refuse_threshold=RNOS_REFUSE)
    executions = 0
    first_step = None
    final_state = "completed"

    for s in states:
        assessment = ctrl.evaluate(s)
        executions += 1
        print(
            f"    step {s.step:02d} | active={s.active_jobs:4d}"
            f" spawned={s.total_jobs_spawned:5d}"
            f" retries={s.retry_count:3d}"
            f" entropy={assessment.entropy:5.2f}"
            f" -> {assessment.decision}"
        )
        if assessment.decision != Decision.ALLOW:
            if first_step is None:
                first_step = s.step
            if assessment.decision == Decision.REFUSE:
                final_state = "refused"
                break
            else:
                final_state = "degraded"

    return ScenarioResult(
        scenario=scenario_name,
        mode="rnos",
        executions=executions,
        first_intervention_step=first_step,
        final_state=final_state,
    )


def _run_cb(scenario_name: str, states: list[PipelineState]) -> ScenarioResult:
    print(f"\n  [cb] {scenario_name}")
    ctrl = SlidingWindowCBController(window_size=CB_WINDOW, threshold=CB_THRESHOLD)
    executions = 0
    first_step = None
    final_state = "completed"

    for s in states:
        cb_assessment = ctrl.evaluate()
        executions += 1
        print(
            f"    step {s.step:02d} | active={s.active_jobs:4d}"
            f" spawned={s.total_jobs_spawned:5d}"
            f" retries={s.retry_count:3d}"
            f" cb_state={cb_assessment.state}"
            f" failure_rate={cb_assessment.failure_rate:.3f}"
            f" -> {cb_assessment.decision}"
        )
        if cb_assessment.decision == Decision.REFUSE:
            if first_step is None:
                first_step = s.step
            final_state = "refused"
            break
        ctrl.record_outcome(s.success)

    return ScenarioResult(
        scenario=scenario_name,
        mode="cb",
        executions=executions,
        first_intervention_step=first_step,
        final_state=final_state,
    )


def _run_hybrid(scenario_name: str, states: list[PipelineState]) -> ScenarioResult:
    print(f"\n  [hybrid] {scenario_name}")
    ctrl = HybridCIController(
        rnos=RNOSCIController(degrade_threshold=RNOS_DEGRADE, refuse_threshold=RNOS_REFUSE),
        cb=SlidingWindowCBController(window_size=CB_WINDOW, threshold=CB_THRESHOLD),
    )
    executions = 0
    first_step = None
    final_state = "completed"

    for s in states:
        assessment = ctrl.evaluate(s)
        executions += 1
        print(
            f"    step {s.step:02d} | active={s.active_jobs:4d}"
            f" spawned={s.total_jobs_spawned:5d}"
            f" retries={s.retry_count:3d}"
            f" entropy={assessment.rnos_entropy:5.2f}"
            f" rnos={assessment.rnos_decision}"
            f" cb={assessment.cb_decision}({assessment.cb_state})"
            f" failure_rate={assessment.cb_failure_rate:.3f}"
            f" -> {assessment.decision} trigger={assessment.trigger_source}"
        )
        if assessment.decision != Decision.ALLOW:
            if first_step is None:
                first_step = s.step
            if assessment.decision == Decision.REFUSE:
                final_state = "refused"
                ctrl.record_outcome(s.success)
                break
        ctrl.record_outcome(s.success)

    return ScenarioResult(
        scenario=scenario_name,
        mode="hybrid",
        executions=executions,
        first_intervention_step=first_step,
        final_state=final_state,
    )


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def _print_results_table(results: list[ScenarioResult]) -> None:
    scenarios = list(dict.fromkeys(r.scenario for r in results))
    modes = ["baseline", "rnos", "cb", "hybrid"]
    idx: dict[tuple[str, str], ScenarioResult] = {
        (r.scenario, r.mode): r for r in results
    }

    col_w = 16
    header = (
        f"{'Scenario':<30} | "
        + " | ".join(f"{m.upper():<{col_w}}" for m in modes)
        + " | Best"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for scenario in scenarios:
        row_results = [idx[(scenario, m)] for m in modes]
        exec_strs = [f"{r.executions} exec" for r in row_results]

        controlled = {m: idx[(scenario, m)].executions for m in ["rnos", "cb", "hybrid"]}
        min_exec = min(controlled.values())
        best_modes = [m.upper() for m, v in controlled.items() if v == min_exec]
        best_str = " = ".join(best_modes)

        cells = " | ".join(f"{s:<{col_w}}" for s in exec_strs)
        print(f"{scenario:<30} | {cells} | {best_str}")

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    scenarios = [
        ("pipeline_fanout_explosion", make_pipeline_fanout_explosion(MAX_STEPS)),
        ("flaky_test_storm", make_flaky_test_storm(MAX_STEPS)),
    ]

    all_results: list[ScenarioResult] = []

    for scenario_name, states in scenarios:
        print(f"\n{'='*72}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*72}")

        all_results.append(_run_baseline(scenario_name, states))
        all_results.append(_run_rnos(scenario_name, states))
        all_results.append(_run_cb(scenario_name, states))
        all_results.append(_run_hybrid(scenario_name, states))

    print(f"\n{'='*72}")
    print("RESULTS TABLE  (metric: executions before first REFUSE termination)")
    print(f"{'='*72}")
    _print_results_table(all_results)

    print("\nConfiguration:")
    print(f"  RNOS DEGRADE threshold : {RNOS_DEGRADE}")
    print(f"  RNOS REFUSE  threshold : {RNOS_REFUSE}")
    print(f"  CB window_size         : {CB_WINDOW}")
    print(f"  CB failure threshold   : {CB_THRESHOLD}")
    print(f"  Max steps per scenario : {MAX_STEPS}")


if __name__ == "__main__":
    main()
