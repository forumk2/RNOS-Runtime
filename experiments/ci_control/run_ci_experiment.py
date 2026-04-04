"""CI control experiment runner.

Runs five control modes (baseline, rnos, cb, persistence, hybrid) over three
scenarios and prints per-step output plus a summary results table.

Modes
-----
baseline    — no control (all steps execute)
rnos        — structural entropy gating only
cb          — sliding-window failure-rate breaker only
persistence — long-window drift detector only (uses RNOS entropy as input signal)
hybrid      — tri-modal: max(rnos, cb, persistence)

Scenarios
---------
pipeline_fanout_explosion — structural fanout growth (RNOS wins)
flaky_test_storm          — burst failure density (CB wins)
gradual_flakiness         — sustained low-rate failure (Persistence wins)

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

from experiments.common.persistence import PersistenceController
from experiments.ci_control.controllers import (
    RNOSCIController,
    SlidingWindowCBController,
    TriModalCIController,
)
from experiments.ci_control.pipeline_model import Decision, PipelineState
from experiments.ci_control.scenarios import (
    make_flaky_test_storm,
    make_gradual_flakiness,
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
PERSIST_WINDOW = 10
PERSIST_ENTROPY_FLOOR = 3.0
PERSIST_DEGRADE = 0.30
PERSIST_REFUSE = 0.50


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


def _run_persistence(scenario_name: str, states: list[PipelineState]) -> ScenarioResult:
    """Run with persistence controller only.

    RNOS entropy is computed internally (to feed the persistence signal) but
    the RNOS decision is never used to halt execution. Only persistence halts.
    """
    print(f"\n  [persistence] {scenario_name}")
    rnos_ctrl = RNOSCIController(degrade_threshold=RNOS_DEGRADE, refuse_threshold=RNOS_REFUSE)
    persist_ctrl = PersistenceController(
        window_size=PERSIST_WINDOW,
        entropy_floor=PERSIST_ENTROPY_FLOOR,
        degrade_threshold=PERSIST_DEGRADE,
        refuse_threshold=PERSIST_REFUSE,
    )
    executions = 0
    first_step = None
    final_state = "completed"

    for s in states:
        rnos_assessment = rnos_ctrl.evaluate(s)
        p = persist_ctrl.evaluate()
        executions += 1
        print(
            f"    step {s.step:02d} | active={s.active_jobs:4d}"
            f" spawned={s.total_jobs_spawned:5d}"
            f" retries={s.retry_count:3d}"
            f" entropy={rnos_assessment.entropy:5.2f}"
            f" persist_score={p.score:.3f}(fill={p.window_fill:02d})"
            f" fail_rate={p.rolling_failure_rate:.2f}"
            f" -> {p.decision.upper()}"
        )
        if p.decision != "allow":
            if first_step is None:
                first_step = s.step
            if p.decision == "refuse":
                final_state = "refused"
                persist_ctrl.update(s.success, rnos_assessment.entropy)
                break
            else:
                final_state = "degraded"
        persist_ctrl.update(s.success, rnos_assessment.entropy)

    return ScenarioResult(
        scenario=scenario_name,
        mode="persistence",
        executions=executions,
        first_intervention_step=first_step,
        final_state=final_state,
    )


def _run_hybrid(scenario_name: str, states: list[PipelineState]) -> ScenarioResult:
    """Run with tri-modal hybrid (max of RNOS, CB, Persistence)."""
    print(f"\n  [hybrid/tri-modal] {scenario_name}")
    ctrl = TriModalCIController(
        rnos=RNOSCIController(degrade_threshold=RNOS_DEGRADE, refuse_threshold=RNOS_REFUSE),
        cb=SlidingWindowCBController(window_size=CB_WINDOW, threshold=CB_THRESHOLD),
        persistence=PersistenceController(
            window_size=PERSIST_WINDOW,
            entropy_floor=PERSIST_ENTROPY_FLOOR,
            degrade_threshold=PERSIST_DEGRADE,
            refuse_threshold=PERSIST_REFUSE,
        ),
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
            f" persist={assessment.persist_decision.upper()}(s={assessment.persist_score:.2f})"
            f" -> {assessment.decision} trigger={assessment.trigger_source}"
        )
        if assessment.decision != Decision.ALLOW:
            if first_step is None:
                first_step = s.step
            if assessment.decision == Decision.REFUSE:
                final_state = "refused"
                ctrl.record_outcome(s, success=s.success)
                break
        ctrl.record_outcome(s, success=s.success)

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
    modes = ["baseline", "rnos", "cb", "persistence", "hybrid"]
    idx: dict[tuple[str, str], ScenarioResult] = {
        (r.scenario, r.mode): r for r in results
    }

    col_w = 14
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

        controlled = {m: idx[(scenario, m)].executions for m in ["rnos", "cb", "persistence", "hybrid"]}
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
        ("gradual_flakiness", make_gradual_flakiness(MAX_STEPS)),
    ]

    all_results: list[ScenarioResult] = []

    for scenario_name, states in scenarios:
        print(f"\n{'='*72}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*72}")

        all_results.append(_run_baseline(scenario_name, states))
        all_results.append(_run_rnos(scenario_name, states))
        all_results.append(_run_cb(scenario_name, states))
        all_results.append(_run_persistence(scenario_name, states))
        all_results.append(_run_hybrid(scenario_name, states))

    print(f"\n{'='*72}")
    print("RESULTS TABLE  (metric: executions before first REFUSE termination)")
    print(f"{'='*72}")
    _print_results_table(all_results)

    print("\nConfiguration:")
    print(f"  RNOS  DEGRADE / REFUSE    : {RNOS_DEGRADE} / {RNOS_REFUSE}")
    print(f"  CB    window / threshold   : {CB_WINDOW} / {CB_THRESHOLD}")
    print(f"  PERSIST window / thresholds: {PERSIST_WINDOW} / degrade={PERSIST_DEGRADE} refuse={PERSIST_REFUSE}")
    print(f"  PERSIST entropy_floor      : {PERSIST_ENTROPY_FLOOR}")
    print(f"  Max steps per scenario     : {MAX_STEPS}")


if __name__ == "__main__":
    main()
