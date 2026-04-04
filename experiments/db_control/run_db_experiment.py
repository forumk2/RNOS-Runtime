"""DB control experiment runner.

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
cascading_query_explosion — structural growth (RNOS wins)
lock_contention           — burst failure density (CB wins)
slow_lock_drift           — sustained low-rate failure (Persistence wins)

Usage
-----
    python -m experiments.db_control.run_db_experiment
    python experiments/db_control/run_db_experiment.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common.persistence import PersistenceController
from experiments.db_control.controllers import (
    RNOSDBController,
    SlidingWindowCBController,
    TriModalDBController,
)
from experiments.db_control.query_model import Decision, QueryState
from experiments.db_control.scenarios import (
    make_cascading_query_explosion,
    make_lock_contention,
    make_slow_lock_drift,
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

def _run_baseline(scenario_name: str, states: list[QueryState]) -> ScenarioResult:
    print(f"\n  [baseline] {scenario_name}")
    for s in states:
        print(
            f"    step {s.step:02d} | join={s.join_depth} cost={s.estimated_cost:8.1f}"
            f" lock_wait={s.lock_wait_ms:5.0f}ms success={s.success}"
            f" -> ALLOW (no control)"
        )
    return ScenarioResult(
        scenario=scenario_name,
        mode="baseline",
        executions=len(states),
        first_intervention_step=None,
        final_state="completed",
    )


def _run_rnos(scenario_name: str, states: list[QueryState]) -> ScenarioResult:
    print(f"\n  [rnos] {scenario_name}")
    ctrl = RNOSDBController(degrade_threshold=RNOS_DEGRADE, refuse_threshold=RNOS_REFUSE)
    executions = 0
    first_step = None
    final_state = "completed"

    for s in states:
        assessment = ctrl.evaluate(s)
        executions += 1
        print(
            f"    step {s.step:02d} | join={s.join_depth} cost={s.estimated_cost:8.1f}"
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
        ctrl.record_outcome(s)

    return ScenarioResult(
        scenario=scenario_name,
        mode="rnos",
        executions=executions,
        first_intervention_step=first_step,
        final_state=final_state,
    )


def _run_cb(scenario_name: str, states: list[QueryState]) -> ScenarioResult:
    print(f"\n  [cb] {scenario_name}")
    ctrl = SlidingWindowCBController(window_size=CB_WINDOW, threshold=CB_THRESHOLD)
    executions = 0
    first_step = None
    final_state = "completed"

    for s in states:
        cb_assessment = ctrl.evaluate()
        executions += 1
        print(
            f"    step {s.step:02d} | join={s.join_depth} cost={s.estimated_cost:8.1f}"
            f" cb_state={cb_assessment.state} failure_rate={cb_assessment.failure_rate:.3f}"
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


def _run_persistence(scenario_name: str, states: list[QueryState]) -> ScenarioResult:
    """Run with persistence controller only.

    RNOS entropy is computed internally (to feed the persistence signal) but
    the RNOS decision is never used to halt execution. Only persistence halts.
    """
    print(f"\n  [persistence] {scenario_name}")
    rnos_ctrl = RNOSDBController(degrade_threshold=RNOS_DEGRADE, refuse_threshold=RNOS_REFUSE)
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
            f"    step {s.step:02d} | join={s.join_depth} cost={s.estimated_cost:8.1f}"
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
                rnos_ctrl.record_outcome(s)
                persist_ctrl.update(s.success, rnos_assessment.entropy)
                break
            else:
                final_state = "degraded"
        rnos_ctrl.record_outcome(s)
        persist_ctrl.update(s.success, rnos_assessment.entropy)

    return ScenarioResult(
        scenario=scenario_name,
        mode="persistence",
        executions=executions,
        first_intervention_step=first_step,
        final_state=final_state,
    )


def _run_hybrid(scenario_name: str, states: list[QueryState]) -> ScenarioResult:
    """Run with tri-modal hybrid (max of RNOS, CB, Persistence)."""
    print(f"\n  [hybrid/tri-modal] {scenario_name}")
    ctrl = TriModalDBController(
        rnos=RNOSDBController(degrade_threshold=RNOS_DEGRADE, refuse_threshold=RNOS_REFUSE),
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
            f"    step {s.step:02d} | join={s.join_depth} cost={s.estimated_cost:8.1f}"
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
        ("cascading_query_explosion", make_cascading_query_explosion(MAX_STEPS)),
        ("lock_contention", make_lock_contention(MAX_STEPS)),
        ("slow_lock_drift", make_slow_lock_drift(MAX_STEPS)),
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
    print(f"  RNOS  DEGRADE / REFUSE   : {RNOS_DEGRADE} / {RNOS_REFUSE}")
    print(f"  CB    window / threshold  : {CB_WINDOW} / {CB_THRESHOLD}")
    print(f"  PERSIST window / thresholds: {PERSIST_WINDOW} / degrade={PERSIST_DEGRADE} refuse={PERSIST_REFUSE}")
    print(f"  PERSIST entropy_floor     : {PERSIST_ENTROPY_FLOOR}")
    print(f"  Max steps per scenario    : {MAX_STEPS}")


if __name__ == "__main__":
    main()
