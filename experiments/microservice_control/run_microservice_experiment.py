"""Microservice control experiment runner.

Runs five control modes (baseline, rnos, cb, persistence, hybrid) over three
scenarios and prints per-step output plus a summary results table.

Modes
-----
baseline    — no control (all steps execute)
rnos        — structural entropy gating only
cb          — sliding-window failure-rate breaker only
persistence — long-window drift detector only (uses latency_trend as signal)
hybrid      — tri-modal: max(rnos, cb, persistence)

Scenarios
---------
fanout_cascade — structural fanout/depth growth    (RNOS wins)
retry_storm    — high-density failure burst         (CB wins)
latency_drift  — sustained latency creep            (Persistence wins)

Persistence signal adaptation
------------------------------
The domain-agnostic PersistenceController receives latency_trend (ms/step)
as its second update() argument (nominally "rnos_entropy"). The entropy_floor
is set to 10.0: any step where latency trend > 10 ms/step is counted as
"above floor", enabling detection of persistent latency drift without
requiring structural explosion or elevated failure density.

Usage
-----
    python -m experiments.microservice_control.run_microservice_experiment
    python experiments/microservice_control/run_microservice_experiment.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common.persistence import PersistenceController
from experiments.microservice_control.controllers import (
    RNOSMSController,
    SlidingWindowCBController,
    TriModalMSController,
)
from experiments.microservice_control.scenarios import (
    make_fanout_cascade,
    make_latency_drift,
    make_retry_storm,
)
from experiments.microservice_control.service_model import Decision, RequestState

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_STEPS = 20
CB_WINDOW = 5
CB_THRESHOLD = 0.60
RNOS_DEGRADE = 8.0
RNOS_REFUSE = 10.0
PERSIST_WINDOW = 10
PERSIST_LATENCY_FLOOR = 10.0   # latency_trend > 10 ms/step = "above floor"
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

def _run_baseline(scenario_name: str, states: list[RequestState]) -> ScenarioResult:
    print(f"\n  [baseline] {scenario_name}")
    for s in states:
        print(
            f"    step {s.step:02d}"
            f" | fanout={s.fanout:3d} depth={s.depth:2d}"
            f" lat={s.latency_ms:6.1f}ms trend={s.latency_trend:+6.1f}"
            f" fail_n={s.failures_last_n}"
            f" -> ALLOW (no control)"
        )
    return ScenarioResult(
        scenario=scenario_name,
        mode="baseline",
        executions=len(states),
        first_intervention_step=None,
        final_state="completed",
    )


def _run_rnos(scenario_name: str, states: list[RequestState]) -> ScenarioResult:
    print(f"\n  [rnos] {scenario_name}")
    ctrl = RNOSMSController(degrade_threshold=RNOS_DEGRADE, refuse_threshold=RNOS_REFUSE)
    executions = 0
    first_step = None
    final_state = "completed"

    for s in states:
        assessment = ctrl.evaluate(s)
        executions += 1
        print(
            f"    step {s.step:02d}"
            f" | fanout={s.fanout:3d} depth={s.depth:2d}"
            f" lat={s.latency_ms:6.1f}ms"
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


def _run_cb(scenario_name: str, states: list[RequestState]) -> ScenarioResult:
    print(f"\n  [cb] {scenario_name}")
    ctrl = SlidingWindowCBController(window_size=CB_WINDOW, threshold=CB_THRESHOLD)
    executions = 0
    first_step = None
    final_state = "completed"

    for s in states:
        cb_assessment = ctrl.evaluate()
        executions += 1
        print(
            f"    step {s.step:02d}"
            f" | fanout={s.fanout:3d} depth={s.depth:2d}"
            f" lat={s.latency_ms:6.1f}ms"
            f" cb_state={cb_assessment.state}"
            f" fail_rate={cb_assessment.failure_rate:.3f}"
            f" -> {cb_assessment.decision}"
        )
        if cb_assessment.decision == Decision.REFUSE:
            if first_step is None:
                first_step = s.step
            final_state = "refused"
            ctrl.record_outcome(s.success)
            break
        ctrl.record_outcome(s.success)

    return ScenarioResult(
        scenario=scenario_name,
        mode="cb",
        executions=executions,
        first_intervention_step=first_step,
        final_state=final_state,
    )


def _run_persistence(scenario_name: str, states: list[RequestState]) -> ScenarioResult:
    """Run with persistence controller only.

    RNOS entropy is not used here. latency_trend is fed as the "entropy"
    signal to PersistenceController so that time_above_entropy_floor tracks
    steps with sustained positive latency drift (trend > entropy_floor).
    Only persistence halts execution.
    """
    print(f"\n  [persistence] {scenario_name}")
    persist_ctrl = PersistenceController(
        window_size=PERSIST_WINDOW,
        entropy_floor=PERSIST_LATENCY_FLOOR,
        degrade_threshold=PERSIST_DEGRADE,
        refuse_threshold=PERSIST_REFUSE,
    )
    executions = 0
    first_step = None
    final_state = "completed"

    for s in states:
        p = persist_ctrl.evaluate()
        executions += 1
        print(
            f"    step {s.step:02d}"
            f" | fanout={s.fanout:3d} lat={s.latency_ms:6.1f}ms"
            f" trend={s.latency_trend:+6.1f}"
            f" persist_score={p.score:.3f}(fill={p.window_fill:02d})"
            f" fail_rate={p.rolling_failure_rate:.2f}"
            f" above_floor={p.time_above_entropy_floor:.2f}"
            f" -> {p.decision.upper()}"
        )
        if p.decision != "allow":
            if first_step is None:
                first_step = s.step
            if p.decision == "refuse":
                final_state = "refused"
                persist_ctrl.update(s.success, s.latency_trend)
                break
            else:
                final_state = "degraded"
        persist_ctrl.update(s.success, s.latency_trend)

    return ScenarioResult(
        scenario=scenario_name,
        mode="persistence",
        executions=executions,
        first_intervention_step=first_step,
        final_state=final_state,
    )


def _run_hybrid(scenario_name: str, states: list[RequestState]) -> ScenarioResult:
    """Run with tri-modal hybrid (max of RNOS, CB, Persistence)."""
    print(f"\n  [hybrid/tri-modal] {scenario_name}")
    ctrl = TriModalMSController(
        rnos=RNOSMSController(degrade_threshold=RNOS_DEGRADE, refuse_threshold=RNOS_REFUSE),
        cb=SlidingWindowCBController(window_size=CB_WINDOW, threshold=CB_THRESHOLD),
        persistence=PersistenceController(
            window_size=PERSIST_WINDOW,
            entropy_floor=PERSIST_LATENCY_FLOOR,
            degrade_threshold=PERSIST_DEGRADE,
            refuse_threshold=PERSIST_REFUSE,
        ),
    )
    executions = 0
    first_step = None
    final_state = "completed"

    for s in states:
        a = ctrl.evaluate(s)
        executions += 1
        print(
            f"    step {s.step:02d}"
            f" | fanout={s.fanout:3d} lat={s.latency_ms:6.1f}ms"
            f" H={a.rnos_entropy:5.2f}"
            f" rnos={a.rnos_decision}"
            f" cb={a.cb_decision}({a.cb_state})"
            f" persist={a.persist_decision.upper()}(s={a.persist_score:.2f})"
            f" -> {a.decision} [{a.trigger_source}]"
        )
        if a.decision != Decision.ALLOW:
            if first_step is None:
                first_step = s.step
            if a.decision == Decision.REFUSE:
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
    modes = ["baseline", "rnos", "cb", "persistence", "hybrid"]
    idx: dict[tuple[str, str], ScenarioResult] = {
        (r.scenario, r.mode): r for r in results
    }

    col_w = 14
    header = (
        f"{'Scenario':<22} | "
        + " | ".join(f"{m.upper():<{col_w}}" for m in modes)
        + " | Best"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    success_criteria: dict[str, str] = {
        "fanout_cascade": "rnos",
        "retry_storm":    "cb",
        "latency_drift":  "persistence",
    }

    for scenario in scenarios:
        row = [idx[(scenario, m)] for m in modes]
        exec_strs = [f"{r.executions} exec" for r in row]

        controlled = {m: idx[(scenario, m)].executions for m in ["rnos", "cb", "persistence", "hybrid"]}
        min_exec = min(controlled.values())
        best_modes = [m.upper() for m, v in controlled.items() if v == min_exec]
        best_str = " = ".join(best_modes)

        # Check success criterion: expected controller should match hybrid
        expected = success_criteria.get(scenario, "?")
        hybrid_r = idx[(scenario, "hybrid")]
        expected_r = idx[(scenario, expected)]
        match = "PASS" if hybrid_r.executions == expected_r.executions else "FAIL"

        cells = " | ".join(f"{s:<{col_w}}" for s in exec_strs)
        print(f"{scenario:<22} | {cells} | {best_str}  [{match}]")

    print(sep)


def _print_success_criteria(results: list[ScenarioResult]) -> None:
    idx: dict[tuple[str, str], ScenarioResult] = {
        (r.scenario, r.mode): r for r in results
    }
    checks = [
        ("fanout_cascade", "rnos",        "fanout_cascade -> RNOS = HYBRID"),
        ("retry_storm",    "cb",           "retry_storm    -> CB   = HYBRID"),
        ("latency_drift",  "persistence",  "latency_drift  -> PERSISTENCE = HYBRID"),
    ]
    print("\nSuccess criteria:")
    all_pass = True
    for scenario, expected_mode, label in checks:
        hybrid_exec = idx[(scenario, "hybrid")].executions
        expected_exec = idx[(scenario, expected_mode)].executions
        passed = hybrid_exec == expected_exec
        all_pass = all_pass and passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}")
        if not passed:
            print(
                f"         hybrid={hybrid_exec} exec  "
                f"{expected_mode}={expected_exec} exec"
            )
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME CRITERIA FAILED'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    scenarios = [
        ("fanout_cascade", make_fanout_cascade(MAX_STEPS)),
        ("retry_storm",    make_retry_storm(MAX_STEPS)),
        ("latency_drift",  make_latency_drift(MAX_STEPS)),
    ]

    all_results: list[ScenarioResult] = []

    for scenario_name, states in scenarios:
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*80}")

        all_results.append(_run_baseline(scenario_name, states))
        all_results.append(_run_rnos(scenario_name, states))
        all_results.append(_run_cb(scenario_name, states))
        all_results.append(_run_persistence(scenario_name, states))
        all_results.append(_run_hybrid(scenario_name, states))

    print(f"\n{'='*80}")
    print("RESULTS TABLE  (metric: executions before first REFUSE termination)")
    print(f"{'='*80}")
    _print_results_table(all_results)
    _print_success_criteria(all_results)

    print("\nConfiguration:")
    print(f"  RNOS  DEGRADE / REFUSE        : {RNOS_DEGRADE} / {RNOS_REFUSE}")
    print(f"  CB    window / threshold       : {CB_WINDOW} / {CB_THRESHOLD}")
    print(f"  PERSIST window / thresholds    : {PERSIST_WINDOW} / degrade={PERSIST_DEGRADE} refuse={PERSIST_REFUSE}")
    print(f"  PERSIST latency_trend floor    : {PERSIST_LATENCY_FLOOR} ms/step")
    print(f"  Max steps per scenario         : {MAX_STEPS}")


if __name__ == "__main__":
    main()
