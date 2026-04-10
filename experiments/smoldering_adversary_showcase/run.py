"""Run the deterministic smoldering-adversary showcase.

Usage:
    python -m experiments.smoldering_adversary_showcase.run --mode all --seed 42
    python -m experiments.smoldering_adversary_showcase.run --mode hybrid --seed 1337
    python -m experiments.smoldering_adversary_showcase.run --all-seeds
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.adaptive_circuit_breaker import AdaptiveCircuitBreaker
from experiments.smoldering_adversary_showcase.agent import (
    SmolderingAgent,
    SmolderingAgentConfig,
)
from experiments.smoldering_adversary_showcase.scenario import (
    CANONICAL_SEEDS,
    SmolderingObservation,
    SmolderingScenario,
    SmolderingState,
    SyntheticAction,
)
from rnos.hybrid import HybridController
from rnos.policy import PolicyConfig
from rnos.runtime import RNOSRuntime
from rnos.types import ActionRecord

logging.getLogger("rnos.runtime").setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "smoldering_adversary_showcase"

MODE_ORDER = ("baseline", "rnos", "cb", "hybrid")
MODE_COLOURS = {
    "baseline": "#922b21",
    "rnos": "#117864",
    "cb": "#9a7d0a",
    "hybrid": "#1b4f72",
}
PHASE_COLOURS = {
    "blend": "#f4f6f7",
    "lean": "#eaf2f8",
    "drift": "#ebdef0",
    "smolder": "#e8f8f5",
}
SERVICE_COLOURS = {
    "public_api": "#5dade2",
    "auth_service": "#ec7063",
    "cache": "#48c9b0",
    "database": "#7dcea0",
    "worker_pool": "#f5b041",
    "internal_admin_endpoint": "#af7ac5",
    "billing_service": "#f1948a",
}

SHOWCASE_POLICY = PolicyConfig(
    degrade_entropy=7.8,
    refuse_entropy=10.2,
    degrade_trust=-0.1,
    refuse_trust=-0.1,
)
CB_WINDOW = 5
CB_THRESHOLD = 0.60
CB_COOLDOWN = 2


@dataclass(slots=True)
class StepRecord:
    """Per-step record for one controller mode."""

    step: int
    phase: str
    action: str
    primary_target: str | None
    note: str
    decision: str
    trigger_source: str
    planned_calls: int
    executed_calls: int
    refused_calls: int
    entropy: float
    cb_state: str
    cb_failure_rate: float
    instability: float
    weighted_failure_rate: float
    damage_delta: float
    cumulative_damage: float
    top_service: str
    recovery_debt_peak: float
    fatigue_accumulated: float
    drift_damage_fraction: float
    pressure_by_service: dict[str, float] = field(default_factory=dict)
    event: str = ""


@dataclass(slots=True)
class ModeResult:
    """Aggregate result for one controller mode."""

    scenario: str
    mode: str
    seed: int
    budget: int
    stealth: float
    goal_bias: str
    total_simulated_calls: int
    refused_calls: int
    completion_step: int
    peak_entropy: float
    first_refusal_step: int | None
    first_cb_trip_step: int | None
    top_pressured_service: str
    cumulative_damage_score: float
    number_of_pivots_attempted: int
    persistence_phase_succeeded: bool
    recovery_debt_peak: float
    fatigue_accumulated: float
    drift_damage_fraction: float
    step_records: list[StepRecord] = field(default_factory=list)
    service_pressure_totals: dict[str, float] = field(default_factory=dict)


class FailureDensityObserver:
    """Non-blocking observer for first CB-style trip timing."""

    def __init__(self, window_size: int = CB_WINDOW, threshold: float = CB_THRESHOLD) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self._window: deque[bool] = deque(maxlen=window_size)
        self.first_trip_step: int | None = None

    @property
    def failure_rate(self) -> float:
        if not self._window:
            return 0.0
        failures = sum(1 for item in self._window if item is False)
        return failures / len(self._window)

    def record(self, step: int, success: bool) -> None:
        self._window.append(success)
        if self.first_trip_step is not None:
            return
        if len(self._window) < self.window_size:
            return
        if self.failure_rate > self.threshold:
            self.first_trip_step = step


def _make_cb() -> AdaptiveCircuitBreaker:
    return AdaptiveCircuitBreaker(
        window_size=CB_WINDOW,
        initial_failure_rate=CB_THRESHOLD,
        initial_cooldown_steps=CB_COOLDOWN,
    )


def _make_runtime(trace_path: Path) -> RNOSRuntime:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("", encoding="utf-8")
    runtime = RNOSRuntime(trace_path=trace_path, policy_config=SHOWCASE_POLICY)
    logging.getLogger("rnos.runtime").setLevel(logging.WARNING)
    return runtime


def _build_action_record(
    action: SyntheticAction,
    executed_calls: int,
    previous_latency_ms: float | None,
    retry_streak: int,
) -> ActionRecord:
    return ActionRecord(
        tool_name=action.primary_target or "public_api",
        payload={
            "action": action.kind,
            "phase": action.phase,
            "targets": list(action.targets),
        },
        depth=max(0, len(action.path) - 1),
        retry_count=retry_streak,
        latency_ms=previous_latency_ms,
        cumulative_calls=executed_calls,
        metadata={"note": action.note},
    )


def _clamp_factor(
    mode: str,
    decision: str,
    trigger_source: str,
    cb_reason: str,
) -> float:
    if decision == "REFUSE":
        return 0.0
    if mode == "cb" and cb_reason == "half_open_probe":
        return 0.38
    if decision == "DEGRADE":
        if mode == "hybrid":
            if trigger_source == "cb":
                return 0.0
            if trigger_source == "both":
                return 0.0
            return 0.30
        if trigger_source == "cb":
            return 0.42
        if trigger_source == "both":
            return 0.34
        return 0.60
    return 1.0


def _event_line(
    step: int,
    action: SyntheticAction,
    decision: str,
    trigger_source: str,
    entropy: float,
    observation: SmolderingObservation,
    cb_reason: str,
) -> str:
    parts = [
        f"step {step:02d}: {action.label()}",
        f"phase={action.phase}",
        f"decision={decision}",
        f"calls={observation.executed_calls}/{observation.planned_calls}",
        f"entropy={entropy:.2f}",
        f"debt={observation.recovery_debt_peak:.1f}",
        f"fatigue={observation.fatigue_accumulated:.1f}",
        f"damage={observation.damage_delta:.1f}",
    ]
    if trigger_source != "none":
        parts.append(f"trigger={trigger_source}")
    if cb_reason:
        parts.append(f"cb={cb_reason}")
    if observation.note:
        parts.append(observation.note)
    return " | ".join(parts)


def _persistence_succeeded(step_records: list[StepRecord]) -> bool:
    drift_targets = [
        record.primary_target
        for record in step_records
        if record.phase in {"lean", "drift"} and record.executed_calls > 0
    ]
    drift_target = Counter(drift_targets).most_common(1)[0][0] if drift_targets else None
    smolder_records = [
        record for record in step_records if record.phase == "smolder" and record.executed_calls > 0
    ]
    smolder_damage = sum(record.damage_delta for record in smolder_records)
    smolder_targets = {
        record.primary_target for record in smolder_records if record.primary_target
    }
    return (
        len(smolder_records) >= 3
        and smolder_damage >= 10.0
        and bool(smolder_targets - {drift_target})
    )


def _run_mode(
    scenario: SmolderingScenario,
    agent_config: SmolderingAgentConfig,
    mode: str,
) -> ModeResult:
    trace_path = (
        REPO_ROOT
        / "logs"
        / "smoldering_adversary_showcase"
        / f"{mode}_{agent_config.seed}.jsonl"
    )
    entropy_runtime: RNOSRuntime | None = None
    rnos_runtime: RNOSRuntime | None = None
    cb = None
    hybrid = None

    if mode == "baseline":
        entropy_runtime = _make_runtime(trace_path)
    elif mode == "rnos":
        rnos_runtime = _make_runtime(trace_path)
        entropy_runtime = rnos_runtime
    elif mode == "cb":
        entropy_runtime = _make_runtime(trace_path)
        cb = _make_cb()
    elif mode == "hybrid":
        rnos_runtime = _make_runtime(trace_path)
        hybrid = HybridController(rnos_runtime, _make_cb())
        entropy_runtime = rnos_runtime
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    scenario_state: SmolderingState = scenario.make_state()
    agent = SmolderingAgent(scenario, agent_config)
    cb_observer = FailureDensityObserver()

    step_records: list[StepRecord] = []
    total_simulated_calls = 0
    refused_calls = 0
    previous_latency_ms: float | None = None
    retry_streak = 0
    first_refusal_step: int | None = None
    actual_cb_trip_step: int | None = None

    for step in range(1, agent_config.budget + 1):
        action = agent.choose_action(step)
        action_record = _build_action_record(
            action,
            executed_calls=total_simulated_calls,
            previous_latency_ms=previous_latency_ms,
            retry_streak=retry_streak,
        )

        entropy = 0.0
        decision = "ALLOW"
        trigger_source = "none"
        cb_state = "closed"
        cb_failure_rate = cb_observer.failure_rate
        cb_reason = ""

        if mode == "baseline":
            assessment = entropy_runtime.evaluate(action_record)
            entropy = assessment.entropy

        elif mode == "rnos":
            assessment = rnos_runtime.evaluate(action_record)
            entropy = assessment.entropy
            decision = assessment.decision.value.upper()
            if decision != "ALLOW":
                trigger_source = "rnos"

        elif mode == "cb":
            assessment = entropy_runtime.evaluate(action_record)
            entropy = assessment.entropy
            cb.tick()
            allowed, cb_reason = cb.should_execute()
            cb_state = cb.state
            cb_failure_rate = cb.stats.get("failure_rate", 0.0)
            if not allowed:
                decision = "REFUSE"
                trigger_source = "cb"
                if actual_cb_trip_step is None:
                    actual_cb_trip_step = step
            elif cb_reason == "half_open_probe":
                decision = "DEGRADE"
                trigger_source = "cb"

        else:
            hybrid.tick()
            hybrid_decision = hybrid.evaluate(action_record)
            entropy = hybrid_decision.rnos_entropy
            decision = hybrid_decision.decision
            trigger_source = (
                hybrid_decision.trigger_source
                if hybrid_decision.decision != "ALLOW"
                else "none"
            )
            cb_state = hybrid_decision.cb_state
            cb_failure_rate = hybrid_decision.cb_failure_rate
            cb_reason = hybrid_decision.cb_reason
            if cb_reason in {"open_blocked", "permanently_open"} and actual_cb_trip_step is None:
                actual_cb_trip_step = step

        clamp = _clamp_factor(mode, decision, trigger_source, cb_reason)
        observation = scenario.apply_action(action, scenario_state, clamp)

        if decision == "REFUSE" and first_refusal_step is None:
            first_refusal_step = step

        if observation.executed_calls > 0:
            action_record.latency_ms = observation.weighted_latency_ms

            if mode == "baseline":
                entropy_runtime.record_outcome(action_record, success=observation.step_success)
            elif mode == "rnos":
                rnos_runtime.record_outcome(action_record, success=observation.step_success)
            elif mode == "cb":
                entropy_runtime.record_outcome(action_record, success=observation.step_success)
                cb.record_result(success=observation.step_success)
            else:
                hybrid.record_outcome(action_record, success=observation.step_success)

            cb_observer.record(step, observation.step_success)
            previous_latency_ms = observation.weighted_latency_ms
            retry_streak = retry_streak + 1 if not observation.step_success else 0
        else:
            retry_streak = max(0, retry_streak - 1)

        total_simulated_calls += observation.executed_calls
        refused_calls += observation.refused_calls

        if mode in {"baseline", "rnos"}:
            cb_failure_rate = cb_observer.failure_rate

        event = _event_line(
            step,
            action,
            decision,
            trigger_source,
            entropy,
            observation,
            cb_reason,
        )
        step_records.append(
            StepRecord(
                step=step,
                phase=action.phase,
                action=action.label(),
                primary_target=action.primary_target,
                note=action.note,
                decision=decision,
                trigger_source=trigger_source,
                planned_calls=observation.planned_calls,
                executed_calls=observation.executed_calls,
                refused_calls=observation.refused_calls,
                entropy=round(entropy, 3),
                cb_state=cb_state,
                cb_failure_rate=round(cb_failure_rate, 3),
                instability=observation.instability,
                weighted_failure_rate=observation.weighted_failure_rate,
                damage_delta=observation.damage_delta,
                cumulative_damage=observation.cumulative_damage,
                top_service=observation.top_service,
                recovery_debt_peak=observation.recovery_debt_peak,
                fatigue_accumulated=observation.fatigue_accumulated,
                drift_damage_fraction=observation.drift_damage_fraction,
                pressure_by_service=observation.pressure_by_service,
                event=event,
            )
        )

        agent.observe(step, action, observation, decision)

    service_pressure_totals = {
        name: round(scenario_state.cumulative_pressure[name], 3)
        for name in scenario.service_names
    }
    top_pressured_service = max(
        service_pressure_totals,
        key=lambda name: (
            service_pressure_totals[name]
            + scenario_state.recovery_debt[name] * 0.6
            + scenario_state.fatigue[name] * 0.6,
            scenario.services[name].privilege_value,
            name,
        ),
    )
    completion_step = max(
        (record.step for record in step_records if record.executed_calls > 0),
        default=0,
    )

    return ModeResult(
        scenario=scenario.name,
        mode=mode,
        seed=agent_config.seed,
        budget=agent_config.budget,
        stealth=agent_config.stealth,
        goal_bias=agent_config.goal_bias,
        total_simulated_calls=total_simulated_calls,
        refused_calls=refused_calls,
        completion_step=completion_step,
        peak_entropy=max(record.entropy for record in step_records) if step_records else 0.0,
        first_refusal_step=first_refusal_step,
        first_cb_trip_step=actual_cb_trip_step or cb_observer.first_trip_step,
        top_pressured_service=top_pressured_service,
        cumulative_damage_score=round(scenario_state.cumulative_damage, 3),
        number_of_pivots_attempted=agent.pivot_attempts,
        persistence_phase_succeeded=_persistence_succeeded(step_records),
        recovery_debt_peak=round(scenario_state.peak_recovery_debt, 3),
        fatigue_accumulated=round(sum(scenario_state.fatigue.values()), 3),
        drift_damage_fraction=round(
            scenario_state.cumulative_drift_damage / scenario_state.cumulative_damage,
            3,
        )
        if scenario_state.cumulative_damage > 0
        else 0.0,
        step_records=step_records,
        service_pressure_totals=service_pressure_totals,
    )


def _print_metrics_table(results: list[ModeResult]) -> str:
    header = (
        f"{'Mode':<10} {'Calls':>8} {'Refused':>8} {'Complete':>9} "
        f"{'Peak H':>8} {'1st Refuse':>11} {'1st CB':>8} "
        f"{'Top Service':<22} {'Damage':>9} {'Debt':>8} {'Fatigue':>8} {'Drift%':>8}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for result in results:
        lines.append(
            f"{result.mode:<10} {result.total_simulated_calls:>8} {result.refused_calls:>8} "
            f"{result.completion_step:>9} {result.peak_entropy:>8.2f} "
            f"{str(result.first_refusal_step):>11} {str(result.first_cb_trip_step):>8} "
            f"{result.top_pressured_service:<22} {result.cumulative_damage_score:>9.2f} "
            f"{result.recovery_debt_peak:>8.2f} {result.fatigue_accumulated:>8.2f} "
            f"{result.drift_damage_fraction:>8.2f}"
        )
    lines.append(sep)
    table = "\n".join(lines)
    print(table)
    return table


def _write_timeline_csv(
    results: list[ModeResult],
    output_dir: Path,
    service_names: tuple[str, ...],
) -> Path:
    path = output_dir / "timeline.csv"
    fieldnames = [
        "mode",
        "step",
        "phase",
        "action",
        "primary_target",
        "decision",
        "trigger_source",
        "planned_calls",
        "executed_calls",
        "refused_calls",
        "entropy",
        "cb_state",
        "cb_failure_rate",
        "instability",
        "weighted_failure_rate",
        "damage_delta",
        "cumulative_damage",
        "top_service",
        "recovery_debt_peak",
        "fatigue_accumulated",
        "drift_damage_fraction",
        "note",
    ] + [f"pressure_{name}" for name in service_names]

    output_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for record in result.step_records:
                row = {
                    "mode": result.mode,
                    "step": record.step,
                    "phase": record.phase,
                    "action": record.action,
                    "primary_target": record.primary_target,
                    "decision": record.decision,
                    "trigger_source": record.trigger_source,
                    "planned_calls": record.planned_calls,
                    "executed_calls": record.executed_calls,
                    "refused_calls": record.refused_calls,
                    "entropy": record.entropy,
                    "cb_state": record.cb_state,
                    "cb_failure_rate": record.cb_failure_rate,
                    "instability": record.instability,
                    "weighted_failure_rate": record.weighted_failure_rate,
                    "damage_delta": record.damage_delta,
                    "cumulative_damage": record.cumulative_damage,
                    "top_service": record.top_service,
                    "recovery_debt_peak": record.recovery_debt_peak,
                    "fatigue_accumulated": record.fatigue_accumulated,
                    "drift_damage_fraction": record.drift_damage_fraction,
                    "note": record.note,
                }
                for name in service_names:
                    row[f"pressure_{name}"] = record.pressure_by_service.get(name, 0.0)
                writer.writerow(row)
    return path


def _write_events(results: list[ModeResult], output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for result in results:
        path = output_dir / f"events_{result.mode}.txt"
        path.write_text(
            "\n".join(record.event for record in result.step_records) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def _write_summary_json(
    scenario: SmolderingScenario,
    results: list[ModeResult],
    output_dir: Path,
    config: SmolderingAgentConfig,
) -> Path:
    summary = {
        "scenario": scenario.name,
        "seed": config.seed,
        "budget": config.budget,
        "stealth": config.stealth,
        "goal_bias": config.goal_bias,
        "policy": {
            "rnos": asdict(SHOWCASE_POLICY),
            "circuit_breaker": {
                "window_size": CB_WINDOW,
                "threshold": CB_THRESHOLD,
                "cooldown_steps": CB_COOLDOWN,
            },
        },
        "modes": {
            result.mode: {
                "total_simulated_calls": result.total_simulated_calls,
                "refused_calls": result.refused_calls,
                "completion_step": result.completion_step,
                "peak_entropy": result.peak_entropy,
                "first_refusal_step": result.first_refusal_step,
                "first_cb_trip_step": result.first_cb_trip_step,
                "top_pressured_service": result.top_pressured_service,
                "cumulative_damage_score": result.cumulative_damage_score,
                "number_of_pivots_attempted": result.number_of_pivots_attempted,
                "persistence_phase_succeeded": result.persistence_phase_succeeded,
                "recovery_debt_peak": result.recovery_debt_peak,
                "fatigue_accumulated": result.fatigue_accumulated,
                "drift_damage_fraction": result.drift_damage_fraction,
                "service_pressure_totals": result.service_pressure_totals,
            }
            for result in results
        },
    }
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def _write_summary_md(
    scenario: SmolderingScenario,
    results: list[ModeResult],
    output_dir: Path,
    config: SmolderingAgentConfig,
    metrics_table: str,
) -> Path:
    lines = [
        "# Smoldering Adversary Showcase",
        "",
        f"Scenario: `{scenario.name}`",
        f"Seed: `{config.seed}`",
        f"Budget: `{config.budget}`",
        f"Stealth: `{config.stealth}`",
        f"Goal bias: `{config.goal_bias}`",
        "",
        "This run models low-and-slow adaptive pressure on a synthetic service graph. The action set is symbolic and does not perform any real offensive activity.",
        "",
        "## Metrics",
        "",
        "```",
        metrics_table,
        "```",
        "",
        "## Notes",
        "",
        "- `recovery_debt_peak` tracks how much slow-recovering strain accumulated on any service.",
        "- `fatigue_accumulated` records the final background wear left in the graph.",
        "- `drift_damage_fraction` estimates how much total damage came from recovery debt and fatigue rather than one-step pressure.",
        "",
    ]
    path = output_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _add_phase_spans(
    ax: Any,
    scenario: SmolderingScenario,
    budget: int,
    ymax: float,
) -> None:
    for phase, start, end in scenario.phase_spans(budget):
        ax.axvspan(
            start - 0.5,
            end + 0.5,
            color=PHASE_COLOURS[phase],
            alpha=0.30,
            lw=0,
        )
        ax.text(
            (start + end) / 2,
            ymax,
            phase.capitalize(),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#555555",
        )


def _plot_entropy(
    results: list[ModeResult],
    scenario: SmolderingScenario,
    output_dir: Path,
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed - skipping plot generation.")
        return None

    path = output_dir / "timeline_entropy.png"
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ymax = max(
        (record.entropy for result in results for record in result.step_records),
        default=1.0,
    )
    ymax = max(ymax, SHOWCASE_POLICY.refuse_entropy) + 0.8
    _add_phase_spans(ax, scenario, results[0].budget, ymax - 0.25)

    for result in results:
        xs = [record.step for record in result.step_records]
        ys = [record.entropy for record in result.step_records]
        ax.plot(
            xs,
            ys,
            label=result.mode.upper(),
            color=MODE_COLOURS[result.mode],
            linewidth=2.0,
            marker="o",
            markersize=3.0,
        )
        if result.first_refusal_step is not None:
            refusal_record = next(
                record
                for record in result.step_records
                if record.step == result.first_refusal_step
            )
            ax.scatter(
                [refusal_record.step],
                [refusal_record.entropy],
                color=MODE_COLOURS[result.mode],
                s=36,
                zorder=5,
            )

    ax.axhline(
        SHOWCASE_POLICY.degrade_entropy,
        color="#7f8c8d",
        linestyle="--",
        linewidth=1.0,
    )
    ax.axhline(
        SHOWCASE_POLICY.refuse_entropy,
        color="#566573",
        linestyle=":",
        linewidth=1.0,
    )
    ax.set_title("Smoldering adversary entropy timeline")
    ax.set_xlabel("Step")
    ax.set_ylabel("RNOS entropy")
    ax.set_xlim(1, results[0].budget)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_damage(
    results: list[ModeResult],
    scenario: SmolderingScenario,
    output_dir: Path,
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed - skipping plot generation.")
        return None

    path = output_dir / "pressure_damage.png"
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ymax = max(
        (record.cumulative_damage for result in results for record in result.step_records),
        default=1.0,
    ) + 1.0
    _add_phase_spans(ax, scenario, results[0].budget, ymax - 0.35)

    for result in results:
        xs = [record.step for record in result.step_records]
        ys = [record.cumulative_damage for record in result.step_records]
        ax.plot(
            xs,
            ys,
            label=result.mode.upper(),
            color=MODE_COLOURS[result.mode],
            linewidth=2.0,
        )

    ax.set_title("Smoldering adversary cumulative damage")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative damage score")
    ax.set_xlim(1, results[0].budget)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_service_pressure(results: list[ModeResult], output_dir: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed - skipping plot generation.")
        return None

    path = output_dir / "service_pressure_breakdown.png"
    fig, ax = plt.subplots(figsize=(11, 5.8))

    modes = [result.mode for result in results]
    bottoms = [0.0] * len(modes)

    for service in results[0].service_pressure_totals:
        values = [result.service_pressure_totals[service] for result in results]
        ax.bar(
            modes,
            values,
            bottom=bottoms,
            label=service,
            color=SERVICE_COLOURS.get(service, "#95a5a6"),
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    ax.set_title("Cumulative service pressure by mode")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Cumulative simulated calls")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", ncol=2, fontsize=8)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_recovery_debt(
    results: list[ModeResult],
    scenario: SmolderingScenario,
    output_dir: Path,
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed - skipping plot generation.")
        return None

    path = output_dir / "recovery_debt_over_time.png"
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ymax = max(
        (record.recovery_debt_peak for result in results for record in result.step_records),
        default=1.0,
    ) + 0.8
    _add_phase_spans(ax, scenario, results[0].budget, ymax - 0.25)

    for result in results:
        xs = [record.step for record in result.step_records]
        ys = [record.recovery_debt_peak for record in result.step_records]
        ax.plot(
            xs,
            ys,
            label=result.mode.upper(),
            color=MODE_COLOURS[result.mode],
            linewidth=2.0,
        )

    ax.set_title("Recovery debt over time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Peak recovery debt in graph")
    ax.set_xlim(1, results[0].budget)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_seed_comparison(
    output_dir: Path,
    collected: dict[int, list[ModeResult]],
) -> Path:
    path = output_dir / "seed_comparison.md"
    lines = [
        "# Seed comparison",
        "",
        "The showcase is deterministic per seed. These rows summarize the broad shape for the canonical seeds.",
        "",
        "| Seed | Mode | Calls | Refused | Completion | Peak entropy | Damage | Debt | Fatigue | Drift% |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for seed in sorted(collected):
        for result in collected[seed]:
            lines.append(
                f"| {seed} | {result.mode} | {result.total_simulated_calls} | {result.refused_calls} "
                f"| {result.completion_step} | {result.peak_entropy:.2f} "
                f"| {result.cumulative_damage_score:.2f} | {result.recovery_debt_peak:.2f} "
                f"| {result.fatigue_accumulated:.2f} | {result.drift_damage_fraction:.2f} |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _run_for_seed(
    *,
    scenario_name: str,
    mode: str,
    seed: int,
    budget: int,
    stealth: float,
    goal_bias: str,
    output_root: Path,
    no_plots: bool,
) -> list[ModeResult]:
    scenario = SmolderingScenario(name=scenario_name)
    config = SmolderingAgentConfig(
        seed=seed,
        stealth=stealth,
        goal_bias=goal_bias,
        budget=budget,
    )
    output_dir = output_root / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = list(MODE_ORDER) if mode == "all" else [mode]
    results = [_run_mode(scenario, config, current_mode) for current_mode in modes]
    results.sort(key=lambda result: MODE_ORDER.index(result.mode))

    metrics_table = _print_metrics_table(results)
    summary_json = _write_summary_json(scenario, results, output_dir, config)
    timeline_csv = _write_timeline_csv(results, output_dir, scenario.service_names)
    summary_md = _write_summary_md(scenario, results, output_dir, config, metrics_table)
    event_paths = _write_events(results, output_dir)

    print(f"\nArtifacts for seed {seed}:")
    print(f"  summary -> {summary_json.relative_to(REPO_ROOT)}")
    print(f"  timeline -> {timeline_csv.relative_to(REPO_ROOT)}")
    print(f"  notes -> {summary_md.relative_to(REPO_ROOT)}")
    for event_path in event_paths:
        print(f"  event log -> {event_path.relative_to(REPO_ROOT)}")

    if not no_plots and len(results) > 1:
        plot_paths = (
            _plot_entropy(results, scenario, output_dir),
            _plot_damage(results, scenario, output_dir),
            _plot_service_pressure(results, output_dir),
            _plot_recovery_debt(results, scenario, output_dir),
        )
        for generated in plot_paths:
            if generated is not None:
                print(f"  plot -> {generated.relative_to(REPO_ROOT)}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deterministic smoldering-adversary showcase."
    )
    parser.add_argument(
        "--scenario",
        choices=["smoldering_adversary", "slow_drift_agent", "smoldering_pressure_showcase"],
        default="smoldering_adversary",
        help="Scenario alias. All aliases resolve to the same synthetic showcase.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "baseline", "rnos", "cb", "hybrid"],
        default="all",
        help="Controller mode to run. Use 'all' for comparative artifacts.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed.")
    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="Run the canonical seed set: 7, 42, and 1337.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=30,
        help="Maximum number of symbolic agent steps.",
    )
    parser.add_argument(
        "--stealth",
        type=float,
        default=0.70,
        help="Stealth bias in [0, 1]. Higher means lower-and-slower pressure.",
    )
    parser.add_argument(
        "--goal-bias",
        choices=["privilege", "disruption"],
        default="disruption",
        help="Synthetic objective bias for target scoring.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base output directory for generated artifacts.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_name = "smoldering_adversary"
    seeds = list(CANONICAL_SEEDS) if args.all_seeds else [args.seed]
    collected: dict[int, list[ModeResult]] = {}

    print("\n=== Smoldering Adversary Showcase ===")
    print(f"scenario={scenario_name} mode={args.mode} budget={args.budget}")
    print(f"stealth={args.stealth:.2f} goal_bias={args.goal_bias}")
    print(
        f"rnos_policy=({SHOWCASE_POLICY.degrade_entropy}, {SHOWCASE_POLICY.refuse_entropy})"
    )
    print(f"cb=window {CB_WINDOW}, threshold {CB_THRESHOLD}, cooldown {CB_COOLDOWN}\n")

    for seed in seeds:
        print(f"Seed {seed}")
        collected[seed] = _run_for_seed(
            scenario_name=scenario_name,
            mode=args.mode,
            seed=seed,
            budget=args.budget,
            stealth=args.stealth,
            goal_bias=args.goal_bias,
            output_root=args.output_dir,
            no_plots=args.no_plots,
        )
        print()

    if len(seeds) > 1:
        comparison = _write_seed_comparison(args.output_dir, collected)
        print(f"Seed comparison -> {comparison.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
