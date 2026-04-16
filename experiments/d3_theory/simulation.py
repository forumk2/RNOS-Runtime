from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from experiments.d3_theory.models import Mode, ModelState


def _p90(values: list[int]) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, int(0.9 * len(ordered)) - 1)
    return ordered[index]


@dataclass
class TraceStep:
    step: int
    mode: str
    regime: str
    branching: int
    validation: int
    disturbance: bool
    clean_probe: bool
    trust_before: float
    trust_after: float
    cumulative_entropy_before: float
    cumulative_entropy_after: float
    unresolved_entropy_before: float
    unresolved_entropy_after: float
    budget_before: int
    budget_after: int
    recovery_before: int
    recovery_after: int
    progress_delta: float
    progress_total: float


@dataclass
class RunResult:
    scenario: str
    model: str
    seed: int
    max_steps: int
    executed_steps: int
    decision_steps: int
    termination_time: int | None
    final_mode: str
    progress: float
    progress_rate: float
    disturbances: int
    successful_recoveries: int
    livelock: bool
    mode_counts: dict[str, int]
    mode_fractions: dict[str, float]
    environment_meta: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] | None = None


def run_simulation(
    scenario: str,
    model: Any,
    environment: Any,
    *,
    seed: int,
    max_steps: int,
    capture_trace: bool = False,
) -> RunResult:
    environment.reset(seed)
    state: ModelState = model.initial_state()
    mode_counts = {mode.value: 0 for mode in Mode}
    disturbances = 0
    progress = 0.0
    successful_recoveries = 0
    executed_steps = 0
    termination_time: int | None = None
    trace_steps: list[dict[str, Any]] | None = [] if capture_trace else None

    for step_index in range(max_steps):
        mode = model.mode_for(state)
        mode_counts[mode.value] += 1

        if mode is Mode.REFUSE:
            termination_time = step_index
            break

        observation = environment.step(mode.value, state, step_index)
        next_state = model.step(state, observation, mode)
        progress_delta = observation.progress_hint if mode is Mode.EXECUTE else 0.0
        progress += progress_delta
        disturbances += int(observation.disturbance)
        executed_steps += 1

        if mode is Mode.SUSPEND and next_state.budget_remaining > state.budget_remaining:
            successful_recoveries += 1

        if trace_steps is not None:
            trace_steps.append(
                asdict(
                    TraceStep(
                        step=step_index,
                        mode=mode.value,
                        regime=observation.regime,
                        branching=observation.branching,
                        validation=observation.validation,
                        disturbance=observation.disturbance,
                        clean_probe=observation.clean_probe,
                        trust_before=state.trust,
                        trust_after=next_state.trust,
                        cumulative_entropy_before=state.cumulative_entropy,
                        cumulative_entropy_after=next_state.cumulative_entropy,
                        unresolved_entropy_before=state.unresolved_entropy,
                        unresolved_entropy_after=next_state.unresolved_entropy,
                        budget_before=state.budget_remaining,
                        budget_after=next_state.budget_remaining,
                        recovery_before=state.recovery_counter,
                        recovery_after=next_state.recovery_counter,
                        progress_delta=progress_delta,
                        progress_total=progress,
                    )
                )
            )

        state = next_state

    decision_steps = sum(mode_counts.values())
    elapsed = termination_time if termination_time is not None else max_steps
    progress_rate = progress / max(elapsed, 1)
    livelock = (
        termination_time is None
        and progress == 0.0
        and mode_counts[Mode.SUSPEND.value] > 0
    )
    final_mode = (
        Mode.REFUSE.value if termination_time is not None else model.mode_for(state).value
    )
    mode_fractions = {
        key: value / decision_steps if decision_steps else 0.0
        for key, value in mode_counts.items()
    }
    return RunResult(
        scenario=scenario,
        model=model.name,
        seed=seed,
        max_steps=max_steps,
        executed_steps=executed_steps,
        decision_steps=decision_steps,
        termination_time=termination_time,
        final_mode=final_mode,
        progress=progress,
        progress_rate=progress_rate,
        disturbances=disturbances,
        successful_recoveries=successful_recoveries,
        livelock=livelock,
        mode_counts=mode_counts,
        mode_fractions=mode_fractions,
        environment_meta=environment.finalize(),
        trace=trace_steps,
    )


def aggregate_runs(runs: list[RunResult]) -> dict[str, Any]:
    if not runs:
        raise ValueError("Cannot aggregate an empty run collection.")

    termination_times = [run.termination_time for run in runs if run.termination_time is not None]
    termination_rate = len(termination_times) / len(runs)
    mode_keys = list(runs[0].mode_fractions)
    mode_fractions = {
        key: statistics.fmean(run.mode_fractions[key] for run in runs)
        for key in mode_keys
    }
    suspension_fractions = [run.mode_fractions.get(Mode.SUSPEND.value, 0.0) for run in runs]
    return {
        "num_runs": len(runs),
        "termination_rate": termination_rate,
        "termination_times": termination_times,
        "mean_termination_time": statistics.fmean(termination_times) if termination_times else None,
        "median_termination_time": statistics.median(termination_times) if termination_times else None,
        "p90_termination_time": _p90(termination_times),
        "mean_progress_rate": statistics.fmean(run.progress_rate for run in runs),
        "mean_progress": statistics.fmean(run.progress for run in runs),
        "mean_disturbances": statistics.fmean(run.disturbances for run in runs),
        "mean_successful_recoveries": statistics.fmean(
            run.successful_recoveries for run in runs
        ),
        "livelock_rate": statistics.fmean(1.0 if run.livelock else 0.0 for run in runs),
        "mean_suspension_fraction": statistics.fmean(suspension_fractions),
        "mean_mode_fractions": mode_fractions,
    }
