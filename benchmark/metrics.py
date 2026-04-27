from dataclasses import dataclass

from agent_runtime.types import ExecutionResult


@dataclass(frozen=True)
class RunMetrics:
    attempts: int
    failures: int
    successes: int
    refusals: int
    total_steps: int
    repeated_failures: int
    duration: float
    succeeded: bool


def compute_wasted_attempts(history: list[ExecutionResult]) -> int:
    wasted = 0
    for result in history:
        similarity = result.ast_similarity_to_previous or 0.0
        progress = result.ast_progress_score or 0.0
        if not result.success and similarity > 0.85 and progress < 0.1:
            wasted += 1
    return wasted


def build_metrics(
    history: list[ExecutionResult],
    duration: float,
    refusals: int,
    total_steps: int,
) -> RunMetrics:
    successes = sum(1 for result in history if result.success)
    failures = sum(1 for result in history if not result.success)
    return RunMetrics(
        attempts=len(history),
        failures=failures,
        successes=successes,
        refusals=refusals,
        total_steps=total_steps,
        repeated_failures=compute_wasted_attempts(history),
        duration=duration,
        succeeded=bool(history and history[-1].success and failures == 0),
    )
