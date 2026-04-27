import logging
import sys
import time
from dataclasses import replace
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent_runtime import executor, planner, rnos_adapter, runner, validator
from agent_runtime.ast_change_vector import compute_change_vector, summarize_change
from agent_runtime.ast_diff import classify_change, compute_progress
from agent_runtime.ast_similarity import ast_similarity_score
from agent_runtime.cevak import compute_cevak
from agent_runtime.intent_signal import classify_intent, compute_intent_score
from agent_runtime.types import ExecutionResult, Task
from agent_runtime.utils import cleanup_workspace

from metrics import RunMetrics, build_metrics
from report import BenchmarkResult, print_report
from scenarios import Scenario, get_scenarios


MAX_NAIVE_ATTEMPTS = 10


def _read_artifact_source(result: ExecutionResult) -> str | None:
    if not result.artifact_path:
        return None
    try:
        return Path(result.artifact_path).read_text(encoding="utf-8")
    except OSError:
        return None


def _merge_metadata(
    validation: ExecutionResult,
    execution: ExecutionResult,
) -> ExecutionResult:
    return replace(
        validation,
        artifact_path=validation.artifact_path or execution.artifact_path,
        ast_fingerprint=validation.ast_fingerprint or execution.ast_fingerprint,
        ast_similarity_to_previous=execution.ast_similarity_to_previous,
        ast_tokens=validation.ast_tokens or execution.ast_tokens,
        ast_progress_score=execution.ast_progress_score,
        ast_change_type=execution.ast_change_type,
        ast_features=validation.ast_features or execution.ast_features,
        ast_change_vector=execution.ast_change_vector,
        ast_change_summary=execution.ast_change_summary,
        intent_score=execution.intent_score,
        intent_class=execution.intent_class,
        cevak=execution.cevak,
    )


def _enrich_execution(
    execution: ExecutionResult,
    previous_source: str | None,
    current_source: str | None,
    previous_tokens: list[str] | None,
    previous_features: dict[str, int] | None,
) -> ExecutionResult:
    if previous_source is None or current_source is None:
        return execution

    similarity_score = ast_similarity_score(previous_source, current_source)
    progress_score = None
    change_type = None
    change_vector = None
    change_summary = None
    intent_score = None
    intent_class = None

    if previous_tokens is not None and execution.ast_tokens is not None:
        progress_score = compute_progress(previous_tokens, execution.ast_tokens)
        change_type = classify_change(progress_score)

    if previous_features is not None and execution.ast_features is not None:
        change_vector = compute_change_vector(
            previous_features,
            execution.ast_features,
        )
        change_summary = summarize_change(change_vector)

    if progress_score is not None and change_vector is not None:
        intent_score = compute_intent_score(
            similarity_score,
            progress_score,
            change_vector,
        )
        intent_class = classify_intent(
            intent_score,
            progress_score,
            similarity_score,
            change_vector,
        )

    return replace(
        execution,
        ast_similarity_to_previous=similarity_score,
        ast_progress_score=progress_score,
        ast_change_type=change_type,
        ast_change_vector=change_vector,
        ast_change_summary=change_summary,
        intent_score=intent_score,
        intent_class=intent_class,
    )


def _select_step(steps: list[str], attempt_index: int) -> str:
    if attempt_index < len(steps):
        return steps[attempt_index]
    return steps[-1]


def run_naive(task: Task, max_attempts: int = MAX_NAIVE_ATTEMPTS) -> RunMetrics:
    cleanup_workspace()
    plan = planner.plan_task(task)
    history: list[ExecutionResult] = []
    previous_source: str | None = None
    previous_tokens: list[str] | None = None
    previous_features: dict[str, int] | None = None

    started = time.perf_counter()
    for attempt_index in range(max_attempts):
        step = _select_step(plan.steps, attempt_index)
        execution = executor.execute_step(step)
        current_source = _read_artifact_source(execution)
        execution = _enrich_execution(
            execution,
            previous_source,
            current_source,
            previous_tokens,
            previous_features,
        )

        validation = validator.validate()
        validation = _merge_metadata(validation, execution)
        validation = replace(validation, cevak=compute_cevak(validation, history))
        history.append(validation)

        if current_source is not None:
            previous_source = current_source
        if execution.ast_tokens is not None:
            previous_tokens = execution.ast_tokens
        if execution.ast_features is not None:
            previous_features = execution.ast_features

    duration = time.perf_counter() - started
    return build_metrics(
        history=history,
        duration=duration,
        refusals=0,
        total_steps=len(plan.steps),
    )


def run_rnos(task: Task) -> RunMetrics:
    started = time.perf_counter()
    history = runner.run_task(task)
    duration = time.perf_counter() - started
    plan = planner.plan_task(task)
    decision = rnos_adapter.evaluate_state(history) if history else None
    refusals = 1 if decision and decision.action == "refuse" else 0

    return build_metrics(
        history=history,
        duration=duration,
        refusals=refusals,
        total_steps=len(plan.steps),
    )


def run_scenario(scenario: Scenario) -> BenchmarkResult:
    naive_metrics = run_naive(scenario.task)
    rnos_metrics = run_rnos(scenario.task)
    return BenchmarkResult(
        scenario_name=scenario.name,
        naive=naive_metrics,
        rnos=rnos_metrics,
    )


def run_benchmark() -> list[BenchmarkResult]:
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        return [run_scenario(scenario) for scenario in get_scenarios()]
    finally:
        logging.disable(previous_disable_level)


def main() -> None:
    results = run_benchmark()
    print_report(results)


if __name__ == "__main__":
    main()
