from dataclasses import replace
import logging
from pathlib import Path

from . import executor, planner, rnos_adapter, validator
from .ast_change_vector import compute_change_vector, summarize_change
from .ast_diff import classify_change, compute_progress
from .ast_similarity import ast_similarity_score
from .types import ExecutionResult, RNOSDecision, Task
from .utils import cleanup_workspace


logger = logging.getLogger(__name__)


def _log_result(step: str, execution: ExecutionResult, validation: ExecutionResult, decision: RNOSDecision) -> None:
    logger.info("runner.step=%s", step)
    logger.info(
        "runner.execution success=%s output=%s error=%s",
        execution.success,
        execution.output,
        execution.error,
    )
    logger.info(
        "runner.validation success=%s output=%s error=%s",
        validation.success,
        validation.output,
        validation.error,
    )
    logger.info(
        "runner.rnos action=%s reason=%s instability=%.2f",
        decision.action,
        decision.reason,
        decision.entropy_score,
    )


def _read_artifact_source(result: ExecutionResult) -> str | None:
    if not result.artifact_path:
        return None

    try:
        return Path(result.artifact_path).read_text(encoding="utf-8")
    except OSError:
        return None


def _with_artifact_metadata(
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
    )


def run_task(task: Task) -> list[ExecutionResult]:
    cleanup_workspace()
    plan = planner.plan_task(task)
    history: list[ExecutionResult] = []
    previous_source: str | None = None
    previous_tokens: list[str] | None = None
    previous_features: dict[str, int] | None = None

    logger.info("runner.plan")
    for step in plan.steps:
        logger.info("  %s", step)

    for step in plan.steps:
        execution = executor.execute_step(step)
        current_source = _read_artifact_source(execution)
        if previous_source is not None and current_source is not None:
            progress_score = None
            change_type = None
            change_vector = None
            change_summary = None
            if previous_tokens is not None and execution.ast_tokens is not None:
                progress_score = compute_progress(previous_tokens, execution.ast_tokens)
                change_type = classify_change(progress_score)
            if previous_features is not None and execution.ast_features is not None:
                change_vector = compute_change_vector(
                    previous_features,
                    execution.ast_features,
                )
                change_summary = summarize_change(change_vector)

            execution = replace(
                execution,
                ast_similarity_to_previous=ast_similarity_score(
                    previous_source,
                    current_source,
                ),
                ast_progress_score=progress_score,
                ast_change_type=change_type,
                ast_change_vector=change_vector,
                ast_change_summary=change_summary,
            )

        validation = validator.validate()
        validation = _with_artifact_metadata(validation, execution)
        history.append(validation)

        if current_source is not None:
            previous_source = current_source
        if execution.ast_tokens is not None:
            previous_tokens = execution.ast_tokens
        if execution.ast_features is not None:
            previous_features = execution.ast_features

        decision = rnos_adapter.evaluate_state(history)
        _log_result(step, execution, validation, decision)

        if decision.action == "refuse":
            logger.error("RNOS REFUSAL TRIGGERED")
            break

    return history
