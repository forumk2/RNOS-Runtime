import logging

from . import executor, planner, rnos_adapter, validator
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


def run_task(task: Task) -> list[ExecutionResult]:
    cleanup_workspace()
    plan = planner.plan_task(task)
    history: list[ExecutionResult] = []

    logger.info("runner.plan")
    for step in plan.steps:
        logger.info("  %s", step)

    for step in plan.steps:
        execution = executor.execute_step(step)
        validation = validator.validate()
        history.append(validation)

        decision = rnos_adapter.evaluate_state(history)
        _log_result(step, execution, validation, decision)

        if decision.action == "refuse":
            logger.error("RNOS REFUSAL TRIGGERED")
            break

    return history
