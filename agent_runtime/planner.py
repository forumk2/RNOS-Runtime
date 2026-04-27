import logging

from .types import Plan, Task


logger = logging.getLogger(__name__)


def _call_llm_for_plan(task: Task) -> list[str]:
    """Deterministic LLM boundary stub for local, repeatable runs."""
    subject = task.prompt.strip() or "the requested development task"
    return [
        f"1. Create a workspace scaffold for: {subject}",
        f"2. Generate a minimal implementation for: {subject}",
        "3. Introduce an invalid experimental patch to exercise RNOS retry",
        "4. Re-apply the unstable patch after retry to test loop detection",
        "5. Re-apply the unstable patch once more to trigger RNOS refusal",
    ]


def plan_task(task: Task) -> Plan:
    steps = _call_llm_for_plan(task)
    logger.info("plan.generated", extra={"steps": len(steps)})
    return Plan(steps=steps)
