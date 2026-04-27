"""Repeated fix/test failure-loop scenario."""

from __future__ import annotations

from agent_runtime.agent import AgentPlan, ScenarioSpec


def create_scenario() -> ScenarioSpec:
    plans = []
    for attempt in range(1, 6):
        plans.append(
            AgentPlan(
                tool="run_tests",
                description=f"run failing unit tests before fix attempt {attempt}",
                command="python -m pytest tests/test_parser.py",
                expected_success=False,
                validation_error="tests still fail: AssertionError expected normalized AST",
                confidence=0.35,
            )
        )
        plans.append(
            AgentPlan(
                tool="edit_file",
                description=f"patch parser normalization attempt {attempt}",
                target="src/parser.py",
                expected_success=False,
                validation_error="edit did not change failing parser behavior",
                confidence=0.4,
                partial_success=attempt == 1,
            )
        )

    return ScenarioSpec(
        name="failure_loop",
        description="Agent repeats test and edit attempts without making structural progress.",
        objective="Fix parser normalization failure.",
        plans=tuple(plans),
        naive_max_steps=10,
        rnos_max_steps=10,
    )
