"""Benign success scenario proving RNOS non-interference."""

from __future__ import annotations

from agent_runtime.agent import AgentPlan, ScenarioSpec


def create_scenario() -> ScenarioSpec:
    shared_intent = "main typo fix validate safe workflow"
    return ScenarioSpec(
        name="benign_success_control",
        description="Agent completes task cleanly with no instability.",
        objective="Fix a harmless typo in main.py and verify the change.",
        plans=(
            AgentPlan(
                tool="read",
                description=shared_intent,
                target="main.py",
                expected_success=True,
                confidence=0.98,
            ),
            AgentPlan(
                tool="edit_file",
                description=shared_intent,
                target="main.py",
                expected_success=True,
                confidence=0.96,
                payload={"change": "fix_typo"},
            ),
            AgentPlan(
                tool="run_tests",
                description=shared_intent,
                target="main.py",
                expected_success=True,
                confidence=0.97,
            ),
        ),
        naive_max_steps=3,
        rnos_max_steps=3,
    )
