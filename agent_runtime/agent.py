"""Deterministic agent primitives for the RNOS Agent Gate demo.

The classes in this module intentionally model a small but realistic
plan/execute/observe loop. Scenario modules provide ordered plans, and the
runner is responsible for applying RNOS control before each plan executes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


PlanTool = Literal["read", "edit_file", "run_tests", "shell_command", "network", "delete"]


@dataclass(frozen=True)
class AgentPlan:
    """A single planned action produced by an agent before execution."""

    tool: PlanTool
    description: str
    target: str = ""
    command: str = ""
    expected_success: bool = True
    validation_error: str = ""
    confidence: float = 0.95
    partial_success: bool = False
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        parts = [self.tool, self.description, self.target, self.command]
        return " ".join(part for part in parts if part)


@dataclass(frozen=True)
class ScenarioSpec:
    """A deterministic scenario used by the benchmark runner."""

    name: str
    description: str
    objective: str
    plans: tuple[AgentPlan, ...]
    naive_max_steps: int = 10
    rnos_max_steps: int = 10


@dataclass
class AgentState:
    """Mutable loop state observed by the planner and RNOS gate."""

    scenario_name: str
    objective: str
    attempts: int = 0
    retry_count: int = 0
    validation_failures: int = 0
    degraded: bool = False
    stopped: bool = False
    plan_texts: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    failed_edit_targets: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)


class DeterministicScenarioAgent:
    """Agent facade that emits scenario plans in a stable order."""

    def __init__(self, scenario: ScenarioSpec) -> None:
        self.scenario = scenario

    def plan(self, state: AgentState) -> AgentPlan | None:
        if state.attempts >= len(self.scenario.plans):
            return None
        return self.scenario.plans[state.attempts]


def constrain_plan(plan: AgentPlan) -> AgentPlan:
    """Constrain side-effecting plans during DEGRADE mode.

    DEGRADE mode does not blindly continue. The runner converts high-side-effect
    writes and shell commands into read-only diagnostic actions, preserving the
    original intent in the payload for auditability.
    """

    if plan.tool in {"read", "run_tests"}:
        return plan

    return AgentPlan(
        tool="read",
        description=f"constrained diagnostic for: {plan.description}",
        target=plan.target,
        expected_success=plan.expected_success,
        validation_error=plan.validation_error or "constrained action still failed validation",
        confidence=min(plan.confidence, 0.55),
        partial_success=plan.partial_success,
        payload={"constrained_from": plan.tool, "original_command": plan.command},
    )
