"""Deterministic tool execution simulation for the Agent Gate."""

from __future__ import annotations

from dataclasses import dataclass, field

from .agent import AgentPlan
from .risk_model import is_destructive_command


@dataclass(frozen=True)
class ToolExecutionResult:
    tool: str
    success: bool
    output: str
    error: str = ""
    target: str = ""
    command: str = ""
    destructive: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "tool": self.tool,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "target": self.target,
            "command": self.command,
            "destructive": self.destructive,
            "metadata": dict(self.metadata),
        }


class ToolExecutor:
    """Executes planned actions without mutating the real workspace."""

    def execute(self, plan: AgentPlan) -> ToolExecutionResult:
        destructive = plan.tool == "delete" or is_destructive_command(plan.command)

        if destructive:
            return ToolExecutionResult(
                tool=plan.tool,
                success=False,
                output="simulated destructive command was not executed",
                error="destructive command detected",
                target=plan.target,
                command=plan.command,
                destructive=True,
            )

        if not plan.expected_success:
            return ToolExecutionResult(
                tool=plan.tool,
                success=False,
                output=f"{plan.tool} attempted: {plan.description}",
                error=plan.validation_error or "simulated execution failure",
                target=plan.target,
                command=plan.command,
                metadata={"partial_success": plan.partial_success},
            )

        if plan.tool == "run_tests":
            output = "test suite passed"
        elif plan.tool == "edit_file":
            output = f"simulated edit applied to {plan.target}"
        elif plan.tool == "shell_command":
            output = f"simulated shell command completed: {plan.command}"
        elif plan.tool == "read":
            output = f"read-only inspection completed for {plan.target or 'workspace'}"
        else:
            output = f"{plan.tool} completed"

        return ToolExecutionResult(
            tool=plan.tool,
            success=True,
            output=output,
            target=plan.target,
            command=plan.command,
            metadata={"partial_success": plan.partial_success},
        )
