"""Main agent loop built around the RNOS runtime."""

from __future__ import annotations

from agent.parser import parse_action
from agent.planner import Planner
from rnos.runtime import RNOSRuntime
from rnos.types import PolicyDecision
from tools.calculator import CalculatorTool
from tools.file_ops import FileOpsTool
from tools.unstable_api import UnstableAPITool


class AgentLoop:
    """Minimal agent harness that plans, evaluates, and executes tools."""

    def __init__(self) -> None:
        self.runtime = RNOSRuntime()
        self.planner = Planner()
        self.tools = {
            "calculator": CalculatorTool(),
            "file_ops": FileOpsTool(),
            "unstable_api": UnstableAPITool(),
        }

    def run(self, objective: str, *, max_steps: int = 3) -> list[dict[str, object]]:
        """Execute a short control loop."""

        events: list[dict[str, object]] = []
        for depth in range(max_steps):
            planned_text = self.planner.next_action(objective, history_summary=str(events[-3:]))
            action = parse_action(planned_text, depth=depth)
            assessment = self.runtime.evaluate(action)

            events.append(
                {
                    "stage": "assessment",
                    "tool": action.tool_name,
                    "decision": assessment.decision.value,
                    "entropy": assessment.entropy,
                    "trust": assessment.trust,
                }
            )

            if assessment.decision is PolicyDecision.REFUSE:
                self.runtime.record_outcome(action, success=False)
                break

            tool = self.tools[action.tool_name]
            if assessment.decision is PolicyDecision.DEGRADE:
                action.payload.setdefault("_rnos_constraints", assessment.constraints)

            result = tool.run(**action.payload)
            self.runtime.record_outcome(action, success=result.ok)
            events.append(
                {
                    "stage": "tool_result",
                    "tool": action.tool_name,
                    "ok": result.ok,
                    "message": result.message,
                    "data": result.data,
                }
            )

            if result.ok:
                break

        return events
