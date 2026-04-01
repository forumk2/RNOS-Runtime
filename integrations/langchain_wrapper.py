"""Optional adapter for plugging RNOS into a LangChain-style workflow."""

from __future__ import annotations

from rnos.runtime import RNOSRuntime
from rnos.types import ActionRecord, RuntimeAssessment


class LangChainRNOSWrapper:
    """Tiny adapter example for future framework integration."""

    def __init__(self, runtime: RNOSRuntime | None = None) -> None:
        self.runtime = runtime or RNOSRuntime()

    def evaluate_tool_call(self, tool_name: str, payload: dict[str, object], depth: int = 0) -> RuntimeAssessment:
        action = ActionRecord(tool_name=tool_name, payload=payload, depth=depth)
        return self.runtime.evaluate(action)
