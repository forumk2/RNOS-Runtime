"""Base types for agent tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class ToolResult:
    ok: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)


class Tool(Protocol):
    name: str

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool."""
